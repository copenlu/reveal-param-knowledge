import os
import json
import torch
import collections
import argparse
import numpy as np
import random
import re, glob

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, OPTForSequenceClassification, GPTNeoXForSequenceClassification
from transformers import BloomForCausalLM, BloomForSequenceClassification, BloomConfig
from transformers import OPTForCausalLM

from transformers import AdamW, get_linear_schedule_with_warmup

# https://huggingface.co/docs/transformers/model_doc/bloom
# from modeling_bloom import BloomForSequenceClassification

from utils_datasets import *

import pdb

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        print('No GPU available, using the CPU instead.')


def load_dataset(tokenizer, args, mode="train"):
    if args.dataset in ["fever", "snli", "csqa", "mnli"]:
        filepath = mode

    else:
        if mode == "train":
            filepath = "../data/%s/train.json" % args.dataset

        elif mode == "dev":
            filepath = "../data/%s/train.json" % args.dataset

        elif mode == "test":
            filepath = "../data/%s/dev.json" % args.dataset

    dataset = fcDataset(filepath=filepath, tokenizer=tokenizer, args=args, mode=mode)

    return dataset


def load_model(args, num_labels, trained_weight_path=None):
    if args.do_eval:
        model_weight_path = trained_weight_path
    else:
        model_weight_path = args.model_name_or_path

    if args.dataset == "csqa":
        num_labels = 1


    if "opt" in args.model_name_or_path:
        # to generate answer for csqa
        # if args.dataset == "csqa":
        #     # model = OPTForCausalLM.from_pretrained(model_weight_path)
        #     model = OPTForSequenceClassification.from_pretrained(model_weight_path, num_labels=1)
        # else:
        model = OPTForSequenceClassification.from_pretrained(model_weight_path, num_labels=num_labels)

    elif "pythia" in args.model_name_or_path:
        model = GPTNeoXForSequenceClassification.from_pretrained(model_weight_path, num_labels=num_labels)

    elif "bloom" in args.model_name_or_path:
        # to generate answer for csqa
        # if args.dataset == "csqa":
        #     # model = BloomForCausalLM.from_pretrained(model_weight_path)
        #     model = BloomForSequenceClassification.from_pretrained(model_weight_path, num_labels=1)
        # else:
        model = BloomForSequenceClassification.from_pretrained(model_weight_path, num_labels=num_labels)

    model.config.use_cache = False
    model.config.output_hidden_states=True

    return model


def eval_with_trainer(test_dataset, model, tokenizer, args):

    training_args = TrainingArguments(
        output_dir="./tmp_dir",
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.test_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        do_train=False,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        learning_rate=args.learning_rate,
        fp16=True
    )

    if "pythia" in args.model_name_or_path:
        trainer = Trainer(
            model,
            args=training_args,
            eval_dataset=test_dataset,
            preprocess_logits_for_metrics=logit_preprocess
        )

    else:
        trainer = Trainer(
            model,
            args=training_args,
            eval_dataset=test_dataset
        )

    results = trainer.predict(test_dataset=test_dataset)

    result = metrics_trainer_eval(results, args)

    return result


def eval_loop(train_dataset, model, args):

    train_loader = DataLoader(train_dataset, batch_size=1)

    model.eval()

    train_iterator = tqdm(train_loader, desc="Gathering Info")

    last_hidden_list = []
    loss_list = []
    for step, batch in enumerate(train_iterator):
        with torch.no_grad():

            batch = tuple(b.to(args.device) for _, b in batch.items())
            output = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[-1])

            try:
                last_pos = (batch[1]==0).nonzero(as_tuple=False)[0, 1].item() -1
            except:
                last_pos = len(batch[0]) -1

            # if not args.inf_baseline:
            loss_list.append(output.loss.item())

            last_hidden_last_token = output.hidden_states[-1][:, last_pos, :].detach()
            last_hidden_list.append(last_hidden_last_token)


    diverse_result = cal_kl_divergence(last_hidden_list, args)

    # if not args.inf_baseline:
    diverse_result["loss_sum"] = np.sum(loss_list)
    diverse_result["loss_mean"] = np.mean(loss_list)

    return diverse_result



# end of eval_loop


def cal_kl_divergence(last_hidden_list, args):


    # kl_func = nn.KLDivLoss(reduction="batchmean", log_target=True)

    cos = nn.CosineSimilarity(dim=1)

    hidden_iterator = tqdm(last_hidden_list, desc="Calculating KL Div")

    kl_div_list = []
    cosine_sim_list = []
    for idx, last_hidden in enumerate(hidden_iterator):

        for b_idx in range(idx+1, len(last_hidden_list)):

            # if args.dataset == "csqa":
            #     p = F.log_softmax(last_hidden, 0)
            #     q = F.log_softmax(last_hidden_list[b_idx], 0)
            #
            # else:
            #     p = F.log_softmax(last_hidden, 1)
            #     q = F.log_softmax(last_hidden_list[b_idx], 1)
            #
            # output = kl_func(p, q).item()
            # kl_div_list.append(output)
            if args.dataset == "csqa":
                output = [i.item() for i in cos(last_hidden, last_hidden_list[b_idx])]
                cosine_sim_list += output
            else:
                output = cos(last_hidden, last_hidden_list[b_idx]).item()
                cosine_sim_list.append(output)

    print("# of computation:", len(cosine_sim_list))

    kl_result = {
        "cos_sum": np.sum(cosine_sim_list),
        "cos_mean": np.mean(cosine_sim_list)
    }

    return kl_result


def logit_preprocess(logits, labels=None):
    return logits[0]


def eval_custom(test_dataset, model, tokenizer, args):
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    print("***** Running Evaluation *****")
    print("  Num examples = %d" % len(test_loader))
    print("  Batch size = %d" % args.test_batch_size)

    model.eval()

    test_iterator = tqdm(test_loader, desc="Test Evaluating")

    # listwise / num_label = 1
    # if "pythia" in args.model_name_or_path:
    softmax = nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    # else:
    # softmax = nn.Softmax(dim=0)

    # binary classification / num_label = 2
    # softmax = torch.nn.Softmax(dim=0)

    last_hidden_list = []
    loss_list = []
    for step, batch in enumerate(test_iterator):
        with torch.no_grad():
            batch = tuple(b.to(args.device) for b in batch)

            output = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))

            batch_last_hidden = []
            for idx, seq in enumerate(batch[1].squeeze(0)):
                try:
                    last_pos = (seq==0).nonzero(as_tuple=False)[0, 0].item() -1
                except:
                    last_pos = len(seq) -1

                selected_last = output.hidden_states[-1][idx, last_pos, :]
                batch_last_hidden.append(selected_last)

            # num_label = 1 (list wise?)
            pred_probs = softmax(output.logits.squeeze(1).unsqueeze(0))

            loss = loss_fn(pred_probs, batch[-1]).item()
            loss_list.append(loss)

            last_hidden_list.append(torch.stack(batch_last_hidden))


    diverse_result = cal_kl_divergence(last_hidden_list, args)

    # if not args.inf_baseline:
    diverse_result["loss_sum"] = np.sum(loss_list)
    diverse_result["loss_mean"] = np.mean(loss_list)

    return diverse_result
# eval_custom



def metrics_trainer_eval(results, args):

    if isinstance(results.predictions, tuple):
        preds = results.predictions[0]
    else:
        preds = results.predictions

    gold_label_list = results.label_ids

    acc = []
    wrong_pred_indices = []

    choice_pred = []
    answer = None
    for idx, (gt, pd) in enumerate(zip(gold_label_list, preds)):

        if args.dataset == "csqa":
            if len(choice_pred) == 5:
                assert answer is not None
                # pdb.set_trace()

                answer = None

            choice_pred.append(pd.max())

            if gt == 1:
                answer = len(choice_pred)

        else:
            pd_label = torch.argmax(torch.tensor(pd)).item()

            # neutral -> also contradiction
            if pd_label == 0 and args.dataset == "hans":
                pd_label = 1

            if pd_label == gt:
                acc.append(1.0)
            else:
                acc.append(0.0)
                wrong_pred_indices.append(idx)


    # for the last question
    # if args.dataset == "csqa":
    #     pass

    if args.inf_filtering:
        print("%.1f of the training dataset" %(args.inf_inst_num))

    print("Accuracy:", np.mean(acc))

    if args.dataset == "hans":
        json.dump(wrong_pred_indices, open("hans_wrong_%s.json" %args.model_name_or_path.split("/")[-1], "w"), indent=4)

    return np.mean(acc)
# end of metrics_trainer_eval


def read_args():
    # arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dataset", type=str, default="averitec", help="dataset_name")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name_or_path", type=str, default="opt-1.3b")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--inf_filtering", action="store_true")
    parser.add_argument("--inf_baseline", action="store_true")
    parser.add_argument("--inf_inst_num", type=float, default=300)
    parser.add_argument("--inf_file", type=str, default="top_inst_inf_func_139.json")
    parser.add_argument("--most_least", type=str, default="most", help="while inf training, which instances to select")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_dev", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)

    # for custom training (without trainer)
    parser.add_argument("--warmup_step", type=int, default=0, help="step of linear warmup")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")


    training_args = parser.parse_args()

    training_args.num_gpu = torch.cuda.device_count()

    return training_args


def lexical_analysis(train_dataset, tokenizer, args):

    vocab_set = set()
    sentence_length = [] # input sequence length

    if args.dataset == "csqa":
        target_example_list = train_dataset.features
    else:
        target_example_list = train_dataset.filtered_examples

    for example in target_example_list:

        if args.dataset in ["averitec", "fever"]:
            claim = tokenizer.tokenize("Claim : %s" % example.claim)
            evidence = tokenizer.tokenize("/ Evidence : %s" % example.evidence)
            answer = tokenizer.tokenize("/ Answer :")

        elif args.dataset in ["snli", "hans", "mnli"]:
            claim = tokenizer.tokenize("Premise : %s" % example.claim)
            evidence = tokenizer.tokenize("/ Hypothesis : %s" % example.evidence)
            answer = tokenizer.tokenize("/ Answer :")

        elif args.dataset == "csqa":
            claim = tokenizer.tokenize("Question : %s" % example.claim)
            evidence = tokenizer.tokenize("/ Choices : %s" % example.evidence)
            answer = tokenizer.tokenize("/ Answer :")

        input_seq_len = len(claim) + len(evidence) + len(answer)

        sentence_length.append(input_seq_len)

        vocab_set = vocab_set.union(set(claim))
        vocab_set = vocab_set.union(set(evidence))

    return len(vocab_set), np.mean(sentence_length)


def main(args):

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = load_dataset(tokenizer, args, mode="train")

    # vocabulary set / sentence length
    num_vocab, avg_seq_len = lexical_analysis(train_dataset, tokenizer, args)

    # hidden representation & perplexity
    # load trained model
    model = load_model(args, num_labels=len(train_dataset.label_dict), trained_weight_path=args.output_dir)
    model.to(args.device)

    if "pythia" in args.model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.encode("<pad>")[0]

    if args.dataset == "csqa":
        diverse_result = eval_custom(train_dataset, model, tokenizer, args)
    else:
        diverse_result = eval_loop(train_dataset, model, args)

    print(args.dataset, args.model_name_or_path)

    if args.inf_baseline:
        print("Results from randomly selected training instances")

    print("\n\n%s\n" %args.inf_file)

    diverse_result["vocab"] = num_vocab
    diverse_result["avg_seq_len"] = avg_seq_len

    for key, value in diverse_result.items():
        print("%.3f" %(value))


if __name__ == "__main__":

    args = read_args()

    set_seed(args)

    main(args)

