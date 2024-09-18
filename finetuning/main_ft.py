import os
import json
import torch
import collections
import argparse
import numpy as np
import random
import re, glob

from torch import nn
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

import wandb

import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

    return model


def compute_acc(results):

    if isinstance(results.predictions, tuple):
        preds = results.predictions[0]
    else:
        preds = results.predictions

    gold_label_list = results.label_ids

    acc = []
    preds_label_list = []

    choice_pred = []
    answer = None
    for gt, pd in zip(gold_label_list, preds):

        pd_label = torch.argmax(torch.tensor(pd)).item()

        if pd_label == gt:
            acc.append(1.0)
        else:
            acc.append(0.0)

    # for the last question
    # if args.dataset == "csqa":
    #     pass

    result_dict = {"eval_acc": np.mean(acc)}

    return result_dict



def train(model, train_dataset, dev_dataset, args):

    if args.inf_filtering:
        output_dir_path = f"./results_inf/{wandb.run.name}"
    else:
        output_dir_path = f"./results/{wandb.run.name}"

    training_args = TrainingArguments(
        output_dir=output_dir_path,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.train_batch_size,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        learning_rate=args.learning_rate,
        report_to=["wandb"],
        fp16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_acc"
    )

    if "pythia" in args.model_name_or_path:
        trainer = Trainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            preprocess_logits_for_metrics=logit_preprocess,
            compute_metrics=compute_acc,
        )

    else:
        trainer = Trainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_acc,
        )

    trainer.train()

    # save model
    # trainer.save_model(output_dir_path)

    return trainer
# end of train


def evaluate(test_dataset, test_features, model, args, tokenizer):
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    print("***** Running Evaluation *****")
    print("  Num examples = %d", len(test_loader))
    print("  Batch size = %d", args.test_batch_size)

    model.eval()

    dev_loss = 0.0
    test_iterator = tqdm(test_loader, desc="Test Evaluating")

    for step, batch in enumerate(test_iterator):
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)

            result = model(input_ids=batch[0], \
                           attention_mask=batch[1], \
                           labels=batch[2])

            loss = result[0]

            dev_loss += loss.item()

    print("  Evaluation Result = %.4f" % (dev_loss / len(test_iterator)))

    return dev_loss
# end of evaluate function


def train_custom(train_dataset, dev_dataset, model, args, tokenizer):

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_step, num_training_steps=len(train_loader)
    )

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    # loss_fn = nn.MSELoss()
    # if "pythia" in args.model_name_or_path:
    softmax = nn.Softmax(dim=1)
    # else:
    #     softmax = nn.Softmax(dim=0)

    set_seed(args)
    model.zero_grad()

    total_loss = 0.0

    for epoch in range(args.n_epochs):
        model.train()
        epoch_loss = 0.0

        train_iterator = tqdm(train_loader, desc="Training")

        for step, batch in enumerate(train_iterator):
            batch = tuple(b.to(args.device) for b in batch)

            output = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))

            # CrossEntropyLoss
            # num_label = 1 (list wise?)
            pred_probs = softmax(output.logits.squeeze(1).unsqueeze(0))
            loss = loss_fn(pred_probs, batch[-1])

            # num_label = 2 (binary classification)
            # pred_probs = softmax(output.logits)
            # loss = loss_fn(pred_probs, batch[-1].squeeze(0))

            # pdb.set_trace()

            # MSELoss
            # pdb.set_trace()

            epoch_loss += loss.item()

            # update the model
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()



        # print("Average loss for %d epoch : %.3f" %(epoch, (epoch_loss / len(train_loader))))

        print("Average loss for %d epoch : %.3f" %(epoch, epoch_loss))

        # save
        if args.inf_filtering:
            output_dir = "./results_inf/%s/checkpoint-%d" % (wandb.run.name, epoch)
        else:
            output_dir = "./results/%s/checkpoint-%d" %(wandb.run.name, epoch)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        model.save_pretrained(output_dir)

        total_loss += epoch_loss

    print("Total training loss in average : %.3f" %(total_loss / len(train_loader) / args.n_epochs))


    if args.inf_filtering:
        output_path = "./results_inf/%s/checkpoint-%d" % (wandb.run.name, epoch)
    else:
        output_path = "./results/%s" %(wandb.run.name)

    tokenizer.save_pretrained(output_path)

# train_custom




def gen_label(test_dataset, model, tokenizer, args):

    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)

    print("***** Running Evaluation *****")
    print("  Num examples = %d" %len(test_loader))
    print("  Batch size = %d" %args.test_batch_size)

    model.eval()

    test_iterator = tqdm(test_loader, desc="Test Evaluating")

    pred_answers_list = []
    gold_answers_list = []
    result = []
    for step, batch in enumerate(test_iterator):
        with torch.no_grad():
            batch = tuple(b.to(args.device) for b in batch)

            output = model.generate(batch[0], max_length=len(batch[0][0])+10)

            pred_answer = tokenizer.decode(output[0][len(batch[0][0]):], skip_special_tokens=True)
            gold_answer = tokenizer.decode(batch[1][0])

            pred_answers_list.append(pred_answer)
            gold_answers_list.append(gold_answer)

            if pred_answer == gold_answer:
                result.append(1.0)
            else:
                result.append(0.0)
                # print(gold_answer, pred_answer)

            # outputs = model(batch[0], batch[1])
            # predicted_class_id = outputs.logits.argmax().item()
            # if predicted_class_id == batch[2].item():
            #     result.append(1.0)
            # else:
            #     result.append(0.0)

    acc = np.mean(result)
    print("Accuracy with %d dev examples : %.2f" %(len(result), acc))

    return acc


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
# end of eval_with_trainer


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
    # else:
    # softmax = nn.Softmax(dim=0)

    # binary classification / num_label = 2
    # softmax = torch.nn.Softmax(dim=0)

    pred_answers_list = []
    gold_answers_list = []
    result = []
    for step, batch in enumerate(test_iterator):
        with torch.no_grad():
            batch = tuple(b.to(args.device) for b in batch)

            output = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))

            # num_label = 1 (list wise?)
            pred_probs = softmax(output.logits.squeeze(1).unsqueeze(0))

            pred_answer = torch.argmax(pred_probs).item()
            gold_answer = batch[-1].tolist()[0]

            # pdb.set_trace()
            # --------------------------

            # num_label = 2 (binary classification)
            # pred_probs = softmax(output.logits)
            #
            # pred_answer = torch.argmax(pred_probs, dim=1).tolist().index(1)
            # gold_answer = batch[-1].tolist()[0].index(1)
            # --------------------------
            pred_answers_list.append([pred_answer, gold_answer])

            if pred_answer == gold_answer:
                result.append(1.0)
            else:
                result.append(0.0)

    if args.inf_filtering:
        print("%.1f of the training dataset" %(args.inf_inst_num))

    print("Acc :", np.mean(result))

    if not args.inf_filtering or not args.mispred_training:
        json.dump(pred_answers_list, open("./preds/%s_%s.json" %(args.dataset, args.model_name_or_path[-4:]), "w"), indent=4)


    return np.mean(result)
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
    pred_list = []

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

            pred_list.append([int(pd_label), int(gt)])

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

    if not args.inf_filtering or not args.mispred_training:
        json.dump(pred_list, open("./preds/%s_%s.json" %(args.dataset, args.model_name_or_path[-4:]), "w"), indent=4)

    if args.mispred_training:
        json.dump(pred_list, open("./mispreds_result/%s" %args.mispred_exclude_list, "w"))


    return np.mean(acc)
# end of metrics_trainer_eval


def main(args):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # training
    if args.do_train:
        train_dataset = load_dataset(tokenizer, args, mode="train")

        if args.dataset in ["fever", "snli"]:
            dev_dataset = load_dataset(tokenizer, args, mode="validation")
        else:
            dev_dataset = load_dataset(tokenizer, args, mode="dev")

        model = load_model(args, num_labels=len(train_dataset.label_dict))
        model.to(args.device)

        if "pythia" in args.model_name_or_path:
            tokenizer.add_special_tokens({'pad_token': "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            model.config.pad_token_id = tokenizer.encode("<pad>")[0]

        #
        if args.dataset == "csqa":
            train_custom(train_dataset, dev_dataset, model, args, tokenizer)
        else:
            train(model, train_dataset, dev_dataset, args)


        if args.inf_filtering:
            output_path = "./results_inf/%s" % (wandb.run.name)
        else:
            output_path = "./results/%s" % (wandb.run.name)
        tokenizer.save_pretrained(output_path)


    # evaluation
    if args.do_eval or args.do_train:
        if args.dataset in ["fever", "snli"]:
            # dev set
            dev_dataset = load_dataset(tokenizer, args, mode="validation")

            # test set
            test_dataset = load_dataset(tokenizer, args, mode="test")


        elif args.dataset == "csqa":
            # dev set
            dev_dataset = load_dataset(tokenizer, args, mode="dev")

            # test set
            test_dataset = load_dataset(tokenizer, args, mode="validation")

        else:
            # dev set
            dev_dataset = load_dataset(tokenizer, args, mode="dev")

            # test set
            test_dataset = load_dataset(tokenizer, args, mode="test")

        # if args.inf_filtering:
        #     model_weight_list = [args.output_dir]
        # else:

        if args.output_dir is None:
            if args.inf_filtering:
                output_dir = "./results_inf/%s" % (wandb.run.name)
            else:
                output_dir = "./results/%s" % (wandb.run.name)
        else:
            output_dir = args.output_dir

        # if args.dataset == "csqa":
        model_weight_list = []
        # else:
        #     model_weight_list = [output_dir]

        for f_path in glob.glob("%s/*" %output_dir):
            if os.path.isdir(f_path):
                model_weight_list.append(f_path)

        print(model_weight_list)


        # evaluate multiple output dirs
        if len(model_weight_list) > 1:

            # dev acc
            dev_acc_dict = {}
            best_acc = 0.0
            best_path = ""
            for p in model_weight_list:
                model = load_model(args, num_labels=len(test_dataset.label_dict), trained_weight_path=p)
                model.to(args.device)

                if "pythia" in args.model_name_or_path:
                    tokenizer.add_special_tokens({'pad_token': "<pad>"})
                    model.resize_token_embeddings(len(tokenizer))
                    model.config.pad_token_id = tokenizer.encode("<pad>")[0]

                print("Load fine-tuned model from %s" % p)

                # to generate answer for csqa
                if args.dataset == "csqa":
                    # eval_result = gen_label(test_dataset, model, tokenizer, args)
                    eval_result = eval_custom(dev_dataset, model, tokenizer, args)
                else:
                    eval_result = eval_with_trainer(dev_dataset, model, tokenizer, args)

                dev_acc_dict[p] = eval_result

                if eval_result > best_acc:
                    best_acc = eval_result
                    best_path = p


            # with the best checkpoint / test acc
            print("Selecting the model with %.2f from %s" %(best_acc, best_path))
            model = load_model(args, num_labels=len(test_dataset.label_dict), trained_weight_path=best_path)
            model.to(args.device)

            if "pythia" in args.model_name_or_path:
                tokenizer.add_special_tokens({'pad_token': "<pad>"})
                model.resize_token_embeddings(len(tokenizer))
                model.config.pad_token_id = tokenizer.encode("<pad>")[0]

            if args.dataset == "csqa":
                # eval_result = gen_label(test_dataset, model, tokenizer, args)
                eval_result = eval_custom(test_dataset, model, tokenizer, args)
            else:
                eval_result = eval_with_trainer(test_dataset, model, tokenizer, args)

            print("From weight", best_path)
            print("Test Acc", eval_result)

        # single output dir
        else:
            model = load_model(args, num_labels=len(test_dataset.label_dict), trained_weight_path=args.output_dir)
            model.to(args.device)

            print("Load fine-tuned model from %s" % args.output_dir)

            # to generate answer for csqa
            if args.dataset == "csqa":
                # eval_result = gen_label(test_dataset, model, tokenizer, args)
                eval_custom(test_dataset, model, tokenizer, args)
            else:
                eval_result = eval_with_trainer(test_dataset, model, tokenizer, args)

            # print(eval_result)
# end of main

# https://docs.wandb.ai/quickstart

def read_args():
    # arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dataset", type=str, default="averitec", help="dataset_name")
    parser.add_argument("--max_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_name_or_path", type=str, default="opt-1.3b")
    parser.add_argument("--n_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    parser.add_argument("--inf_filtering", action="store_true")
    parser.add_argument("--inf_baseline", action="store_true")
    parser.add_argument("--inf_inst_num", type=float, default=300)
    parser.add_argument("--inf_file", type=str, default="top_inst_inf_func_139.json")
    parser.add_argument("--most_least", type=str, default="most", help="while inf training, which instances to select")

    parser.add_argument("--mispred_training", action="store_true")
    parser.add_argument("--mispred_exclude_list", type=str, default="")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_dev", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--local_rank", type=int, default=-1)

    # for custom training (without trainer)
    parser.add_argument("--warmup_step", type=int, default=0, help="step of linear warmup")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")

    # for deepspeed
    # parser = deepspeed.add_config_arguments(parser)

    training_args = parser.parse_args()

    training_args.num_gpu = torch.cuda.device_count()

    return training_args


if __name__ == "__main__":


    # wandb set-up
    args = read_args()

    set_seed(args)

    # main(args)

    if args.do_eval:
        main(args)

    else:
        wandb.init(project="neuron-attrib", config=args, group=args.model_name_or_path),

        print(args)

        if args.inf_filtering:
            if not os.path.exists(f"./results_inf/{wandb.run.name}"):
                print(f"Creating directory ./results_inf/{wandb.run.name}")
                os.makedirs(f"./results_inf/{wandb.run.name}")

        else:
            if not os.path.exists(f"./results/{wandb.run.name}"):
                print(f"Creating directory ./results/{wandb.run.name}")
                os.makedirs(f"./results/{wandb.run.name}")

        main(wandb.config)