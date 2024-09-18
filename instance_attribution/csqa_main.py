import os
import json
import torch
import collections
import argparse
import logging
import numpy as np
import random
import re, pickle, glob

from scipy import stats

from torch import nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
from tqdm import tqdm, trange
from transformers import AutoTokenizer, OPTForSequenceClassification, GPTNeoXForSequenceClassification, GPTNeoForCausalLM, BloomForSequenceClassification

from csqa_utils_influence import *
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

        logging.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
    else:
        logging.info('No GPU available, using the CPU instead.')


def load_dataset(tokenizer, args, mode="train"):
    if args.dataset in ["fever", "snli", "csqa"]:
        filepath = mode

    else:
        if mode == "train":
            filepath = "../../data/%s/train.json" % args.dataset

        elif mode == "dev":
            filepath = "../../data/%s/train.json" % args.dataset

        elif mode == "test":
            filepath = "../../data/%s/dev.json" % args.dataset

    dataset = fcDataset(filepath=filepath, tokenizer=tokenizer, args=args, mode=mode)

    return dataset


def load_model(args, num_labels):

    if args.dataset == "csqa":
        num_labels = 1

    if "opt" in args.model_name_or_path:
        model = OPTForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)

    elif "pythia" in args.model_name_or_path:
        model = GPTNeoXForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)

    elif "neo" in args.model_name_or_path:
        model = GPTNeoForCausalLM.from_pretrained(args.model_path)

    elif "bloom" in args.model_name_or_path:
        model = BloomForSequenceClassification.from_pretrained(args.model_path, num_labels=num_labels)

    return model


def get_target_params(model, args):

    all_params = list(model.named_parameters())
    param_influence = []

    for n, p in all_params:
        if args.word_embeddings_attr in n:
            continue

        elif args.input_ff_attr in n:
            if args.target_param == "mlp":
                param_influence.append(p)
            elif args.target_param == "mlp_5":
                names = n.split(".")
                if int(names[3]) > 5:
                    param_influence.append(p)

            # elif args.target_param == "last-tok":
            #     pdb.set_trace()

        elif args.output_ff_attr in n:
            if args.target_param == "mlp":
                param_influence.append(p)
            elif args.target_param == "mlp_5":
                names = n.split(".")
                if int(names[3]) > 5:
                    param_influence.append(p)

            # elif args.target_param == "last-tok":
            #     pdb.set_trace()

        elif n == "score.weight":
            param_influence.append(p)

    param_size = 0
    for p in param_influence:
        param_size += torch.numel(p)

    logging.info("  Parameter size = %d", param_size)

    # exit()

    return param_influence


def set_model_attr(args):

    if "opt" in args.model_name_or_path:
        args.transformer_layers_attr = "model.decoder.layers"
        args.input_ff_attr = "fc1"
        args.output_ff_attr = "fc2"
        args.word_embeddings_attr = "model.decoder.embed"

    elif "bloom" in args.model_name_or_path:
        args.transformer_layers_attr = "transformer.h"
        args.input_ff_attr = "mlp.dense_h_to_4h"
        args.output_ff_attr = "mlp.dense_4h_to_h"
        args.word_embeddings_attr = "transformer.word_embeddings"

    elif "pythia" in args.model_name_or_path:
        args.transformer_layers_attr = "gpt_neox.layers"
        args.input_ff_attr = "mlp.dense_h_to_4h"
        args.output_ff_attr = "mlp.dense_4h_to_h"
        args.word_embeddings_attr = "gpt_neox.embed_in"

    return args



def get_attributes(x: nn.Module, attributes: str):

    attr_list = attributes.split(".")

    for at in attr_list:
        x = getattr(x, at)
    return x


def register_hook(model, args):

    if 'opt' in args.model_name_or_path:
        projection_layers = get_attributes(model, "model.decoder.layers.11")

    elif 'bloom' in args.model_name_or_path:
        projection_layers = get_attributes(model, "transformer.h.23")

    elif 'pythia' in args.model_name_or_path:
        projection_layers = get_attributes(model, "gpt_neox.layers.23")

    pooled_output = {}

    def getActivation():
        # hook func will be called after forward has computed an output
        def hook(model, input, output):
            pooled_output["0"] = output[0]
        return hook

    handle = projection_layers.register_forward_hook(getActivation())

    return pooled_output, handle


def run_similarity_funcs(train_dataset, model, args):

    # model.train()
    train_loader_sim_func = DataLoader(train_dataset, batch_size=args.train_batch_size)

    pooled_output, handle = register_hook(model, args)

    target_params = get_target_params(model, args)

    loss_f = nn.CrossEntropyLoss(reduction="mean")
    softmax = nn.Softmax(dim=1)

    train_grads = []
    for idx, sample in tqdm(enumerate(train_loader_sim_func)):

        batch = tuple(s.to(args.device) for s in sample)

        output = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))

        pred_probs = softmax(output.logits.squeeze(1).unsqueeze(0))
        loss = loss_f(pred_probs, batch[-1])

        tmp_grads = autograd.grad(loss, target_params, retain_graph=True)[0].data.view(-1)
        train_grads.append(tmp_grads)


    handle.remove()

    return_train_grads = torch.stack(train_grads)

    logging.info("Train grads matrix : (%d, %d)" %(return_train_grads.shape[0], return_train_grads.shape[1]))

    return return_train_grads, target_params


def do_inf_function(idx, batch, model, param_influence, train_dataset, train_dataloader, args):

    # L_test gradient
    model.zero_grad()
    test_loss = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])[0]
    test_grads = autograd.grad(test_loss, param_influence)

    # compute inverse Hessian-vector product
    model.train()
    logging.info("**** Computing IHVP for test step %d ****", idx)

    inverse_hvp = get_inverse_hvp_lissa(test_grads, model, param_influence, train_dataset, args)

    # compute influence score for each data
    inf_score_list = compute_influence_score(train_dataloader, model, param_influence, inverse_hvp, args)

    return inf_score_list


def get_only_ihvp(idx, batch, model, param_influence, train_dataset, softmax, loss_f, args):
    model.zero_grad()

    outputs = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))
    pred_probs = softmax(outputs.logits.squeeze(1).unsqueeze(0))
    test_loss = loss_f(pred_probs, batch[-1])

    test_grads = autograd.grad(test_loss, param_influence)

    # compute inverse Hessian-vector product
    model.train()
    logging.info("**** Computing IHVP for test step %d ****", idx)

    inverse_hvp = get_inverse_hvp_lissa(test_grads, model, param_influence, train_dataset, args)

    return inverse_hvp


def compute_influence_score(train_dataloader, model, param_influence, ihvp, args):
    influences = np.zeros(len(train_dataloader.dataset))

    for idx, sample in tqdm(enumerate(train_dataloader)):
        batch = tuple(s.to(args.device) for s in sample.values())

        model.zero_grad()
        train_loss = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])[0]
        train_grads = autograd.grad(train_loss, param_influence, retain_graph=True)
        influences[idx] = torch.dot(ihvp, gather_flat_grad(train_grads)).item()

    return influences


def do_inf_func_with_precomputed_ihvp(inverse_hvp_list, train_dataloader, model, param_influence, args):

    influences = np.zeros(len(inverse_hvp_list))
    ihvp_tensor = torch.stack(inverse_hvp_list)

    print("Size of the ihvp_tensor : (%d, %d)" %(ihvp_tensor.size()[0], ihvp_tensor.size()[1]))
    print("Starting computation of inf scores with training dataset ...")

    softmax = nn.Softmax(dim=1)
    loss_f = nn.CrossEntropyLoss(reduction="mean")

    total_result = {}
    for idx, sample in tqdm(enumerate(train_dataloader)):
        batch = tuple(s.to(args.device) for s in sample)

        model.zero_grad()
        outputs = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))

        pred_probs = softmax(outputs.logits.squeeze(1).unsqueeze(0))
        train_loss = loss_f(pred_probs, batch[-1])

        train_grads = autograd.grad(train_loss, param_influence, retain_graph=True)

        # scores = torch.dot(ihvp_tensor, gather_flat_grad(train_grads))
        # scores for each dev
        scores = torch.matmul(ihvp_tensor, gather_flat_grad(train_grads))

        if idx == 0:
            print("Shape of the score : %d" %len(scores))

        total_result[idx] = scores.tolist()

    # total_result keys : train_indices
    # values : scores_list / length of # of test instances

    return total_result



def main(args):
    print(args)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load datasets
    # train dataset
    train_dataset = load_dataset(tokenizer, args, mode="train")

    # test dataset
    if args.dataset == "snli":
        dev_dataset = load_dataset(tokenizer, args, mode="test")

    elif args.dataset == "csqa":
        dev_dataset = load_dataset(tokenizer, args, mode="validation")

    else:
        dev_dataset = load_dataset(tokenizer, args, mode="test")

    # model
    model = load_model(args, num_labels=len(train_dataset.label_dict))
    model.to(args.device)

    if "pythia" in args.model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.encode("<pad>")[0]

    # target params

    if args.IA_method == "inf-func":
        param_influence = get_target_params(model, args)

    elif args.IA_method == "grad-sim":
        train_grads, param_influence = run_similarity_funcs(train_dataset, model, args)

        pooled_output, handle = register_hook(model, args)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=1)

    softmax = nn.Softmax(dim=1)
    loss_f = nn.CrossEntropyLoss(reduction="mean")

    total_result = {}
    inverse_hvp_list = []

    for idx, sample in enumerate(dev_dataloader):
        model.eval()
        batch = tuple(s.to(args.device) for s in sample)

        if args.IA_method == "inf-func":
            # get test prediction
            # with torch.no_grad():
            #     outputs = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))

            # pred_probs = softmax(outputs.logits.squeeze(1).unsqueeze(0))
            # loss = loss_f(pred_probs, batch[-1])

            # original
            # inf_score_list = do_inf_function(idx, batch, model, param_influence, train_dataset, train_dataloader, args)

            tmp_ihvp = get_only_ihvp(idx, batch, model, param_influence, train_dataset, softmax, loss_f, args)
            inverse_hvp_list.append(tmp_ihvp)


        elif args.IA_method == "grad-sim":
            model.zero_grad()

            outputs = model(input_ids=batch[0].squeeze(0), attention_mask=batch[1].squeeze(0))

            pred_probs = softmax(outputs.logits.squeeze(1).unsqueeze(0))
            loss = loss_f(pred_probs, batch[-1])

            test_grads = autograd.grad(loss, param_influence)[0].view(-1)

            # simple dot product
            inf_score_list = train_grads @ test_grads.T

            total_result[idx] = inf_score_list.tolist()


    if args.IA_method == "inf-func":
        total_result = do_inf_func_with_precomputed_ihvp(inverse_hvp_list, train_dataloader, model, param_influence, args)

    pickle.dump(total_result, open(args.output_file, "wb"))

# end of main


"""
norm = lambda x : x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

sim_func_dict = {
    "cos" : lambda x, y: norm(x) @ norm(y).T,
    "dot" : lambda x, y: x @ y.T,
    "euc" : lambda x, y: -np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
}
"""

def load_fever_result(args):

    result_json_file = {int(r.split("-")[-1].replace(".json", "")[-1]): r for r in glob.glob("./%s/*.json" %args.output_file)}

    total = {}

    if result_json_file == {}:
        result_file_list = {int(r.split("-")[-1].replace(".pickle", "")): r for r in glob.glob("./%s/*.pickle" %args.output_file)}
        print(result_file_list)

        test_scores = {}
        # range
        # 0 - 6 / part0
        # 6 - len(result_file_list) / part1
        for file_idx in range(6, len(result_file_list)):
            filename = result_file_list[file_idx]
            # file - test idx : train inst scores (splitted)
            file = pickle.load(open(filename, "rb"))
            print(filename, len(file[0]))

            for test_idx, score_list in file.items():
                # test_scores.setdefault(idx, [])
                # test_scores[idx] += score_list
                score_tensor = torch.tensor(score_list)
                ranking = torch.argsort(score_tensor)[:100]
                selected_scores = torch.index_select(score_tensor, dim=0, index=ranking).tolist()

                test_scores.setdefault(test_idx, {"instances":[], "scores":[]})
                test_scores[test_idx]["instances"] += [r.item()+(20000*file_idx) for r in ranking]
                test_scores[test_idx]["scores"] += selected_scores

        json.dump(test_scores, open("./gs_fever_opt125m/grad-sim-fever-part1.json", "w"), indent=2)

    else:
        for fkey in sorted(result_json_file):
            file = json.load(open(result_json_file[fkey]))

            for test_idx, result in file.items():
                total.setdefault(test_idx, {"instances":[], "scores":[]})
                total[test_idx]["instances"] += result["instances"]
                total[test_idx]["scores"] += result["scores"]

                assert len(set(total[test_idx]["instances"])) == len(total[test_idx]["instances"])


    return total

# end of load_fever_result



def rank_train_influence(args, inf_result=None):
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # train_dataset = load_dataset(tokenizer, args, mode="train")
    # dev_dataset = load_dataset(tokenizer, args, mode="dev")

    # inf_result is not None with fever dataset
    if inf_result is None:
        result_file = pickle.load(open(args.output_file, "rb"))
    else:
        result_file = inf_result

    neuron_result = json.load(open(args.neuron_file_name))

    print(args.output_file)
    print(args.neuron_file_name)

    logging.info("%s : %d" %(args.output_file, len(result_file)))

    inf_ratio = 0.0
    neuron_ratio = 0.0

    zero_overlap = 0

    for data_idx, scores in result_file.items():

        if isinstance(scores, dict):
            score_tensor = torch.tensor(stats.zscore(scores["scores"]))

            if args.ia_adaptive_th != 0.0:
                max_score = torch.max(score_tensor)
                # 14th Dec
                # negative max_score ??? - I skipped the z normalize(stats.zscore)
                ranking = torch.nonzero(score_tensor > max_score * args.ia_adaptive_th).squeeze(1).tolist()

            else:
                sorted_score = torch.argsort(score_tensor, descending=True)
                ranking = [scores["instances"][i] for i in sorted_score[:args.inf_instance_number]]

        else:
            score_tensor = torch.tensor(stats.zscore(scores))

            # adaptive ranking
            if args.ia_adaptive_th != 0.0:
                max_score = torch.max(score_tensor)
                ranking = torch.nonzero(score_tensor > max_score * args.ia_adaptive_th).squeeze(1).tolist()

            # top-n ranking
            else:
                ranking = torch.argsort(score_tensor, descending=True)[:args.inf_instance_number].tolist()

        # Overlapping
        cnt = 0
        for r in ranking:
            if r in neuron_result[str(data_idx)]:
                cnt += 1

        logging.info("\nResult of %d-th dev example" %(int(data_idx)))
        logging.info("Overlap : %d / Neuron : %d / Inf instance : %d" %(cnt, len(neuron_result[str(data_idx)]), len(ranking)))

        if len(neuron_result[str(data_idx)]) == 0:
            neuron_ratio += 0.0
            inf_ratio += 0.0

        else:
            neuron_ratio += cnt / len(neuron_result[str(data_idx)])
            inf_ratio += cnt / len(ranking)

        if cnt == 0:
            zero_overlap += 1

        # print("end")

    logging.info("** cnt / len(insts from ia) : %.3f" %(inf_ratio/len(result_file)))
    logging.info("** cnt / len(insts from na) : %.4f" %(neuron_ratio/len(result_file)))
    # logging.info(zero_overlap / len(result_file))
# end of rank_train_influence


def compare_ia_results(args):
    result_1 = pickle.load(open("inf-func-1114.pickle", "rb"))
    result_2 = pickle.load(open("inf_func_last_linear.pickle", "rb"))

    if args.top_bottom == "top":
        flag = True
        # top instances
    else:
        flag = False

    avg_overlap = []
    for data_idx, scores in result_1.items():
        score_tensor = torch.tensor(stats.zscore(scores))
        ranking = torch.argsort(score_tensor, descending=flag)[:args.inf_instance_number].tolist()

        score_2 = torch.tensor(stats.zscore(result_2[data_idx]))
        ranking_2 = torch.argsort(score_2, descending=flag)[:args.inf_instance_number].tolist()

        cnt = 0
        for r in ranking:
            if r in ranking_2:
                cnt += 1

        avg_overlap.append(cnt/args.inf_instance_number)
        logging.info("test data %d : %d" %(data_idx, cnt))

    logging.info(np.mean(avg_overlap))



if __name__ == "__main__":

    # arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dataset", type=str, default="averitec", help="dataset_name")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--model_name_or_path", type=str, default="opt-1.3b")
    parser.add_argument("--output_file", type=str, default="./influence_scores",
                        help="Path, url or short name of the model")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--IA_method", type=str, default="inf-func")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--damping", type=float, default=3e-3, help="probably need damping for deep models")
    parser.add_argument("--lissa_depth", default=0.25, type=float)
    parser.add_argument("--target_param", type=str, default="all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--do_rank", action="store_true")
    parser.add_argument("--do_ia_compare", action="store_true")
    parser.add_argument("--inf_instance_number", type=int, default=10)
    parser.add_argument("--ia_adaptive_th", type=float, default=0.0)
    parser.add_argument("--neuron_file_name", type=str)
    parser.add_argument("--top_bottom", type=str, default="top")
    parser.add_argument("--scale", type=float, default=1e4)
    args = parser.parse_args()

    args.split = False

    args = set_model_attr(args)

    # log file name
    log_file_name = ".log"
    ia_file_info = args.output_file.split("/")[0].split(".pickle")[0]

    if args.do_rank:
        if os.path.isdir(args.output_file):
            log_file_name = "tmp.log"

        else:
            log_file_name = "compare_ia_na_" + ia_file_info + log_file_name

    else:
        log_file_name = "compute_ia" + ia_file_info + log_file_name


    # logging console print
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file_name,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    if args.do_rank:
        logging.info(args)

        inf_result=None
        if os.path.isdir(args.output_file):
            inf_result = load_fever_result(args)

        rank_train_influence(args, inf_result=inf_result)

    elif args.do_ia_compare:
        logging.info(args)
        # haeun : Comparison between two IA results
        compare_ia_results(args)

    else:
        main(args)

    # test
    # eval_result = evaluate(test_loader, model, args, kobart_tokenizer)
    # logging.info('********** Test Result **********')
    # logging.info('Accuracy : {:.4f}'.format(eval_result["accuracy"]))