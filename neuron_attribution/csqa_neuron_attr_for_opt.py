from eai_knowledge_neurons_for_opt import *

from utils_na import *
from collections import Counter

import numpy as np
import argparse
import random
import torch
import os


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
# end of set_seed


CREATE_FUNCTIONS = {
  "averitec": create_averitec_examples,
  "fever": create_fever_examples,
  "snli": create_snli_examples,
  "csqa": create_csqa_examples,
}


def load_examples(args, mode="train"):
  if args.dataset in ["fever", "snli", "csqa"]:
    filepath = mode

  elif args.dataset == "averitec":
    if mode == "train":
      filepath = "../data/%s/train.json" % args.dataset

    elif mode == "dev":
      filepath = "../data/%s/train.json" % args.dataset

    elif mode == "test":
      filepath = "../data/%s/dev.json" % args.dataset

  examples = CREATE_FUNCTIONS[args.dataset](filepath, mode)

  return examples


def refining(original, diverse_neurons, original_attr_scores):

  total = []
  original_str = ["%d-%d" %(i[0], i[1]) for i in original]
  total += original_str

  for dn in diverse_neurons:
    total += ["%d-%d" %(i[0], i[1]) for i in dn]

  new_neurons = []
  new_attr_scores = []
  for k, v in Counter(total).items():
    # only retain the neurons from the original prompt
    if v > 6 and k in original_str:
      new_neurons.append(k)
      selected_idx = original_str.index(k)
      new_attr_scores.append(original_attr_scores[selected_idx])

  pct = len(new_neurons) / len(original)
  print("%d / %d /// Percentage : %.2f" %(len(new_neurons), len(original), pct))

  assert len(new_neurons) == len(new_attr_scores)

  return new_neurons, new_attr_scores


def run_neuron_attr(examples, label_dict, model, tokenizer, args):

  gap = 1000
  kn = KnowledgeNeurons(model, tokenizer, model_type=args.model_type, args=args)

  # neuron attribution 기준을 max prob token from baseline? or each run?
  # go to paper

  if args.start_point + gap > len(examples):
    print("From %d to the end / total %d" %(args.start_point, len(examples)))
    example_sample_list = examples[args.start_point:]

  else:
    print("From %d to %d / total %d" %(args.start_point, args.start_point+gap, len(examples)))
    example_sample_list = examples[args.start_point:args.start_point+gap]

  neuron_for_example = []
  acc = []
  for idx, examp in tqdm(enumerate(example_sample_list)):

    # pdb.set_trace()
    if args.dataset in ["snli", "csqa"]:
      ground_truth_label = examp.label

    else:
      ground_truth_label = label_dict[examp.label]

    # prompt ver1
    prompt = "Question : %s/ Choices : %s/ Answer :" %(examp.claim, examp.evidence)

    # prompt ver2
    # prompt = "Evidence : %s/ Claim :%s/ Answer :" %(examp.evidence, examp.claim)

    if args.target_pos == "seq-end":
      t_pos = -1

    # elif args.target_pos == "claim-end":
    #   t_pos = len(tokenizer.tokenize("Claim : %s" %examp.claim))
    #
    # elif args.target_pos == "evi-end":
    #   t_pos = len(tokenizer.tokenize("Claim : %s/ Evidence : %s" %(examp.claim, examp.evidence)))

    # adaptive_threshold 0.8 / 0.5
    # try:
    actual_diverse_flag = args.diverse

    kn.args.diverse = False
    neuron_list, attribution_scores, original_pred_label = kn.get_coarse_neurons(prompt, examp, ground_truth_label, target_position=t_pos, batch_size=1, \
                                         steps=20, adaptive_threshold=0.1)

    if neuron_list == []:
      print(neuron_list, idx)

    # if actual_diverse_flag:
    #   kn.args.diverse = True
    #
    #   diverse_results = []
    #   cnt = 0
    #   while len(diverse_results) < 9:
    #     diverse_tmp_nr, _, diverse_pred_label = kn.get_coarse_neurons(prompt, examp, ground_truth_label,
    #                                                                                 target_position=t_pos, batch_size=1, \
    #                                                                                 steps=20, adaptive_threshold=0.3)
    #
    #     # check the predicted label
    #     if original_pred_label == diverse_pred_label:
    #       diverse_results.append(diverse_tmp_nr)
    #
    #     cnt += 1
    #     if cnt > 20:
    #       break
    #
    #   # refining and return new neuron_list / attribution_scores
    #   neuron_list, attribution_scores = refining(neuron_list, diverse_results, attribution_scores)


    # 1st Nov : just run for the labels (step = 0 ?)
    if original_pred_label == ground_truth_label:
      acc.append(1.0)

    else:
      acc.append(0.0)
      # print(examp.claim)
      # print(neuron_list)

    neuron_for_example.append(
      {
        "claim": examp.claim,
        "evidence": examp.evidence,
        "pd_label": original_pred_label,
        "gt_label": ground_truth_label,
        "idx": idx + args.start_point,
        "neurons": neuron_list,
        "attributions": attribution_scores
      }
    )

    torch.cuda.empty_cache()

  print("Accuracy", np.mean(acc))

  return neuron_for_example


def single_example_neuron_attr(label_dict, model, tokenizer, args):

  kn = KnowledgeNeurons(model, tokenizer, model_type=args.model_type, args=args)

  # prompt = "Claim : / Evidence : / Answer :"

  prompt = "Evidence : / Claim : / Answer :"

  neuron_list, attribution_scores, max_baseline_label = kn.get_coarse_neurons(prompt, "", 0,
                                                                              target_position=-1, batch_size=1, \
                                                                              steps=20, adaptive_threshold=0.1)

  neuron_for_example = [{
        "claim": prompt,
        "evidence": "",
        "neurons": neuron_list,
        "attributions": attribution_scores
      }]

  return neuron_for_example


if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--model_type", type=str, default="facebook/opt-1.3b")
  parser.add_argument("--dataset", type=str, default="averitec")
  parser.add_argument("--max_len", type=int, default=1024)
  parser.add_argument("--start_point", type=int)
  parser.add_argument("--diverse", action="store_true")
  parser.add_argument("--output_dir", type=str)
  parser.add_argument("--mode", type=str, default="train")
  parser.add_argument("--target_pos", type=str, default="seq-end")

  args = parser.parse_args()

  filepath = os.path.join("results_neuron", args.output_dir)
  if os.path.exists("%s/neuron_%s_%s.json" % (filepath, args.mode, str(args.start_point))):
    print("%s/neuron_%s_%s.json" % (filepath, args.mode, str(args.start_point)))
    print("Already exists / Stop")
    exit()


  args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  set_seed(args)

  print("** Args **")
  print(args)

  label_dict = json.load(open("../data/commonsense_qa/label_dict.json"))
  tokenizer, model = load_tokenizer_model(args.model_type, args.model_path, len(label_dict))

  tokenizer.add_special_tokens({'pad_token': "<pad>"})
  model.resize_token_embeddings(len(tokenizer))
  model.config.pad_token_id = tokenizer.encode("<pad>")[0]

  if args.mode == "single":
    neuron_results = single_example_neuron_attr(label_dict, model, tokenizer, args)

  else:
    examples, _ = load_examples(args, mode=args.mode)
    neuron_results = run_neuron_attr(examples, label_dict, model, tokenizer, args)


  # write the results
  filepath = os.path.join("results_neuron", args.output_dir)

  if not os.path.exists(filepath):
    os.mkdir(filepath)

  json.dump(neuron_results, open("%s/neuron_%s_%s.json" % (filepath, args.mode, str(args.start_point)), "w"), indent=4)
