import json, jsonlines
import logging
from torch.utils.data import Dataset
import torch
import glob, os
import pickle

from scipy import stats
from datasets import load_dataset

from transformers import AutoTokenizer, OPTForSequenceClassification, GPTNeoXForSequenceClassification, GPTNeoForSequenceClassification, BloomForSequenceClassification
from transformers import BloomTokenizerFast

import pdb


def load_tokenizer_model(model_type, model_path, num_labels):

  if "csqa" in model_path:
    num_labels = 1

  if "opt" in model_type:
    model = OPTForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

  elif "pythia" in model_type:
    model = GPTNeoXForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

  elif "bloom" in model_type:
    model = BloomForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

  elif "neo" in model_type:
    model = GPTNeoForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

  # if "bloom" in model_type and "csqa" in model_path:
  #   tokenizer = AutoTokenizer.from_pretrained("../FV_finetuning/results/csqa_bloom_re")
  # else:
  tokenizer = AutoTokenizer.from_pretrained(model_type)
  
  return tokenizer, model


class fcExample():
  def __init__(self, claim=None, evidence=None,
               label=None, source=None, idx=None):
    self.claim = claim
    self.evidence = evidence
    self.label = label
    self.source = source
    self.idx = idx


def create_mnli_examples(filename, mode):

  label_dict = json.load(open("../data/mnli/label_dict.json"))

  if mode == "train":
    reader = jsonlines.open("../data/mnli/mnli_train.jsonl")

  elif mode == "dev":
    reader = jsonlines.open("../data/mnli/mnli_dev.jsonl")

  elif mode == "test":
    reader = jsonlines.open("../data/mnli/mnli_test.jsonl")

  examples = []
  label_set = set()
  for line in reader:

    examples.append(
      fcExample(claim=line["query"],
                evidence=line["document"],
                label=line["label"])
    )


    label_set.add(line["label"])

  print(label_set)
  print("From %s / Total %d examples are created" % (filename, len(examples)))

  return examples, label_dict


def create_hans_examples(filename, mode, model_name):

  examples = []
  label_dict = json.load(open("../data/mnli/label_dict.json"))
  hans_label_dict = json.load(open("../data/hans/label_dict.json"))

  target_indices = json.load(open("../data/hans/hans_wrong_%s.json" %model_name))
  print("The target indices len : %d" %len(target_indices))

  label_set = []

  with jsonlines.open("../data/hans/hans_test.jsonl") as reader:
    cnt = 0
    for line in reader:
      if cnt in target_indices:
        label_text = hans_label_dict[str(line["label"])]
        int_label = label_dict[label_text]

        label_set.append(int_label)

        examples.append(
          fcExample(
            claim=line["query"],
            evidence=line["document"],
            label=int_label,
            idx=line["idx"]
          )
        )

      cnt += 1

  print(set(label_set))

  print("From hans testset / Total %d examples are created" %(len(examples)))

  return examples, label_dict


def create_averitec_examples(filename, mode=None):
  data = json.load(open(filename))
  label_dict = json.load(open("../data/averitec/label_dict.json"))

  if mode == "train":
    target_indices = json.load(open("../data/averitec/train_idx.json"))

  elif mode == "dev":
    target_indices = json.load(open("../data/averitec/dev_idx.json"))

  else:
    target_indices = [i for i in range(len(data))]

  examples = []
  for idx, d in enumerate(data):
    if idx in target_indices:
      # for evidence (in QA form)
      evid = ""
      source_list = []
      for idx, qa in enumerate(d["questions"]):
        evid += "(%s)" % str(idx)
        evid += qa["question"] + " "
        evid += qa["answers"][0]["answer"]
        source_list.append(qa["answers"][0]["source_url"])
        # break

      examples.append(
        fcExample(claim=d["claim"],
                  evidence=evid,
                  label=d["label"],
                  source=source_list)
      )

  print("From %s / Total %d examples are created" % (filename, len(examples)))

  return examples, label_dict


def create_snli_examples(filename, mode=None):
  mnli_data = load_dataset("snli", split=filename)
  label_dict = json.load(open("../data/snli/label_dict.json"))

  if "train" in filename:
    if os.path.exists("../data/snli/train_idx.json"):
      print("Loading pre-sampled indices ...")
      train_idx_list = json.load(open("../data/snli/train_idx.json"))

    else:
      print("Sampling training instances ...")

      valid_data_indices = []
      for idx, d in enumerate(mnli_data):
        if d["label"] != -1:
          valid_data_indices.append(idx)

      print("Original %d // Filtered %d" % (len(mnli_data), len(valid_data_indices)))

      train_idx_list = random.sample(valid_data_indices, 10000)
      json.dump(train_idx_list, open("../data/snli/train_idx.json", "w"))

  else:
    train_idx_list = None

  examples = []
  label_list = []
  for d_idx, d in enumerate(mnli_data):
    if train_idx_list:
      if d_idx not in train_idx_list:
        continue

    if d["label"] != -1:
      examples.append(
        fcExample(claim=d["premise"],
                  evidence=d["hypothesis"],
                  label=d["label"])
      )

      label_list.append(d["label"])

  print(set(label_list))
  print("From %s / Total %d examples are created" % (filename, len(examples)))

  return examples, label_dict


def create_csqa_examples(filename, mode=None):

  if filename == "dev":
    filename = "train"

  csqa_data = load_dataset("commonsense_qa", split=filename)
  label_dict = json.load(open("../data/commonsense_qa/label_dict.json"))

  if mode == "train":
    target_indices = json.load(open("../data/commonsense_qa/train_idx.json"))

  elif mode == "dev":
    target_indices = json.load(open("../data/commonsense_qa/dev_idx.json"))

  else:
    target_indices = [i for i in range(len(csqa_data))]


  examples = []
  # concept_list = []
  for idx, d in enumerate(csqa_data):

    if idx in target_indices:
      # id / question / question_concept / label / text / answerKey
      # concept_list.append(d["question_concept"])

      answer_idx = d["choices"]["label"].index(d["answerKey"])

      # for listwise model
      examples.append(
        fcExample(claim=d["question"],
                  evidence=d["choices"]["text"],
                  label=answer_idx)
      )

  # # of unique question_concepts : 2151
  print("From %s / Total %d examples are created" % (filename, len(examples)))

  return examples, label_dict


# https://huggingface.co/datasets/copenlu/fever_gold_evidence
def create_fever_examples(filename):

  fever_data = load_dataset("copenlu/fever_gold_evidence", split=filename, streaming=True)

  examples = []
  for d in fever_data:
    evid = ""
    for e in d["evidence"]:
      assert len(e) == 3
      evid += "%s %s" %(e[0], e[-1])

    examples.append(
      fcExample(claim=d["claim"],
                evidence=evid,
                label=d["label"])
    )


  logging.info("From %s / Total %d examples are created" %(filename, len(examples)))
  return examples


CREATE_FUNCTION = {
  "averitec": create_averitec_examples,
  "fever": create_fever_examples
}

prompt_dataset = {
  "averitec": "Generate the answer to the question. There are four options for the answer, (1) Conflicting Evidence/Cherrypicking, (2) Refuted, (3) Supported, (4) Not Enough Evidence. / Question : What's the relationship between the evidence and the claim?"
}


class blockDataset(Dataset):
  def __init__(self, tokenizer, args):
    self.data = self.merge_files(args.neuron_folder, args.mode)
    self.tokenizer = tokenizer
    self.args = args

    if self.args.inf_result != "":
      features, neuron_list = self.create_inf_features()
    else:
      features, neuron_list = self.create_features()

    if args.dataset == "csqa":
      self.features = self.data
    else:
      self.features = features


    self.neuron_list = neuron_list

    if "pythia" in args.model_type:
      self.pad_token_ids = "<pad>"
    else:
      self.pad_token_ids = self.tokenizer.pad_token



  def merge_files(self, folder, mode):
    train_files = glob.glob("%s/neuron_%s_*.json" % (folder, mode))

    print("Loading ...")

    # sorting
    key = "neuron_%s_" % mode
    before_sort = {int(f.split(key)[-1].replace(".json", "")): f for f in train_files}
    sorted_key = sorted(before_sort)

    total_train = []
    for tf in sorted_key:
      tmp_t = json.load(open(before_sort[tf]))
      total_train += tmp_t
      print(before_sort[tf], len(tmp_t))

    return total_train


  def create_features(self):

    features = []
    neuron_list = []
    for d in self.data:
      claim = self.tokenizer.tokenize("Claim : %s" %(d["claim"]))
      evidence = self.tokenizer.tokenize("/ Evidence : %s" %(d["evidence"]))
      answer = self.tokenizer.tokenize("/ Answer :")

      if self.args.target_pos == "seq-end":
        target_pos = -1
      elif self.args.target_pos == "claim-end":
        target_pos = len(claim)
      elif self.args.target_pos == "evi-end":
        target_pos = len(claim) + len(evidence)

      while len(claim) + len(evidence) + len(answer) > (self.args.max_len - 1):
        # print("Truncation with len :", (len(claim) + len(evidence) + len(answer)))
        evidence.pop(-1)

        if self.args.target_pos == "evi-end":
          target_pos -= 1

      input_sen = ["</s>"] + claim + evidence + answer
      attn_mask = [1] * len(input_sen)

      input_ids = self.tokenizer.convert_tokens_to_ids(input_sen)

      features.append([input_ids, attn_mask])
      neuron_list.append([d["neurons"], d["attributions"], target_pos])

    return features, neuron_list


  def inf_common_neurons(self, inf_results, train_nrs):

    top_10_insts = torch.argsort(torch.tensor(inf_results), descending=True)[:10]

    total_nr = {}
    # key : neuron / value : [scores ... ]
    for inst in top_10_insts:
      nr_info = train_nrs[inst]
      important_neurons = torch.argsort(torch.tensor(nr_info["attributions"]), descending=True)[:5]

      for nr_idx in important_neurons:
        nr_key = nr_info["neurons"][nr_idx]
        nr_score = nr_info["attributions"][nr_idx]

        nr_key = "%d-%d" %(nr_key[0], nr_key[1])
        total_nr.setdefault(nr_key, 0.0)
        total_nr[nr_key] += nr_score

    neuron_list = []
    attr_scores = []
    for nr_key, score in total_nr.items():
      nr_key_list = [int(i) for i in nr_key.split("-")]
      neuron_list.append(nr_key_list)
      attr_scores.append(score)

    return neuron_list, attr_scores


  def create_inf_features(self):

    inf_results = pickle.load(open(self.args.inf_result, "rb"))

    train_nrs = self.merge_files(self.args.neuron_folder, "train")


    features = []
    neuron_list = []

    # for analysis
    neurons = []

    for idx, d in enumerate(self.data):

      if self.args.dataset == "averitec":
        claim = self.tokenizer.tokenize("Claim : %s" % (d["claim"]))
        evidence = self.tokenizer.tokenize("/ Evidence : %s" % (d["evidence"]))
        answer = self.tokenizer.tokenize("/ Answer :")

      elif self.args.dataset in ["snli", "mnli", "hans"]:
        claim = self.tokenizer.tokenize("Premise : %s" % d["claim"])
        evidence = self.tokenizer.tokenize("/ Hypothesis : %s" % d["evidence"])
        answer = self.tokenizer.tokenize("/ Answer :")
        # labels = example.label

      elif self.args.dataset == "csqa":
        claim = self.tokenizer.tokenize("Question : %s" % d["claim"])
        evidence = self.tokenizer.tokenize("/ Choices : %s" % d["evidence"])
        answer = self.tokenizer.tokenize("/ Answer :")
        # labels = self.label_dict[example.label]

      target_pos = -1

      while len(claim) + len(evidence) + len(answer) > (self.args.max_len - 1):
        # print("Truncation with len :", (len(claim) + len(evidence) + len(answer)))
        evidence.pop(-1)

      input_sen = ["</s>"] + claim + evidence + answer
      attn_mask = [1] * len(input_sen)

      input_ids = self.tokenizer.convert_tokens_to_ids(input_sen)

      features.append([input_ids, attn_mask])

      # method 1: influential training instances for this dev instance
      # take top-1 training instance and its neurons
      #
      # inf_tr_scores= torch.tensor(inf_results[idx])
      # max_tr_instance = torch.argmax(inf_tr_scores).item()
      # max_tr_info = train_nrs[max_tr_instance]
      # neuron_list.append([max_tr_info["neurons"], max_tr_info["attributions"], target_pos])
      # ------------


      # method 2: common neurons among top-10 instances
      # inf_nr_list, attr_list = self.inf_common_neurons(inf_results[idx], train_nrs)
      # neuron_list.append([inf_nr_list, attr_list, target_pos])
      # ------------

      # method 3:
      inf_lists = torch.argsort(torch.tensor(inf_results[idx]), descending=True)[:self.args.neuron_num]

      inf_nr_list = []
      attr_list = []
      for idx, inf_inst in enumerate(inf_lists):
        nr_info = train_nrs[inf_inst]
        max_nr_idx = torch.argmax(torch.tensor(nr_info["attributions"])).item()
        max_nr = nr_info["neurons"][max_nr_idx]

        inf_nr_list.append(max_nr)
        attr_list.append(-idx)

      neuron_list.append([inf_nr_list, attr_list, target_pos])
      # ---------------


      # for analysis
      # max_nr_idx = torch.argsort(torch.tensor(max_tr_info["attributions"]), descending=True)[:1]
      #
      # for idx in max_nr_idx:
      #   max_nr = max_tr_info["neurons"][idx]
      #   neurons.append("%d-%d" %(max_nr[0], max_nr[1]))


    # import json
    # json.dump(list(set(neurons)), open("gradsim_result_important_neuron_top_1.json", "w"), indent=4)
    # print("length of list : %d" %(len(neurons)))
    # print("Unique neurons : %d" %(len(set(neurons))))
    # exit()

    return features, neuron_list


  def __len__(self):
    return len(self.features)


  def __getitem__(self, idx):
    if self.args.dataset == "csqa":
      example = self.features[idx]

      claim = self.tokenizer.tokenize("Question : %s" % example["claim"])

      gold_label = torch.tensor(example["gt_label"])  # for listwise

      # evidence = [example["evidence"][example["pd_label"]]]

      input_ids_list = []
      attn_mask_list = []
      label_list = []
      mask_idx_list = []
      for idx, choice in enumerate(example["evidence"]):
        evidence = self.tokenizer.tokenize("/ Choice : %s" % choice)
        answer = self.tokenizer.tokenize("/ Answer :")

        flag = True
        original_len = len(claim) + len(evidence) + len(answer)

        while len(claim) + len(evidence) + len(answer) > (self.args.max_len - 1):
          # print("Truncate")
          evidence.pop(-1)
          flag = False

        if not flag:
          print("Truncated with length %d" % original_len)

        if "pythia" in self.args.model_type:
          tokenized_sen = ["<|endoftext|>"] + claim + evidence + answer
        else:
          tokenized_sen = ["</s>"] + claim + evidence + answer

        attn_mask = [1] * len(tokenized_sen)

        if self.args.model_type in ["bigscience/bloom-560m", "EleutherAI/pythia-410m"]:
          mask_idx_list.append(len(tokenized_sen) - 1)
        # opt
        else:
          mask_idx_list.append(len(tokenized_sen) - 1 + (128 * len(input_ids_list)))

        if len(tokenized_sen) < self.args.max_len:
          attn_mask += [0] * (self.args.max_len - len(tokenized_sen))
          tokenized_sen += [self.pad_token_ids] * (self.args.max_len - len(tokenized_sen))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sen)

        assert len(input_ids) == len(attn_mask) == self.args.max_len

        input_ids_list.append(input_ids)
        attn_mask_list.append(attn_mask)

        if idx == gold_label:
          label_list.append(1)
        else:
          label_list.append(0)

      input_ids_tensor = torch.tensor(input_ids_list).squeeze(0)
      attn_mask_tensor = torch.tensor(attn_mask_list).squeeze(0)

      mask_idx_tensor = torch.tensor(mask_idx_list)

      label_tensor = torch.tensor(label_list)  # for binary classification

      return {"input_ids": input_ids_tensor, "attention_mask": attn_mask_tensor, "idx": idx, "mask_idx": mask_idx_tensor}


    else:
      feature_point = self.features[idx]
      return {"input_ids": torch.tensor(feature_point[0]), "attention_mask": torch.tensor(feature_point[1]), "idx": idx}


class fcDataset(Dataset):

  def __init__(self, filepath, tokenizer, args):
    self.filename = filepath
    self.tokenizer = tokenizer
    examples, label_dict = CREATE_FUNCTION[args.dataset](self.filename)
    self.label_dict = label_dict
    self.examples = examples

    self.args = args
    self.pad_token_ids = self.tokenizer.pad_token

    self.features = self.convert_examples_to_features(self.examples)

  def __len__(self):
    return len(self.examples)

  def get_test_features(self, example):
    prompt = prompt_dataset[self.args.dataset]
    claim = "/ Claim : %s" % example.claim
    evidence = "/ Evidence : %s" % example.evidence

    input_sentence = prompt + claim + evidence + "/ Answer is"
    target_label = self.label_dict[example.label]

    tokenized_sen = ["</s>"] + self.tokenizer.tokenize(input_sentence)

    input_ids_list = self.tokenizer.convert_tokens_to_ids(tokenized_sen)

    input_ids = torch.tensor(input_ids_list)
    gold_labels = torch.tensor(target_label)

    return input_ids, gold_labels

  def convert_examples_to_features(self, examples):

    features = []
    for example in examples:
      # prompt = prompt_dataset[self.args.dataset]
      if self.args.dataset in ["averitec", "fever"]:
        claim = self.tokenizer.tokenize("Claim : %s" % example.claim)
        evidence = self.tokenizer.tokenize("/ Evidence : %s" % example.evidence)
        answer = self.tokenizer.tokenize("/ Answer :")
        labels = self.label_dict[example.label]


      elif self.args.dataset == "snli":
        claim = self.tokenizer.tokenize("Premise : %s" % example.claim)
        evidence = self.tokenizer.tokenize("/ Hypothesis : %s" % example.evidence)
        answer = self.tokenizer.tokenize("/ Answer :")
        labels = example.label

      elif self.args.dataset == "csqa":
        claim = self.tokenizer.tokenize("Question : %s" % example.claim)
        evidence = self.tokenizer.tokenize("/ Choices : %s" % example.evidence)
        answer = self.tokenizer.tokenize("/ Answer :")
        labels = self.label_dict[example.label]

      while len(claim) + len(evidence) + len(answer) > (self.args.max_len - 1):
        evidence.pop(-1)

      tokenized_sen = ["</s>"] + claim + evidence + answer
      attn_mask = [1] * len(tokenized_sen)

      if len(tokenized_sen) < self.args.max_len:
        attn_mask += [0] * (self.args.max_len - len(tokenized_sen))
        tokenized_sen += [self.pad_token_ids] * (self.args.max_len - len(tokenized_sen))

      input_ids_list = self.tokenizer.convert_tokens_to_ids(tokenized_sen)

      # should return tensors
      input_ids = torch.tensor(input_ids_list)
      attn_mask = torch.tensor(attn_mask)

      assert len(input_ids) == len(attn_mask) == self.args.max_len

      features.append([input_ids, attn_mask, torch.tensor(labels)])

    return features

  def __getitem__(self, idx):
    feature_point = self.features[idx]
    return {"input_ids": feature_point[0], "attention_mask": feature_point[1], "labels": feature_point[2]}

    # else:
    #   input_ids, gold_labels = self.get_test_features(self.examples[idx])
    #   return input_ids, gold_labels