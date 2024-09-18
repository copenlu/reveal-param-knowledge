import json, random
import logging
from torch.utils.data import Dataset
import torch
import os
import jsonlines

from datasets import load_dataset
from collections import Counter

import pdb


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


def create_snli_examples(filename, mode):

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

      print("Original %d // Filtered %d" %(len(mnli_data), len(valid_data_indices)))

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



def create_csqa_examples(filename, mode):

  if filename == "dev":
    filename = "train"

  csqa_data = load_dataset("commonsense_qa", split=filename)
  label_dict = json.load(open("../data/commonsense_qa/label_dict.json"))
  # label_dict = json.load(open("../data/commonsense_qa/binary_label_dict.json"))

  if mode == "train":
    target_indices = json.load(open("../data/commonsense_qa/train_idx.json"))

  elif mode == "dev":
    target_indices = json.load(open("../data/commonsense_qa/dev_idx.json"))

  else:
    target_indices = [i for i in range(len(csqa_data))]

  examples = []
  concept_list = []
  for idx, d in enumerate(csqa_data):

    if idx in target_indices:
      # id / question / question_concept / label / text / answerKey
      concept_list.append(d["question_concept"])
      # choice_seq = ""
      # for choice_label, choice_text in zip(d["choices"]["label"], d["choices"]["text"]):
      #   choice_seq += "(%s) %s" %(choice_label, choice_text)

      answer_idx = d["choices"]["label"].index(d["answerKey"])
      answer_text = d["choices"]["text"][answer_idx]

      # for listwise model
      examples.append(
        fcExample(claim=d["question"],
                  evidence=d["choices"]["text"],
                  label=answer_idx)
      )

      # for classification head
      # examples.append(
      #   fcExample(claim=d["question"],
      #             evidence=choice_seq,
      #             label=d["answerKey"])
      # )

      # for generating answer
      # examples.append(
      #   fcExample(claim=d["question"],
      #             evidence=choice_seq,
      #             label=answer_text)
      # )

      # for binary classification on each choice
      # for choice_label, choice_text in zip(d["choices"]["label"], d["choices"]["text"]):
      #   if choice_label == d["answerKey"]:
      #     answer_label = "correct"
      #   else:
      #     answer_label = "incorrect"
      #
      #   examples.append(
      #     fcExample(claim=d["question"],
      #               evidence=answer_text,
      #               label=answer_label)
      #   )

  # # of unique question_concepts : 2151
  print("From %s / Total %d examples are created" % (filename, len(examples)))

  # print(set(concept_list))
  # return set(concept_list)
  return examples, label_dict



def create_averitec_examples(filename, mode):
  
  data = json.load(open(filename))
  label_dict = json.load(open("../data/averitec/label_dict.json"))

  if mode == "train":
    target_indices = json.load(open("../data/averitec/train_idx.json"))

  elif mode == "dev":
    target_indices = json.load(open("../data/averitec/dev_idx.json"))
    # target_indices = target_indices[len(target_indices)//2:]

  else:
    target_indices = [i for i in range(len(data))]

  examples = []
  for idx, d in enumerate(data):
    if idx in target_indices:
      # for evidence (in QA form)
      evid = ""
      source_list = []
      for idx, qa in enumerate(d["questions"]):
        evid += "(%s)" %str(idx)
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

  print("From %s / Total %d examples are created" %(filename, len(examples)))

  return examples, label_dict


# https://huggingface.co/datasets/copenlu/fever_gold_evidence
def create_fever_examples(filename, mode):

  fever_data = load_dataset("copenlu/fever_gold_evidence", split=filename, streaming=True)
  label_dict = json.load(open("../data/fever/label_dict.json"))

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

  print("From %s / Total %d examples are created" %(filename, len(examples)))

  return examples, label_dict


def create_hans_examples(filename, mode):

  examples = []
  label_dict = json.load(open("../data/mnli/label_dict.json"))
  hans_label_dict = json.load(open("../data/hans/label_dict.json"))

  label_set = []

  with jsonlines.open("../data/hans/hans_test.jsonl") as reader:
    for line in reader:

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

  print(set(label_set))

  print("From hans testset / Total %d examples are created" %(len(examples)))

  return examples, label_dict


CREATE_FUNCTION = {
  "averitec": create_averitec_examples,
  "fever": create_fever_examples,
  "snli": create_snli_examples,
  "csqa": create_csqa_examples,
  "hans": create_hans_examples,
  "mnli": create_mnli_examples,
}

prompt_dataset = {
  "averitec": "Generate the answer to the question. There are four options for the answer, (1) Conflicting Evidence/Cherrypicking, (2) Refuted, (3) Supported, (4) Not Enough Evidence. / Question : What's the relationship between the evidence and the claim?"
}


class fcDataset(Dataset):

  def __init__(self, filepath=None, mode=None, tokenizer=None, args=None):
    self.filename = filepath
    self.mode = mode
    self.tokenizer = tokenizer
    examples, label_dict = CREATE_FUNCTION[args.dataset](self.filename, mode)
    self.label_dict = label_dict
    self.examples = examples

    self.args = args

    if "pythia" in args.model_name_or_path:
      self.pad_token_ids = "<pad>"
    else:
      self.pad_token_ids = self.tokenizer.pad_token

    if self.args.inf_filtering and self.mode == "train":
      if self.args.inf_baseline and self.mode == "train":
        # num = len(json.load(open("../data/%s/%s" %(args.dataset, args.inf_file))))

        cutoff = args.inf_inst_num
        # if deals with percentage
        if isinstance(args.inf_inst_num, float):
          cutoff = int(args.inf_inst_num * len(examples))

        self.target_inst_indices = random.sample(range(len(examples)), k=cutoff)
        print("Randomly select", len(self.target_inst_indices))

      else:
        if args.dataset == "csqa":
          self.target_inst_indices = json.load(open("../data/%s/%s" % ("commonsense_qa", args.inf_file)))
        else:
          self.target_inst_indices = json.load(open("../data/%s/%s" %(args.dataset, args.inf_file)))

        # IA results
        if isinstance(self.target_inst_indices, dict):
          # reverse = True for the least influential
          if args.most_least == "most":
            reverse_flag = False
          else:
            reverse_flag = True

          sorted_dict = sorted(self.target_inst_indices.items(), key=lambda x:x[1], reverse=reverse_flag)

          cutoff = args.inf_inst_num
          # if deals with percentage
          if isinstance(args.inf_inst_num, float):
            cutoff = int(args.inf_inst_num * len(examples))

          self.target_inst_indices = [int(i[0]) for i in sorted_dict[:cutoff]]

        # NA results
        elif isinstance(self.target_inst_indices, list):
          freq_instances = Counter(self.target_inst_indices)

          cutoff = args.inf_inst_num
          # if deals with percentage
          if isinstance(args.inf_inst_num, float):
            cutoff = int(args.inf_inst_num * len(examples))

          if args.most_least == "most":
            self.target_inst_indices = [i[0] for i in freq_instances.most_common(cutoff)]
          
          # least influential
          else:
            # self.target_inst_indices = [i[0] for i in freq_instances.most_common()[-args.inf_inst_num:]]
            # least_insts = [i for i in range(len(examples)) if i not in self.target_inst_indices]
            #
            # print("Least influential training instance pool : %d" %(len(least_insts)))
            #
            # if len(least_insts) < cutoff:
            #   # n least common elements - c.most_common()[:-n-1:-1]
            #   plus = [i[0] for i in freq_instances.most_common()[:-len(freq_instances)-1:-1] if i[1] == 1]
            #   least_insts += plus
            #
            # pdb.set_trace()
            # self.target_inst_indices = random.sample(least_insts, k=cutoff)


            self.target_inst_indices = [i[0] for i in freq_instances.most_common()[:-cutoff-1:-1]]
            
        print("Influential training instances", len(self.target_inst_indices))


    if self.args.mispred_training and self.mode == "train":
      if args.dataset == "csqa":
        excl_list = json.load(open("../data/%s/%s" % ("commonsense_qa", args.mispred_exclude_list)))
      else:
        excl_list = json.load(open("../data/%s/%s" % (args.dataset, args.mispred_exclude_list)))

      print("# of excluded training instance:", len(excl_list))

      self.target_inst_indices = [i for i in range(len(self.examples)) if i not in excl_list]

      print("Total : %d / Selected : %d" %(len(self.examples), len(self.target_inst_indices)))


    #
    # if args.dataset == "csqa":
    #   self.features = self.gen_convert_examples_to_features(self.examples)

    # else:
    if args.dataset != "csqa":
      self.features = self.convert_examples_to_features(self.examples)

    # csqa case
    else:
      if args.inf_filtering and self.mode == "train":
        features = []
        for idx, ex in enumerate(self.examples):
          if idx in self.target_inst_indices:
            features.append(ex)

        self.features = features
      else:
        self.features = self.examples


  def __len__(self):
    return len(self.features)


  def convert_examples_to_features(self, examples):

    features = []

    self.filtered_examples = []
    for idx, example in enumerate(examples):

      if self.mode == "train":
        if self.args.inf_filtering and idx not in self.target_inst_indices:
          continue
        elif self.args.mispred_training and idx not in self.target_inst_indices:
          continue
        else:
          self.filtered_examples.append(example)

      # prompt = prompt_dataset[self.args.dataset]
      if self.args.dataset in ["averitec", "fever"]:
        claim = self.tokenizer.tokenize("Claim : %s" % example.claim)
        evidence = self.tokenizer.tokenize("/ Evidence : %s" % example.evidence)
        answer = self.tokenizer.tokenize("/ Answer :")
        labels = self.label_dict[example.label]

      elif self.args.dataset in ["snli", "hans", "mnli"]:
        claim = self.tokenizer.tokenize("Premise : %s" % example.claim)
        evidence = self.tokenizer.tokenize("/ Hypothesis : %s" % example.evidence)
        answer = self.tokenizer.tokenize("/ Answer :")
        labels = example.label

      elif self.args.dataset == "csqa":
        claim = self.tokenizer.tokenize("Question : %s" % example.claim)
        evidence = self.tokenizer.tokenize("/ Choices : %s" % example.evidence)
        answer = self.tokenizer.tokenize("/ Answer :")
        labels = self.label_dict[example.label]

      flag = True
      original_len = len(claim) + len(evidence) + len(answer)

      while len(claim) + len(evidence) + len(answer) > (self.args.max_len - 1):
        # print("Truncate")
        evidence.pop(-1)
        flag = False

      if not flag:
        print("Truncated with length %d" % original_len)


      if "pythia" in self.args.model_name_or_path:
        tokenized_sen = ["<|endoftext|>"] + claim + evidence + answer
      else:
        tokenized_sen = ["</s>"] + claim + evidence + answer

      attn_mask = [1] * len(tokenized_sen)

      if len(tokenized_sen) < self.args.max_len:
        attn_mask += [0] * (self.args.max_len - len(tokenized_sen))
        tokenized_sen += [self.pad_token_ids] * (self.args.max_len - len(tokenized_sen))

      input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sen)

      assert len(input_ids) == len(attn_mask) == self.args.max_len

      features.append([input_ids, attn_mask, labels])

    if self.mode == "train":
      if self.args.inf_filtering or self.args.mispred_training:
        if len(self.target_inst_indices) < len(examples):
          assert len(self.target_inst_indices) == len(features)
        else:
          assert int(self.args.inf_inst_num * len(examples)) == len(features)

        if self.args.inf_filtering:
          print("%d number of features are used for influential training" %int(self.args.inf_inst_num * len(examples)))

        else:
          print("%d number of features are used for mispred training" %len(features))

    return features


  def get_test_features(self, examp):

    claim = self.tokenizer.tokenize("Question : %s" % examp.claim)
    evidence = self.tokenizer.tokenize("/ Choices : %s" % examp.evidence)
    answer = self.tokenizer.tokenize("/ Answer :")

    label = self.tokenizer.tokenize(examp.label)

    tokenized_sen = ["</s>"] + claim + evidence + answer
    input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_sen))

    gold_label = torch.tensor(self.tokenizer.convert_tokens_to_ids(label))

    return input_ids, gold_label


  def gen_convert_examples_to_features(self, examples):

    features = []
    for idx, example in enumerate(examples):

      if self.mode == "train":
        if self.args.inf_filtering and idx not in self.target_inst_indices:
          continue

      claim = self.tokenizer.tokenize("Question : %s" % example.claim)
      evidence = self.tokenizer.tokenize("/ Choices : %s" % example.evidence)
      answer = self.tokenizer.tokenize("/ Answer :")
      labels = self.tokenizer.tokenize(example.label)
      # labels = self.label_dict[example.label]

      flag = True
      original_len = len(claim) + len(evidence) + len(answer) + len(labels)

      while len(claim) + len(evidence) + len(answer) + len(labels) > (self.args.max_len - 1):
        # print("Truncate")
        evidence.pop(-1)
        flag = False

      if not flag:
        print("Truncated with length %d" % original_len)


      tokenized_sen = ["</s>"] + claim + evidence + answer + labels
      attn_mask = [1] * len(tokenized_sen)

      if len(tokenized_sen) < self.args.max_len:
        attn_mask += [0] * (self.args.max_len - len(tokenized_sen))
        tokenized_sen += [self.pad_token_ids] * (self.args.max_len - len(tokenized_sen))

      input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sen)

      assert len(input_ids) == len(attn_mask) == self.args.max_len

      features.append([input_ids, attn_mask, input_ids])

    if self.mode == "train":
      if self.args.inf_filtering:
        assert self.args.inf_inst_num == len(features)

    return features


  def __getitem__(self, idx):
    # if self.filename in ["validation", "dev"] and self.args.dataset == "csqa":
    #   input_ids, gold_labels = self.get_test_features(self.examples[idx])
    #   return input_ids, gold_labels

    if self.args.dataset == "csqa":
      # self.examples[idx].claim
      # self.examples[idx].evidence
      # self.examples[idx].label

      example = self.examples[idx]

      claim = self.tokenizer.tokenize("Question : %s" % example.claim)
      gold_label = torch.tensor(example.label) # for listwise

      input_ids_list = []
      attn_mask_list = []
      label_list = []
      for idx, choice in enumerate(example.evidence):
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

        if "pythia" in self.args.model_name_or_path:
          tokenized_sen = ["<|endoftext|>"] + claim + evidence + answer
        else:
          tokenized_sen = ["</s>"] + claim + evidence + answer

        attn_mask = [1] * len(tokenized_sen)

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

      input_ids_tensor = torch.tensor(input_ids_list)
      attn_mask_tensor = torch.tensor(attn_mask_list)
      label_tensor = torch.tensor(label_list) # for binary classification

      return [input_ids_tensor, attn_mask_tensor, gold_label]


    else:
      feature_point = self.features[idx]
      return {"input_ids": torch.tensor(feature_point[0]), "attention_mask": torch.tensor(feature_point[1]), "labels": torch.tensor(feature_point[2])}

    # else:
    #   input_ids, gold_labels = self.get_test_features(self.examples[idx])
    #   return input_ids, gold_labels




if __name__ == "__main__":
  logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
  # console = logging.StreamHandler()
  # console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  # console.setFormatter(formatter)
  # logging.getLogger('').addHandler(console)

  print("validation")

  vali_concept = create_csqa_examples("validation", "validation")

  print("train")

  train_concept = create_csqa_examples("train", "train")

  print(len(train_concept))
  print(len(vali_concept))
  print(len(train_concept & vali_concept))

  exit()

  filename = "../data/averitec/train.json"
  # filename = "train"

  examples, _ = create_averitec_examples(filename)

  inf_func_highly = [563, 1565, 1837, 2737, 2883, 2664, 2419, 415, 1150, 1751]
  grad_sim_highly = [563, 2737, 415, 1837, 2664, 2883, 1565, 1150, 1538, 0]

  print("Examples from influence function")
  for i in inf_func_highly:
    inst = examples[i]
    print(i)
    print(inst.claim)
    print(inst.evidence)
    print(inst.label)
    print()

  print("Examples from gradient similarity")
  for i in grad_sim_highly:
    inst = examples[i]
    print(i)
    print(inst.claim)
    print(inst.evidence)
    print(inst.label)
    print()

  # examples = create_fever_examples(filename)

  logging.info("%d number of examples created" %len(examples))