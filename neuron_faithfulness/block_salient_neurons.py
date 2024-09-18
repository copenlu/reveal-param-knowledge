import torch
import argparse

import os
import pdb
import numpy as np

from utils_na import *
from eai_patch import *
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.nn.functional as F

import random

class BlockExp:
    def __init__(self,
                 model=None,
                 model_type=None,
                 dataset=None,
                 args=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_type = model_type
        self.dataset = dataset
        self.args = args

        if "opt" in model_type:
            self.transformer_layers_attr = "model.decoder.layers"
            self.input_ff_attr = "fc1"
            self.output_ff_attr = "fc2"
            self.word_embeddings_attr = "model.decoder.embed_tokens"


        elif "bloom" in model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.dense_h_to_4h"
            self.output_ff_attr = "mlp.dense_4h_to_h"
            self.word_embeddings_attr = "transformer.word_embeddings"


        elif "pythia" in model_type:
            self.transformer_layers_attr = "gpt_neox.layers"
            self.input_ff_attr = "mlp.dense_h_to_4h"
            self.output_ff_attr = "mlp.dense_4h_to_h"
            self.word_embeddings_attr = "gpt_neox.embed_in"


    # def register_mlp_hook(self, layer_idx_list, target_pos):
    #
    #     transformer_layers = get_attributes(self.model, self.transformer_layers_attr)
    #
    #     mlp_hook_dict = {}
    #
    #     def getActivation(name):
    #         def hook(model, input, output):
    #             mlp_hook_dict[name] = output[target_pos, :]
    #         return hook
    #
    #     handle_list = []
    #     for layer_idx in layer_idx_list:
    #         ff_layer = get_attributes(transformer_layers[layer_idx], self.input_ff_attr)
    #         h = ff_layer.register_forward_hook(getActivation(layer_idx.item()))
    #         handle_list.append(h)
    #
    #     return mlp_hook_dict, handle_list


    def get_baseline_activation(self, encoded_input):

        # register hook
        # mlp_acts, handle_list = self.register_mlp_hook(layer_idx_list, target_pos)

        # forward

        if self.args.dataset == "csqa":
            outputs = self.model(input_ids=encoded_input["input_ids"].squeeze(0).to(self.device),
                                       attention_mask=encoded_input["attention_mask"].squeeze(0).to(self.device))
            probs = F.softmax(outputs.logits.detach().cpu(), dim=0).squeeze(1)

        else:
            outputs = self.model(input_ids=encoded_input["input_ids"].to(self.device),
                                 attention_mask=encoded_input["attention_mask"].to(self.device))
            probs = F.softmax(outputs.logits.detach().cpu(), dim=1)

        return probs
        # return outputs

    # main
    def run(self, encoded_input):

        neuron_info = self.dataset.neuron_list[encoded_input["idx"]]

        if self.args.attr_threshold >= 1.0:
            activated_nrs = self.use_top(neuron_info[0], neuron_info[1], new_ths=self.args.attr_threshold)

        else:
            activated_nrs = self.adjust_threshold(neuron_info[0], neuron_info[1], new_ths=self.args.attr_threshold)

        if self.args.baseline_exp:
            activated_nrs = self.random_block(num_neurons=len(activated_nrs))

        # print("Number of neurons selected :", len(activated_nrs))

        target_pos = neuron_info[2]

        layer_idx_list = activated_nrs[:, 0]
        nr_idx_list = activated_nrs[:, 1]

        # print("number of neurons modified :", len(nr_idx_list))

        # step 1 : baseline
        baseline_logits = self.get_baseline_activation(encoded_input)
        baseline_label = torch.argmax(baseline_logits).item()

        if self.args.dataset == "csqa":
            baseline_prob = baseline_logits[baseline_label].item()
        else:
            baseline_prob = baseline_logits[:, baseline_label].item()

        # step 2 : modified run (either suppressed or amplified)
        blocked_logits = self.blocked_run(encoded_input, layer_idx_list, nr_idx_list, target_pos, baseline_label)

        blocked_label = torch.argmax(blocked_logits).item()

        if self.args.dataset == "csqa":
            blocked_prob = blocked_logits[baseline_label].item()
        else:
            blocked_prob = blocked_logits[:, baseline_label].item()

        # print(baseline_prob, blocked_prob)

        result_dict = {
            "baseline_label": baseline_label,
            "blocked_label": blocked_label,
            "baseline_prob": baseline_prob,
            "blocked_prob": blocked_prob
        }

        return result_dict


    def blocked_run(self, encoded_input, layer_list, neuron_list, target_pos, baseline_label):

        if self.args.dataset == "csqa":



            self.patch_ff_layer(
                layer_list, neuron_list, encoded_input["mask_idx"][0][baseline_label], self.args.block_mode, encoded_input["mask_idx"]
            )

            outputs = self.model(input_ids=encoded_input["input_ids"].squeeze(0).to(self.device),
                                       attention_mask=encoded_input["attention_mask"].squeeze(0).to(self.device))

            self.unpatch_ff_layer(layer_list)

            probs = F.softmax(outputs.logits.detach().cpu(), dim=0).squeeze(1)

        else:
            self.patch_ff_layer(
                layer_list, neuron_list, target_pos, self.args.block_mode
            )

            outputs = self.model(input_ids=encoded_input["input_ids"].to(self.device),
                                 attention_mask=encoded_input["attention_mask"].to(self.device))

            self.unpatch_ff_layer(layer_list)

            probs = F.softmax(outputs.logits.detach().cpu(), dim=1)

        return probs
        # return outputs

    def patch_ff_layer(self, layer_list, neuron_list, target_pos, mode, mask_idx_tensor=[]):
        transformer_layers = get_attributes(self.model, self.transformer_layers_attr)

        if mode == "suppress":
            for layer_idx, neuron_idx in zip(layer_list, neuron_list):
                ff_layer = get_attributes(transformer_layers[layer_idx], self.input_ff_attr)
                set_attribute_recursive(
                    transformer_layers[layer_idx],
                    self.input_ff_attr,
                    Patch(
                        ff_layer,
                        target_pos,
                        replacement_activations=None,
                        mode=mode,
                        target_positions=[neuron_idx],
                        end_pos=mask_idx_tensor
                    )
                )

        elif mode == "sufficiency":

            layer_neuron_dict = {}

            if self.args.neuron_num != 0:
                for layer_idx, neuron_idx in zip(layer_list, neuron_list):
                    layer_neuron_dict.setdefault(layer_idx.item(), [])
                    layer_neuron_dict[layer_idx.item()].append(neuron_idx.item())

            # print(layer_neuron_dict)

            for l_idx in range(len(transformer_layers)):
                if l_idx in layer_neuron_dict:
                    neuron_idx = layer_neuron_dict[l_idx]
                else:
                    neuron_idx = []

                ff_layer = get_attributes(transformer_layers[l_idx], self.input_ff_attr)
                set_attribute_recursive(
                    transformer_layers[l_idx],
                    self.input_ff_attr,
                    Patch(
                        ff_layer,
                        target_pos,
                        replacement_activations=None,
                        mode=mode,
                        target_positions=neuron_idx,
                        end_pos=mask_idx_tensor
                    )
                )

                # pdb.set_trace()
    # end of patch_ff_layer


    def unpatch_ff_layer(self, layer_list):
        transformer_layers = get_attributes(self.model, self.transformer_layers_attr)

        if self.args.block_mode == "suppress":
            for layer_idx in layer_list:
                ff_layer = get_attributes(transformer_layers[layer_idx], self.input_ff_attr)

                set_attribute_recursive(
                    transformer_layers[layer_idx],
                    self.input_ff_attr,
                    ff_layer.ff
                )

        elif self.args.block_mode == "sufficiency":
            for layer_idx in range(len(transformer_layers)):
                ff_layer = get_attributes(transformer_layers[layer_idx], self.input_ff_attr)

                set_attribute_recursive(
                    transformer_layers[layer_idx],
                    self.input_ff_attr,
                    ff_layer.ff
                )
    # end of unpatch_ff_layer


    def random_block(self, num_neurons):

        transformer_layers = get_attributes(self.model, self.transformer_layers_attr)
        num_layers = len(transformer_layers)
        mlp_out_shape = get_attributes(transformer_layers[0], self.input_ff_attr).out_features

        layer_idx = np.random.randint(num_layers, size=num_neurons)
        neuron_idx = np.random.randint(mlp_out_shape, size=num_neurons)

        neuron_list = []
        for l, n in zip(layer_idx, neuron_idx):
            neuron_list.append([l, n])

        return torch.tensor(neuron_list)


    def use_top(self, neuron_list, attr_score, new_ths=1.0):

        neuron_ranking = torch.argsort(torch.tensor(attr_score), descending=True)
        new_neurons = torch.index_select(torch.tensor(neuron_list), 0, neuron_ranking[:self.args.neuron_num])

        return new_neurons


    def adjust_threshold(self, neuron_list, attr_score, new_ths=0.8):
        # tensor 변환 시 소숫점이 짤려서 아예 아무것도 안 걸리는 경우가 있음
        # 머 이런 거지 같은 경우가!
        tensor_attr = torch.tensor(attr_score)
        new_threshold = tensor_attr.max().item() * new_ths

        if self.args.neuron_num == -1:
            new_selected = torch.nonzero(tensor_attr > new_threshold)
            new_neurons = torch.tensor(neuron_list)[new_selected.flatten(0)]

            return new_neurons

        else:
            upper = tensor_attr.max().item() * (new_ths + 0.2)
            new_selected = torch.nonzero((tensor_attr >= new_threshold) & (tensor_attr < upper))

            new_neurons = torch.tensor(neuron_list)[new_selected.flatten(0)]

            if len(new_neurons) == 0:
                pdb.set_trace()

            # select n number of neurons

            # if not enough -> randomly select from rest of the list
            if new_neurons.size(0) < self.args.neuron_num:
                rest_indices = torch.nonzero(tensor_attr < new_threshold)
                other_pool = torch.tensor(neuron_list)[rest_indices.flatten(0)]


                try:
                    other_neurons = other_pool[torch.randint(other_pool.size(0), (1, (self.args.neuron_num-new_neurons.size(0))))]
                except:
                    pdb.set_trace()

                new_neurons = torch.cat((new_neurons, other_neurons[0]))

            elif new_neurons.size(0) > self.args.neuron_num:
                new_neurons = new_neurons[torch.randint(new_neurons.size(0), (1, self.args.neuron_num))][0]

            try:
                assert new_neurons.size(0) == self.args.neuron_num
            except:
                pdb.set_trace()

            return new_neurons


def analyze(total_result):
    diff = []
    change = 0

    for i, tr in enumerate(total_result):
        if tr["baseline_label"] != tr["blocked_label"]:
            change += 1
        else:
            # print(i)
            pass
        tmp_d = tr["baseline_prob"] - tr["blocked_prob"]

        # pdb.set_trace()
        diff.append(tmp_d)

    print("# of prediction changed %d / %d" %(change, len(total_result)))
    print("Average of 'baseline_prob - blocked_prob'", np.mean(diff))

# end of analyze


def main(args):

    if args.dataset == "csqa":
        label_dict = json.load(open("../data/commonsense_qa/label_dict.json"))
    else:
        label_dict = json.load(open("../data/%s/label_dict.json" % args.dataset))


    tokenizer, model = load_tokenizer_model(args.model_type, args.model_path, len(label_dict))

    dataset = blockDataset(tokenizer, args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    bm = BlockExp(model=model, model_type=args.model_type, dataset=dataset, args=args)

    test_iterator = tqdm(dataloader, desc="Block activation and Predict")


    total_result = []
    for step, batch in enumerate(test_iterator):
        result_dict = bm.run(batch)
        total_result.append(result_dict)

    analyze(total_result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_type", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset", type=str, default="averitec")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--neuron_folder", type=str)
    parser.add_argument("--block_mode", type=str, default="suppress")
    parser.add_argument("--neuron_num", type=int, default="-1", help="Number of neurons that are going to be used for blocking")
    parser.add_argument("--attr_threshold", type=float, default=0.7)
    parser.add_argument("--target_pos", type=str, default="seq-end")
    parser.add_argument("--inf_result", type=str, default="")
    parser.add_argument("--baseline_exp", action="store_true")

    args = parser.parse_args()
    print(args)

    main(args)