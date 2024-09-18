# main knowledge neurons class
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from tqdm import tqdm
import numpy as np
import collections
from typing import List, Optional, Tuple, Callable
import math
from functools import partial
from transformers import PreTrainedTokenizerBase

import random
from eai_patch import *

import pdb


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        model_type: str = "bert",
        device: str = None,
        args = None
    ):
        self.model = model
        self.model_type = model_type
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.args = args


        if args.dataset == "csqa":
            self.baseline_activations = []
        else:
            self.baseline_activations = None

        self.end_pos = []

        # Refining - for diverse examples
        self.food_list = [l.strip() for l in open("./refining_srcs/foods.txt").readlines()]
        self.animal_list = [l.strip() for l in open("./refining_srcs/animals.txt").readlines()]


        if self.model_type == "bert":
            self.transformer_layers_attr = "bert.encoder.layer"
            self.input_ff_attr = "intermediate"
            self.output_ff_attr = "output.dense.weight"
            self.word_embeddings_attr = "bert.embeddings.word_embeddings.weight"
            self.unk_token = getattr(self.tokenizer, "unk_token_id", None)

        elif "gpt" in model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.c_fc"
            self.output_ff_attr = "mlp.c_proj.weight"
            self.word_embeddings_attr = "transformer.wpe"

        elif "bloom" in model_type:
            self.transformer_layers_attr = "transformer.h"
            self.input_ff_attr = "mlp.dense_h_to_4h"
            self.output_ff_attr = "mlp.dense_4h_to_h"
            self.word_embeddings_attr = "transformer.word_embeddings"

            self.model_type = "gpt"
        
        # haeun
        elif "pythia" in model_type:
            self.transformer_layers_attr = "gpt_neox.layers"
            self.input_ff_attr = "mlp.dense_h_to_4h"
            self.output_ff_attr = "mlp.dense_4h_to_h"
            self.word_embeddings_attr = "gpt_neox.embed_in"

            self.model_type = "gpt"

        elif "opt" in model_type:
            self.transformer_layers_attr = "model.decoder.layers"
            self.input_ff_attr = "fc1"
            self.output_ff_attr = "fc2"
            self.word_embeddings_attr = "model.decoder.embed_tokens"

            self.model_type = "gpt"
        
        else:
            raise NotImplementedError

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        return get_attributes(self.model, self.word_embeddings_attr)

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    # https://www.vedantu.com/english/animals-name
    # https://www.vedantu.com/english/food-names-for-kids
    def _get_diverse(self):

        # word / sentence
        # first word
        category = random.choice(["animal", "food"])

        if category == "animal":
            insert_word = random.choice(self.animal_list)

        elif category == "food":
            insert_word = random.choice(self.food_list)

        return insert_word


    def _prepare_inputs(self, prompt, target=None, encoded_input=None):

        if self.args.dataset == "csqa":
            question = self.tokenizer.tokenize("Question : %s" %(self.examp_obj.claim))

            input_ids_list = []
            attn_mask_list = []
            end_pos = []
            for idx, choice in enumerate(self.examp_obj.evidence):
                choice = self.tokenizer.tokenize("/ Choices : %s" %(choice))
                answer = self.tokenizer.tokenize("/ Answer :")

                flag = True
                original_len = len(question) + len(choice) + len(answer)

                while len(question) + len(choice) + len(answer) > (self.args.max_len - 1):
                    evidence.pop(-1)
                    flag = False

                if not flag:
                    print("Truncated with length %d" %original_len)

                tokenized_sen = ["</s>"] + question + choice + answer
                attn_mask = [1] * len(tokenized_sen)

                if self.args.model_type in ["bigscience/bloom-560m", "EleutherAI/pythia-410m"]:
                    end_pos.append(len(tokenized_sen) - 1)
                # opt
                else:
                    end_pos.append(len(tokenized_sen)-1 + (128 * len(input_ids_list)))

                if len(tokenized_sen) < self.args.max_len:
                    attn_mask += [0] * (self.args.max_len - len(tokenized_sen))
                    tokenized_sen += [self.tokenizer.pad_token] * (self.args.max_len - len(tokenized_sen))

                input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sen)

                assert len(input_ids) == len(attn_mask) == self.args.max_len

                input_ids_list.append(input_ids)
                attn_mask_list.append(attn_mask)

            input_ids_tensor = torch.tensor(input_ids_list).to(self.device)
            attn_mask_tensor = torch.tensor(attn_mask_list).to(self.device)

            encoded_input = {
                "input_ids": input_ids_tensor, "attention_mask": attn_mask_tensor
            }

            self.end_pos = end_pos


        elif self.args.diverse:
            random_insert = self.tokenizer.tokenize(self.to_insert)

            claim = self.tokenizer.tokenize("Claim : %s" % self.examp_obj.claim)
            evidence = self.tokenizer.tokenize("/ Evidence : %s" % self.examp_obj.evidence)
            answer = self.tokenizer.tokenize("/ Answer :")

            while len(claim) + len(evidence) + len(answer) > (self.args.max_len - 1 -len(random_insert)):
                print("Truncation with len :", (len(claim) + len(evidence) + len(answer)))
                evidence.pop(-1)

                if self.args.target_pos == "evi-end":
                    self.target_position -= 1

            input_sen = ["</s>"] + claim + evidence + answer

            # insert random word
            insert_pos = random.choice(range(1, len(input_sen)-len(answer)))

            input_sen = input_sen[:insert_pos] + random_insert + input_sen[insert_pos:]
            attn_mask = [1] * len(input_sen)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_sen)

            # haeun : we dont need to pad
            # assert len(input_ids) == len(attn_mask)  self.args.max_len

            encoded_input = {"input_ids": torch.tensor(input_ids).unsqueeze(0).to(self.device),
                             "attention_mask": torch.tensor(attn_mask).unsqueeze(0).to(self.device)}


        elif self.args.mode == "single":
            # continue here
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)


        elif encoded_input is None:
            # original
            # encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            claim = self.tokenizer.tokenize("Claim : %s" %self.examp_obj.claim)
            evidence = self.tokenizer.tokenize("/ Evidence : %s" %self.examp_obj.evidence)
            answer = self.tokenizer.tokenize("/ Answer :")

            # prompt ver2
            # claim = self.tokenizer.tokenize("/ Claim : %s" %self.examp_obj.claim)
            # evidence = self.tokenizer.tokenize("Evidence : %s" %self.examp_obj.evidence)
            # answer = self.tokenizer.tokenize("/ Answer :")
            # -----------

            while len(claim) + len(evidence) + len(answer) > (self.args.max_len -1):
                print("Truncation with len :", (len(claim) + len(evidence) + len(answer)))
                evidence.pop(-1)

                if self.args.target_pos == "evi-end":
                    self.target_position -= 1

            input_sen = ["</s>"] + claim + evidence + answer

            # prompt ver2
            # input_sen = ["</s>"] + evidence + claim + answer
            # -----------

            attn_mask = [1] * len(input_sen)

            input_ids = self.tokenizer.convert_tokens_to_ids(input_sen)

            # haeun : we dont need to pad
            # assert len(input_ids) == len(attn_mask)  self.args.max_len

            encoded_input = {"input_ids": torch.tensor(input_ids).unsqueeze(0).to(self.device),
                             "attention_mask": torch.tensor(attn_mask).unsqueeze(0).to(self.device)}

        if self.model_type == "bert":
            mask_idx = torch.where(
                encoded_input["input_ids"][0] == self.tokenizer.mask_token_id
            )[0].item()
        else:
            # with autoregressive models we always want to target the last token
            mask_idx = self.target_position

        if target is not None:
            if "bert" in self.model_type:
                # original
                # target = self.tokenizer.encode(target)
                target = self.tokenizer.convert_tokens_to_ids(target)
            else:
                # haeun : we are doing text classification w/ gpt
                # the target is the idx of label
                target = [target]

        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        if self.model_type == "gpt":
            n_sampling_steps = len(target_label)
        else:
            n_sampling_steps = 1  # TODO: we might want to use multiple mask tokens even with bert models

        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""


        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input).logits
            probs = F.softmax(outputs[:, mask_idx, :], dim=-1)
            if n_sampling_steps > 1:
                target_idx = target_label[i]
            else:
                target_idx = target_label
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            all_argmax_probs.append(argmax_prob)

            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if self.model_type == "bert":
            return self.model.config.intermediate_size
        else:
            return self.model.config.hidden_size * 4

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu", dataset=None):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """

        if dataset == "csqa":
            out = []
            for act in activations:
                if len(act.shape) == 1:
                    tiled_activations = einops.repeat(act, "d -> (r) d", r=steps)
                    tmp_out = (
                            tiled_activations
                            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
                    )

                else:
                    tiled_activations = einops.repeat(act, "b d -> (r b) d", r=steps)
                    tmp_out = (
                            tiled_activations
                            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
                    )
                out.append(tmp_out)

            out = torch.stack(out)
            # out.shape = (5, 20, 3072)

        else:
            if len(activations.shape) == 1:
                tiled_activations = einops.repeat(activations, "d -> (r) d", r=steps)
                out = (
                        tiled_activations
                        * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
                )

            else:
                tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
                out = (
                    tiled_activations
                    * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
                )
        return out

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                if len(acts.shape) == 2:
                    # if self.args.dataset == "csqa":
                    #     indices = torch.tensor(self.end_pos).to(self.device)
                    #     self.baseline_activations = torch.index_select(acts, 0, indices)
                    # else:
                    self.baseline_activations = acts[mask_idx, :]
                else:
                    # if self.args.dataset == "csqa":
                    #
                    #     selected_acts = []
                    #     for idx, end_pos in enumerate(self.end_pos):
                    #         tmp = torch.index_select(acts[idx], 0, torch.tensor(end_pos).to(self.device))
                    #         selected_acts.append(tmp)
                    #
                    #     self.baseline_activations = torch.stack(selected_acts)
                    #     # shape : (5, 1, 4096)
                    # else:
                    self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        
        # baseline_outputs : CausalLMOutputWithPast (gpt-neo)
        # baseline_outputs[0].shape (logits): (1, 192, 50257)
        # baseline_outputs[1].shape (past_key_values): 12개

        baseline_outputs = self.model(**encoded_input).logits

        # haeun : to calculate the neuron's attribution on predicted token
        max_prob_label = torch.argmax(baseline_outputs).item()

        # baseline_activation is from hook_fn (last token of the input - the answer "is")
        baseline_activations = self.baseline_activations

        if self.args.dataset == "csqa":
            self.baseline_activations = []
        else:
            self.baseline_activations = None

        return baseline_outputs, baseline_activations, max_prob_label

    def get_scores(
        self,
        prompt,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """

        scores = []
        # encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        encoded_input, mask_idx, target = self._prepare_inputs(prompt, ground_truth)

        # for layer_idx in tqdm(
        #     range(self.n_layers()),
        #     desc="Getting attribution scores for each layer...",
        #     disable=not pbar,
        # ):
        for layer_idx in range(self.n_layers()):

            layer_scores, max_baseline_label = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
            )

            scores.append(layer_scores)

        return torch.stack(scores), max_baseline_label

    def get_coarse_neurons(
        self,
        prompt,
        examp_obj,
        ground_truth: str,
        target_position = -1,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
    ) -> List[List[int]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        self.target_position = target_position

        self.examp_obj = examp_obj

        self.to_insert = self._get_diverse()

        attribution_scores, max_baseline_label = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
        )
        # haeun: attribution_scores.shape = (num_layer, activation_size) = (12, 3072)

        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"

        if adaptive_threshold is not None:
            threshold = attribution_scores.max().item() * adaptive_threshold
        
        if threshold is not None:
            selected = torch.nonzero(attribution_scores > threshold).cpu()
            scores_for_selected = attribution_scores[selected[:, 0], selected[:, 1]].detach().cpu().tolist()
            return selected.tolist(), scores_for_selected, max_baseline_label
        
        else:
            s = attribution_scores.flatten().detach().cpu().numpy()
            return (
                torch.nonzero(attribution_scores > np.percentile(s, percentile))
                .cpu()
                .tolist()
            ), attribution_scores

    def get_refined_neurons(
        self,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = 0.3,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        quiet=False,
    ) -> List[List[int]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.

        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        for prompt in tqdm(
            prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            coarse_neurons.append(
                self.get_coarse_neurons(
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                )
            )
        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                negative_examples,
                desc="Getting coarse neurons for negative examples",
                disable=quiet,
            ):
                negative_neurons.append(
                    self.get_coarse_neurons(
                        negative_example,
                        ground_truth,
                        batch_size=batch_size,
                        steps=steps,
                        adaptive_threshold=coarse_adaptive_threshold,
                        threshold=coarse_threshold,
                        percentile=coarse_percentile,
                        pbar=False,
                    )
                )
        if not quiet:
            total_coarse_neurons = sum([len(i) for i in coarse_neurons])
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        refined_neurons = []
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        for neuron, count in c.items():
            if count > t:
                refined_neurons.append(list(neuron))

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            print(f"{total_refined_neurons} neurons remaining after refining")
        return refined_neurons


    def get_scores_for_layer(
        self,
        prompt: str,
        ground_truth: str,
        layer_idx: int,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        """
        assert steps % batch_size == 0
        n_batches = steps // batch_size

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(prompt, ground_truth, encoded_input)

        # for autoregressive models, we might want to generate > 1 token
        if self.model_type == "gpt":
            n_sampling_steps = len(target_label)
        else:
            n_sampling_steps = 1  # TODO: we might want to use multiple mask tokens even with bert models


        # haeun : only csqa / to select the sequence
        baseline_prob = self.model(**encoded_input).logits
        max_seq_idx = torch.argmax(baseline_prob).item()

        new_encoded_input = {"input_ids": encoded_input["input_ids"][max_seq_idx].unsqueeze(0),
                             "attention_mask": encoded_input["attention_mask"][max_seq_idx].unsqueeze(0)}


        # new_encoded_input


        if attribution_method == "integrated_grads":
            integrated_grads = []

            # n_sampling_steps : 1

            for i in range(n_sampling_steps):
                if i > 0 and self.model_type == "gpt":
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )

                # first calculating baseline_activation
                (baseline_outputs, baseline_activations, max_baseline_label) = self.get_baseline_with_activations(
                    new_encoded_input, layer_idx, mask_idx)

                max_baseline_label = max_seq_idx

                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1

                # if self.args.dataset == "csqa":
                #     scaled_weights = self.scaled_input(
                #         baseline_activations, steps=steps, device=self.device, dataset="csqa"
                #     )
                #     chunk_dim = 1
                # else:
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                chunk_dim = 0


                scaled_weights.requires_grad_(True)

                # scaled_weights : (step, intermediate_mlp_activation_size)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches, dim=chunk_dim):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            new_encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            new_encoded_input["attention_mask"],
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }
                    if self.model_type == "bert":
                        inputs["token_type_ids"] = einops.repeat(
                            new_encoded_input["token_type_ids"],
                            "b d -> (r b) d",
                            r=batch_size,
                        )


                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                        end_pos=self.end_pos
                    )

                    # then forward through the model to get the logits

                    outputs = self.model(**inputs).logits

                    # then calculate the gradients for each step w/r/t the inputs
                    # probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1) # original
                    if self.args.dataset == "csqa":
                        probs = outputs[0]
                        # probs = F.softmax(outputs.squeeze(0), dim=0)
                    else:
                        probs = F.softmax(outputs, dim=-1)

                    # Original
                    if n_sampling_steps > 1:
                        target_idx = target_label[i]
                    else:
                        target_idx = target_label

                    # print("target", target_idx, "max_token", max_token)
                    # pdb.set_trace()

                    # haeun: torch.autograd.grad
                    # Computes and returns the sum of gradients of outputs with respect to the inputs
                    # haeun: torch.unbind -> 차원 축소

                    # haeun: compute gradient with predicted label
                    # if self.args.dataset == "csqa":
                    #     grad = torch.autograd.grad(
                    #         torch.unbind(probs[max_baseline_label]), batch_weights
                    #     )[0]
                    #     grad = grad[max_baseline_label]
                    # else:

                    # needs to be checked

                    grad = torch.autograd.grad(probs, batch_weights)[0]



                    # original
                    # grad = torch.autograd.grad(
                    #     torch.unbind(probs[:, target_idx]), batch_weights
                    # )[0]

                    # haeun: grad.shape (1, 3072) - activation size

                    grad = grad.sum(dim=0)
                    # grad.shape : (3072) - sum으로 차원 축소

                    integrated_grads_this_step.append(grad.detach())

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )
                
                # integrated_grads_this_step len : 20 (num of steps)

                # then (1) sum, and (2) multiply by W-hat / m
                # (1)
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)

                # element-wise sum -> shape (3072)

                # (2) - Riemann approximation
                # if self.args.dataset == "csqa":
                #     integrated_grads_this_step *= baseline_activations[max_baseline_label].squeeze(0) / steps
                # else:
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps

                integrated_grads.append(integrated_grads_this_step)

                if n_sampling_steps > 1:
                    prompt += next_token_str

            # haeun: list type, integrated_grads, len - 1 (only one token as output)
            # this part is for normalization with multiple tokens output
            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )

            return integrated_grads, max_baseline_label

        elif attribution_method == "max_activations":
            activations = []
            for i in range(n_sampling_steps):
                if i > 0 and self.model_type == "gpt":
                    # retokenize new inputs
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations, _
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError

    def modify_activations(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        mode: str = "suppress",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the groundtruth being generated + the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch so we can unpatch them later
        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def suppress_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
        and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="suppress",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def enhance_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    @torch.no_grad()
    def modify_weights(
        self,
        prompt: str,
        neurons: List[List[int]],
        target: str,
        mode: str = "edit",
        erase_value: str = "zero",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        Update the *weights* of the neural net in the positions specified by `neurons`.
        Specifically, the weights of the second Linear layer in the ff are updated by adding or subtracting the value
        of the word embeddings for `target`.
        """
        assert mode in ["edit", "erase"]
        assert erase_value in ["zero", "unk"]
        results_dict = {}

        _, _, target_label = self._prepare_inputs(prompt, target)
        # get the baseline probabilities of the target being generated + the argmax / greedy completion before modifying the weights
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # get the word embedding values of the baseline + target predictions
        word_embeddings_weights = self._get_word_embeddings()
        if mode == "edit":
            assert (
                self.model_type == "bert"
            ), "edit mode currently only working for bert models - TODO"
            original_prediction_id = argmax_tokens[0]
            original_prediction_embedding = word_embeddings_weights[
                original_prediction_id
            ]
            target_embedding = word_embeddings_weights[target_label]

        if erase_value == "zero":
            erase_value = 0
        else:
            assert self.model_type == "bert", "GPT models don't have an unk token"
            erase_value = word_embeddings_weights[self.unk_token]

        # modify the weights by subtracting the original prediction's word embedding
        # and adding the target embedding
        original_weight_values = []  # to reverse the action later
        for layer_idx, position in neurons:
            output_ff_weights = self._get_output_ff_layer(layer_idx)
            if self.model_type == "gpt2":
                # since gpt2 uses a conv1d layer instead of a linear layer in the ff block, the weights are in a different format
                original_weight_values.append(
                    output_ff_weights[position, :].detach().clone()
                )
            else:
                original_weight_values.append(
                    output_ff_weights[:, position].detach().clone()
                )
            if mode == "edit":
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] -= original_prediction_embedding * 2
                    output_ff_weights[position, :] += target_embedding * 2
                else:
                    output_ff_weights[:, position] -= original_prediction_embedding * 2
                    output_ff_weights[:, position] += target_embedding * 2
            else:
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] = erase_value
                else:
                    output_ff_weights[:, position] = erase_value

        # get the probabilities of the target being generated + the argmax / greedy completion after modifying the weights
        (
            new_gt_prob,
            new_argmax_prob,
            new_argmax_completion_str,
            new_argmax_tokens,
        ) = self._generate(prompt, target)
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        def unpatch_fn():
            # reverse modified weights
            for idx, (layer_idx, position) in enumerate(neurons):
                output_ff_weights = self._get_output_ff_layer(layer_idx)
                if self.model_type == "gpt2":
                    output_ff_weights[position, :] = original_weight_values[idx]
                else:
                    output_ff_weights[:, position] = original_weight_values[idx]

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def edit_knowledge(
        self,
        prompt: str,
        target: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="edit",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def erase_knowledge(
        self,
        prompt: str,
        neurons: List[List[int]],
        erase_value: str = "zero",
        target: Optional[str] = None,
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="erase",
            erase_value=erase_value,
            undo_modification=undo_modification,
            quiet=quiet,
        )
