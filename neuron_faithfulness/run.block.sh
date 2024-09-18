#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=18000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:a40:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10:00:00

# averitec - trained model
#MODEL_PATH="../FV_finetuning/results/averitec_opt125m/checkpoint-1038"
#MODEL_PATH="../FV_finetuning/results/averitec_bloom"
#MODEL_PATH="../FV_finetuning/results/averitec_pythia410m/checkpoint-2076"

# mnli - trained model
#MODEL_PATH="../FV_finetuning/results/mnli_opt125m/checkpoint-2500"
#MODEL_PATH="../FV_finetuning/results/mnli_bloom560m/checkpoint-12500"
#MODEL_PATH="../FV_finetuning/results/mnli_pythia410m/checkpoint-10000"

# csqa - trained model
#MODEL_PATH="../FV_finetuning/results/csqa_listwise_opt/checkpoint-0"
MODEL_PATH="../FV_finetuning/results/csqa_listwise_bloom/checkpoint-0"
#MODEL_PATH="../FV_finetuning/results/csqa_pythia410m/checkpoint-0"


# averitec - neuron attrib results
#NEURON_PATH="./results_neuron/averitec_opt125m"
#NEURON_PATH="./results_neuron/averitec_bloom560m"
#NEURON_PATH="./results_neuron/averitec_pythia410m"

# mnli - neuron attrib results
#NEURON_PATH="./results_neuron/mnli_opt125m"
#NEURON_PATH="./results_neuron/mnli_bloom560m"
#NEURON_PATH="./results_neuron/mnli_pythia410m"

# csqa - neuron attrib results
#NEURON_PATH="./results_neuron/csqa_opt125m"
NEURON_PATH="./results_neuron/csqa_bloom560m"
#NEURON_PATH="./results_neuron/csqa_pythia410m"


# model_type : facebook/opt-125m , bigscience/bloom-560m
#MODEL_TYPE="facebook/opt-125m"
MODEL_TYPE="bigscience/bloom-560m"
#MODEL_TYPE="EleutherAI/pythia-410m"

# averitec / snli
DATASET="csqa"

# should be bigger than 1.0 (14th Jan)
THRESHOLD=2.0

# default : neuron_num 2
# block mode : suppress (= 0.0, aka comprehensiveness) / enhance (* 2) / sufficiency

python block_salient_neurons.py \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --mode "validation" \
  --dataset ${DATASET} \
  --neuron_num 100 \
  --max_len 128 \
  --block_mode "suppress" \
  --attr_threshold ${THRESHOLD} \
  --neuron_folder ${NEURON_PATH}