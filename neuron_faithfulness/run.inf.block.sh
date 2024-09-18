#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=18000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:a40:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=24:00:00

# averitec - trained model
#MODEL_PATH="../FV_finetuning/results/averitec_opt125m/checkpoint-1038"
#MODEL_PATH="../FV_finetuning/results/averitec_bloom"
#MODEL_PATH="../FV_finetuning/results/averitec_pythia410m/checkpoint-2076"

# mnli - trained model
MODEL_PATH="../FV_finetuning/results/mnli_opt125m/checkpoint-2500"
#MODEL_PATH="../FV_finetuning/results/mnli_bloom560m/checkpoint-12500"
#MODEL_PATH="../FV_finetuning/results/mnli_pythia410m/checkpoint-10000"

# csqa - trained model
#MODEL_PATH="../FV_finetuning/results/csqa_listwise_opt/checkpoint-0"
#MODEL_PATH="../FV_finetuning/results/csqa_listwise_bloom/checkpoint-0"
#MODEL_PATH="../FV_finetuning/results/csqa_pythia410m/checkpoint-0"


# csqa

#NEURON_PATH="./results_neuron/csqa_bloom560m"
#INF_PATH="../influence-func/fc-influ-func/csqa_bloom560m/inf_func.pickle"

# mnli
NEURON_PATH="./results_neuron/mnli_opt125m"
INF_PATH="../influence-func/fc-influ-func/mnli_opt125m/inf_func.pickle"

#NEURON_PATH="./results_neuron/mnli_bloom560m"
#INF_PATH="../influence-func/fc-influ-func/averitec_pythia410m/grad_sim.pickle"

#NEURON_PATH="./results_neuron/mnli_pythia410m"
#INF_PATH="../influence-func/fc-influ-func/averitec_pythia410m/grad_sim.pickle"

# averitec - neuron attrib results
#NEURON_PATH="./results_neuron/averitec_opt125m"
#INF_PATH="../influence-func/fc-influ-func/averitec_opt125m/inf_func.pickle"

#NEURON_PATH="./results_neuron/averitec_bloom560m"
#INF_PATH="../influence-func/fc-influ-func/averitec_bloom560m/inf_func.pickle"

#NEURON_PATH="./results_neuron/averitec_pythia410m"
#INF_PATH="../influence-func/fc-influ-func/averitec_pythia410m/inf_func.pickle"


# snli - neuron attrib results
#NEURON_PATH="./results_neuron/snli_opt125m"
#INF_PATH="../influence-func/fc-influ-func/snli_opt125m/grad_sim.pickle"

#NEURON_PATH="./results_neuron/snli_bloom560m"
#INF_PATH="../influence-func/fc-influ-func/snli_bloom560m/inf_func.pickle"


# model_type : facebook/opt-125m , bigscience/bloom-560m
MODEL_TYPE="facebook/opt-125m"
#MODEL_TYPE="bigscience/bloom-560m"
#MODEL_TYPE="EleutherAI/pythia-410m"

# averitec / mnli / csqa
DATASET="mnli"


# block mode : suppress (= 0.0 / comprehensiveness) / enhance (* 2) / sufficiency
THRESHOLD=2.0

python block_salient_neurons.py \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --mode "test" \
  --dataset ${DATASET} \
  --neuron_num 100 \
  --block_mode "suppress" \
  --attr_threshold ${THRESHOLD} \
  --max_len 512 \
  --neuron_folder ${NEURON_PATH} \
  --inf_result ${INF_PATH}


INF_PATH="../influence-func/fc-influ-func/csqa_bloom560m/grad_sim.pickle"

python block_salient_neurons.py \
  --model_type ${MODEL_TYPE} \
  --model_path ${MODEL_PATH} \
  --mode "test" \
  --dataset ${DATASET} \
  --neuron_num 100 \
  --block_mode "suppress" \
  --attr_threshold ${THRESHOLD} \
  --neuron_folder ${NEURON_PATH} \
  --max_len 512 \
  --inf_result ${INF_PATH}