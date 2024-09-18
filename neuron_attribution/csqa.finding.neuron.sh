#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=24000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p copenlu --gres=gpu:a100:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=7-00:00:00


#OUTPUT_PATH="csqa_bloom560m"
OUTPUT_PATH="csqa_opt125m"

# csqa models
MODEL_PATH="../FV_finetuning/results/csqa_listwise_opt/checkpoint-0"
#MODEL_PATH="../FV_finetuning/results/csqa_listwise_bloom/checkpoint-0"
#MODEL_PATH="../FV_finetuning/results/csqa_pythia410m/checkpoint-0"


# model_type : facebook/opt-125m , bigscience/bloom-560m
MODEL_TYPE="facebook/opt-125m"
#MODEL_TYPE="bigscience/bloom-560m"
#MODEL_TYPE="EleutherAI/pythia-410m"

# averitec : train or test
# snli : train or test
# csqa : train or validation / max len 128
DATASET="csqa"
MODE="validation"

# ======= gap =======

# csqa (1000)
# train : 0 - 4 / 5 - 9
# validation : 0 - 1

for num in {0..1};
  do
    new_num=$(( $num*1000 ))
    python csqa_neuron_attr.py \
      --model_type ${MODEL_TYPE} \
      --model_path ${MODEL_PATH} \
      --output_dir ${OUTPUT_PATH} \
      --dataset ${DATASET} \
      --mode ${MODE} \
      --max_len 128 \
      --start_point ${new_num}
  done