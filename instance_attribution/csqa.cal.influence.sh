#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=20 --mem=25000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:a40:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10-00:00:00


# csqa models
#MODEL_PATH="../../FV_finetuning/results/csqa_listwise_opt/checkpoint-0"
#MODEL_PATH="../../FV_finetuning/results/csqa_listwise_bloom/checkpoint-0"
MODEL_PATH="../../FV_finetuning/results/csqa_pythia410m/checkpoint-0"


OUT_FILE="./csqa_pythia410m/inf_func.pickle"
DATASET="csqa"

#MODEL_TYPE="facebook/opt-125m"
#MODEL_TYPE="bigscience/bloom-560m"
MODEL_TYPE="EleutherAI/pythia-410m"

# IA_methods : ["inf-func", "grad-sim"]
# grad-sim : train_batch_size "1"
# facebook/opt-125m ,, bigscience/bloom-560m

# opt-125m / inf func / parameters
# averitec (common) : damping 0.04 / lissa_depth 0.3 / scale 1e4 / batch size 8
# snli specific : lissa_depth 0.2

# bloom-560m / inf func / parameters
# averitec (common) : damping 0.05 / lissa_depth 0.2 / scale 1e5 / batch size 2
# snli specific : damping 0.065 / lissa_depth 0.07 / scale 1e6 / batch size 4

python csqa_main.py \
  --model_name_or_path ${MODEL_TYPE} \
  --max_len 128 \
  --model_path ${MODEL_PATH} \
  --output_file ${OUT_FILE} \
  --dataset ${DATASET} \
  --IA_method "grad-sim" \
  --train_batch_size 1 \
  --lissa_depth 0.15 \
  --damping 0.05 \
  --target_param "linear" \
  --scale 1e5
