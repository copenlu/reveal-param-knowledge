#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=20 --mem=25000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:titanrtx:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=10-00:00:00

# averitec models
#MODEL_PATH="../../FV_finetuning/results/averitec_opt/checkpoint-1038"
#MODEL_PATH="../../FV_finetuning/results/averitec_bloom"
#MODEL_PATH="../../FV_finetuning/results/averitec_pythia410m/checkpoint-2076"

# snli models
#MODEL_PATH="../../FV_finetuning/results/snli_opt/checkpoint-2500"
#MODEL_PATH="../../FV_finetuning/results/snli_bloom/checkpoint-5000"

# mnli models
#MODEL_PATH="../../FV_finetuning/results/mnli_opt125m/checkpoint-2500"
MODEL_PATH="../../FV_finetuning/results/mnli_bloom560m/checkpoint-12500"
#MODEL_PATH="../../FV_finetuning/results/mnli_pythia410m/checkpoint-10000"


OUT_FILE="./hans_bloom560m/grad_sim.pickle"
DATASET="hans"

#MODEL_TYPE="facebook/opt-125m"
MODEL_TYPE="bigscience/bloom-560m"
#MODEL_TYPE="EleutherAI/pythia-410m"

# IA_methods : ["inf-func", "grad-sim"]
# grad-sim : train_batch_size "1"
# facebook/opt-125m ,, bigscience/bloom-560m

# opt-125m / inf func / parameters
# averitec (common) : damping 0.04 / lissa_depth 0.3 / scale 1e4 / batch size 8
# snli specific : lissa_depth 0.2
# mnli (hans) specific : batch size 8 / lissa_depth 0.3 / damping 0.03 / scale 1e5

# bloom-560m / inf func / parameters
# averitec (common) : damping 0.05 / lissa_depth 0.2 / scale 1e5 / batch size 2
# snli specific : damping 0.065 / lissa_depth 0.07 / scale 1e6 / batch size 4
# mnli specific : damping 0.065 / lissa_depth 0.07 / scale 1e6 / batch size 2
# hans specific : batch size 2 / lissa_depth 0.07 / damping 0.05 / scale 1e6

# pythia / inf func / parameters
# averitec :  batch size 4 / lissa_depth 0.3 / damping 0.03 / scale 1e5
# hans : batch size

python main.py \
  --model_name_or_path ${MODEL_TYPE} \
  --max_len 512 \
  --model_path ${MODEL_PATH} \
  --output_file ${OUT_FILE} \
  --dataset ${DATASET} \
  --IA_method "grad-sim" \
  --train_batch_size 1 \
  --lissa_depth 0.07 \
  --damping 0.05 \
  --target_param "linear" \
  --scale 1e6
