#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=18000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:titanrtx:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=4-00:00:00

# dataset = ["averitec", "fever", "snli", "csqa", "mnli", "hans"]

DATASET="hans"
MODEL_NAME="facebook/opt-125m"
OUTPUT_DIR="./results/mnli_opt125m/checkpoint-2500"
# original learning rate : 1e-5

# bigscience/bloom-560m
# facebook/opt-125m , facebook/opt-350m , facebook/opt-1.3b
# EleutherAI/pythia-410m

python3 main_ft.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --max_len 512 \
  --n_epochs 5 \
  --output_dir ${OUTPUT_DIR}

MODEL_NAME="bigscience/bloom-560m"
OUTPUT_DIR="./results/mnli_bloom560m/checkpoint-12500"

python3 main_ft.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --max_len 512 \
  --n_epochs 5 \
  --output_dir ${OUTPUT_DIR}

MODEL_NAME="EleutherAI/pythia-410m"
OUTPUT_DIR="./results/mnli_pythia410m/checkpoint-10000"

python3 main_ft.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --max_len 512 \
  --n_epochs 5 \
  --output_dir ${OUTPUT_DIR}