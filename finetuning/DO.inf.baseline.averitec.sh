#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=18000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:a40:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=4-00:00:00

# dataset = ["averitec", "fever", "mnli", "csqa"]
DATASET="averitec"

#MODEL_NAME="facebook/opt-125m"
#MODEL_NAME="bigscience/bloom-560m"
MODEL_NAME="EleutherAI/pythia-410m"

INF_FILE="grad_sim_opt125m.json"

MAX_LEN=256
TRAIN_BATCH=1
MOST_LEAST="most"
SEED=60


# inf_inst_num : number of filtered instances

# bigscience/bloom-560m
# facebook/opt-125m , facebook/opt-1.3b

# --inf_baseline : for random inst training

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.1 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.3 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.5 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.7 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.9 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

SEED=73

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.1 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.3 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.5 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.7 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}

python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
  --inf_baseline \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.9 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}