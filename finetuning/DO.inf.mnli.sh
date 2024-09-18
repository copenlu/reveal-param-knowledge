#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=18000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:titanrtx:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=20:00:00

# dataset = ["averitec", "fever", "mnli", "csqa"]
DATASET="mnli"

MODEL_NAME="facebook/opt-125m" # train batch size 4
#MODEL_NAME="bigscience/bloom-560m" # train batch size 2
#MODEL_NAME="EleutherAI/pythia-410m" # train batch size 2


#INF_FILE="grad_sim_pythia410m.json"
#INF_FILE="inf_func_pythia410m.json"
INF_FILE="na_ia_mnli_opt125m.json"

# inf_inst_num : number of filtered instances

# bigscience/bloom-560m
# facebook/opt-125m , facebook/opt-1.3b

MOST_LEAST="most"
SEED=42

# averitec, mnli
MAX_LEN=256
TRAIN_BATCH=1

# csqa
#MAX_LEN=128


python3 main_ft.py \
  --do_train \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size ${TRAIN_BATCH} \
  --max_len ${MAX_LEN} \
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
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.9 \
  --most_least ${MOST_LEAST} \
  --n_epochs 5 \
  --seed ${SEED}