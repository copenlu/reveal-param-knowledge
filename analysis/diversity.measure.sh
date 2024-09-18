#!/bin/bash

# dataset = ["averitec", "mnli", "csqa"]
DATASET="averitec"

# "facebook/opt-125m" , "bigscience/bloom-560m" , "EleutherAI/pythia-410m"
MODEL_NAME="facebook/opt-125m"

# averitec trained model
OUTPUT_DIR="./results/averitec_opt125m/checkpoint-1038"

INF_FILE="neuron_influential_averitec_opt125m_ver2.json"

# --inf_baseline : for random selection

# 42 / 60 / 73
SEED=42

python3 diversity_measurement.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --max_len 512 \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.1 \
  --most_least "most" \
  --n_epochs 5 \
  --output_dir ${OUTPUT_DIR} \
  --seed ${SEED}

python3 diversity_measurement.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --max_len 512 \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.1 \
  --most_least "least" \
  --output_dir ${OUTPUT_DIR} \
  --seed ${SEED}