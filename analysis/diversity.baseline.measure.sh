#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=18000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu --gres=gpu:a40:1
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=20:00:00

# dataset = ["averitec", "mnli", "csqa"]
DATASET="csqa"

# "facebook/opt-125m" , "bigscience/bloom-560m" , "EleutherAI/pythia-410m"
MODEL_NAME="EleutherAI/pythia-410m"

# averitec trained model
#OUTPUT_DIR="./results/averitec_opt125m/checkpoint-1038"
#OUTPUT_DIR="results/averitec_bloom"
#OUTPUT_DIR="./results/averitec_pythia410m/checkpoint-2076"

# mnli trained model
#OUTPUT_DIR="./results/mnli_opt125m/checkpoint-2500"
#OUTPUT_DIR="./results/mnli_bloom560m/checkpoint-12500"
#OUTPUT_DIR="./results/mnli_pythia410m/checkpoint-10000"

# csqa trained model
#OUTPUT_DIR="./results/csqa_listwise_opt/checkpoint-0"
#OUTPUT_DIR="./results/csqa_listwise_bloom/checkpoint-0"
OUTPUT_DIR="./results/csqa_pythia410m/checkpoint-0"


#INF_FILE="neuron_influential_averitec_opt125m_ver2.json"
INF_FILE="grad_sim_bloom560m.json"
#INF_FILE="inf_func_bloom560m.json"


# --inf_baseline : for random selection


python3 diversity_measurement.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --inf_baseline \
  --max_len 128 \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.1 \
  --most_least "most" \
  --n_epochs 5 \
  --output_dir ${OUTPUT_DIR} \
  --seed 42

python3 diversity_measurement.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --inf_baseline \
  --max_len 128 \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.1 \
  --most_least "most" \
  --n_epochs 5 \
  --output_dir ${OUTPUT_DIR} \
  --seed 60

python3 diversity_measurement.py \
  --do_eval \
  --model_name_or_path ${MODEL_NAME} \
  --dataset ${DATASET} \
  --learning_rate "1e-5" \
  --train_batch_size 1 \
  --inf_baseline \
  --max_len 128 \
  --inf_filtering \
  --inf_file ${INF_FILE} \
  --inf_inst_num 0.1 \
  --most_least "most" \
  --n_epochs 5 \
  --output_dir ${OUTPUT_DIR} \
  --seed 73