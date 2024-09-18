#!/bin/bash
# normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=24000M
# we run on the gpu partition and we allocate 2 titanx gpus
#SBATCH -p gpu
#We expect that our program should not run longer than 4 hours
#Note that a program will be killed once it exceeds this time!
#SBATCH --time=7-00:00:00

# map_mode : top-1, top-n, all, voting (to obtain highly influential train insts)
MAP_MODE="top-n"

NEURON_F="./results_neuron/csqa_opt125m"


# filename for saving the result
#NA_INF_INSTS="../data/commonsense_qa/neuron_influential_opt125m.json"
NA_INF_INSTS="-"
MAP_OUTFILE="-"
DATASET="csqa"

INF_RESULTS="../influence-func/fc-influ-func/csqa_opt125m/grad_sim.pickle"
# DATASET : averitec / mnli / csqa

# map_mode : top_n && threshold > 1
THRESHOLD=2

python neuron_analysis_feb.py \
  --neuron_folder ${NEURON_F} \
  --map_mode ${MAP_MODE} \
  --threshold ${THRESHOLD} \
  --na_inf_outfile ${NA_INF_INSTS} \
  --mapping_outfile ${MAP_OUTFILE} \
  --inf_result ${INF_RESULTS} \
  --dataset ${DATASET}