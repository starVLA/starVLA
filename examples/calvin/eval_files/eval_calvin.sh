#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export PYTHONPATH=$(pwd):${PYTHONPATH} # let Calvin client find websocket tools from main repo
export calvin_python=/path/to/your/conda/envs/calvin/bin/python

host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=results/Checkpoints/0123_starvla_qwen3_calvin_task_D_D/checkpoints/steps_30000_pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

${calvin_python} ./examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" \
    --args.port $base_port \
    --args.dataset_path /path/to/calvin/task_D_D/ \
    --args.num_sequences 1000
