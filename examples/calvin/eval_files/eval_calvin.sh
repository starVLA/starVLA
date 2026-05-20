#!/bin/bash

###########################################################################################
# === Please modify the following paths according to your environment ===
export STARVLA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA 
export calvin_python=/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/calvin_venv/bin/python
export PYTHONPATH=${STARVLA_ROOT}:${PYTHONPATH}     # ★ 关键这行
host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA/results/Checkpoints/qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview_d_style_ft_20260520_043105/checkpoints/steps_2000_pytorch_model.pt
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}

${calvin_python} ${STARVLA_ROOT}/examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" --args.port $base_port \
    --args.dataset_path /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d \
    --args.calvin_config_path /inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/calvin_models/conf \
    --args.eval_sequences_path ${STARVLA_ROOT}/examples/calvin/eval_files/eval_sequences.json \
    --args.num_sequences 100 \
    --args.seed 0 \
    --args.eval-log-dir ${LOG_DIR}

