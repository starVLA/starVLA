#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_MODE="${WANDB_MODE:-disabled}"

NUM_PROCESSES=8 # 卡的数量
CONFIG_YAML="examples/calvin/train_files/starvla_train_calvin_abc_cosmopredict2_gr00t_online.yaml"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-cosmopredict2_gr00t_calvin_abc_${RUN_TIMESTAMP}}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name CosmoPredict2GR00T \
  --datasets.vla_data.data_root_dir . \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --run_root_dir results/Checkpoints \
  --run_id "${RUN_ID}"