#!/usr/bin/env bash
# Single-GPU walk-through for G1 SONIC/Dex3 QwenOFT training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../../../../" && pwd)"

cd "${ROOT_DIR}"

Framework_name=QwenOFT
freeze_module_list=''
config_yaml=./examples/realRobots/UnitreeG1_WholeBody/step2_training/train_files/starvla_qwenoft_g1_sonic.yaml
data_root_dir=playground/Datasets/UnitreeG1_WholeBody
data_mix=unitree_g1_test_sonic
run_root_dir=./results/Checkpoints
run_id=starvla_qwenoft_g1_sonic_smoke
num_processes=1
BATCH=${BATCH:-1}
MAX_STEPS=${MAX_STEPS:-1000}
SAVE_EVERY=${SAVE_EVERY:-200}
EVAL_EVERY=${EVAL_EVERY:-500}
LOG_EVERY=${LOG_EVERY:-10}

mkdir -p "${run_root_dir}/${run_id}"
cp "${SCRIPT_DIR}/run_starvla_qwenoft_g1_sonic_train.sh" "${run_root_dir}/${run_id}/run_starvla_qwenoft_g1_sonic_train.sh"
cp "${config_yaml#./}" "${run_root_dir}/${run_id}/$(basename "${config_yaml}")"

export WANDB_MODE=${WANDB_MODE:-disabled}
export PYTHONPATH="${ROOT_DIR}/starVLA:${PYTHONPATH:-}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
    --config_yaml "${config_yaml}" \
    --datasets.vla_data.per_device_batch_size "${BATCH}" \
    --trainer.max_train_steps "${MAX_STEPS}" \
    --trainer.save_interval "${SAVE_EVERY}" \
    --trainer.logging_frequency "${LOG_EVERY}" \
    --trainer.eval_interval "${EVAL_EVERY}" \
    --run_root_dir "${run_root_dir}" \
    --run_id "${run_id}" \
    --wandb_project starVLA_unitree_g1_wholebody
