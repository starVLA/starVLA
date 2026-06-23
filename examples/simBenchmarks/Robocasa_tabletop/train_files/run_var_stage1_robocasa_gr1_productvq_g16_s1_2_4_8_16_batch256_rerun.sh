#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" starVLA/training/train_var_stage1.py \
  --config_yaml examples/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun.yaml
