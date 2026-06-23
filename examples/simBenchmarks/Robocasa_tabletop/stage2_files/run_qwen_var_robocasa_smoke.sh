#!/usr/bin/env bash
set -euo pipefail

CONFIG_YAML="${CONFIG_YAML:-examples/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_epoch027_smoke.yaml}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/python}"

"${PYTHON_BIN}" starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}"
