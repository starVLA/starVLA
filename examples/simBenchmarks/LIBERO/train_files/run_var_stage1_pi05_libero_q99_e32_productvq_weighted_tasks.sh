#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" starVLA/training/train_var_stage1.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_productvq_weighted_tasks.yaml
