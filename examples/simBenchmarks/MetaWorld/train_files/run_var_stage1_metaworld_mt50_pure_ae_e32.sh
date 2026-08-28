#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" starVLA/training/train_var_stage1.py \
  --config_yaml examples/simBenchmarks/MetaWorld/train_files/train_var_stage1_metaworld_mt50_pure_ae_e32.yaml
