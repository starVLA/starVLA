#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

/home/zhangfeihong/miniconda3/envs/starVLA/bin/python starVLA/training/train_var_stage1.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_productvq_g8_ft_reconfocus_lr1e5.yaml
