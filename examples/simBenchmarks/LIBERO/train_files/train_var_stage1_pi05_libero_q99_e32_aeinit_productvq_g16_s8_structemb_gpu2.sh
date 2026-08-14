#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="/home/zhangfeihong/starVLA:${PYTHONPATH:-}"

/home/zhangfeihong/miniconda3/envs/starVLA/bin/python starVLA/training/train_var_stage1.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s8_structemb.yaml
