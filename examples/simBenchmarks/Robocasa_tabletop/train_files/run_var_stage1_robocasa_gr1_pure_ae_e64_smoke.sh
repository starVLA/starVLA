#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/zhangfeihong/starVLA:${PYTHONPATH:-}"

/home/zhangfeihong/miniconda3/envs/starVLA/bin/python starVLA/training/train_var_stage1.py \
  --config_yaml examples/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_pure_ae_e64_smoke.yaml
