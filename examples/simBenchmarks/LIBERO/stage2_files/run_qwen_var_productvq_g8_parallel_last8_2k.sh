#!/usr/bin/env bash
set -euo pipefail

cd /home/zhangfeihong/starVLA

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export PYTHONPATH="${PYTHONPATH:-/home/zhangfeihong/starVLA}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g8_stage2_parallel_last8_2k_fullcache"
mkdir -p "${RUN_DIR}"

/home/zhangfeihong/miniconda3/envs/starVLA/bin/accelerate launch \
  --num_processes 2 \
  --main_process_port 29527 \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_parallel_last8_2k.yaml
