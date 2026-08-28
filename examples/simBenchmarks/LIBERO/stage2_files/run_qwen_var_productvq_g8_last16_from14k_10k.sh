#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g8_stage2_last16_from14k_10k_fullcache"
mkdir -p "${RUN_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/accelerate}"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_qwenpi_libero_stable.yaml \
  --main_process_port "${MAIN_PROCESS_PORT:-29525}" \
  --num_processes "${NUM_PROCESSES:-2}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g8_last16_from14k_10k.yaml
