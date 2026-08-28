#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/python}"

"${PYTHON_BIN}" starVLA/training/build_var_stage2_token_cache.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s8_structemb.yaml \
  --stage1_artifact playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s8_structemb/best_recon.ckpt \
  --output playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s8_structemb/stage2_token_cache_full.pt \
  --mode train \
  --device "${CACHE_DEVICE:-cpu}" \
  --batch_size "${CACHE_BATCH_SIZE:-512}" \
  --num_workers "${CACHE_NUM_WORKERS:-0}"
