#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${CACHE_OUTPUT:-playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/stage2_token_cache_full.pt}"

"${PYTHON_BIN}" starVLA/training/build_var_stage2_token_cache.py \
  --config_yaml examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb.yaml \
  --stage1_artifact playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/best_recon.ckpt \
  --output "${OUTPUT}" \
  --mode train \
  --device "${CACHE_DEVICE:-cuda}" \
  --batch_size "${CACHE_BATCH_SIZE:-256}" \
  --num_workers "${CACHE_NUM_WORKERS:-8}" \
  --max_batches "${CACHE_MAX_BATCHES:-0}"
