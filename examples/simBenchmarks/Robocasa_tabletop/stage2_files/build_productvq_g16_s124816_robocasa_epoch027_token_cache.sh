#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT="${CACHE_OUTPUT:-playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/stage2_token_cache_epoch027.pt}"

"${PYTHON_BIN}" starVLA/training/build_var_stage2_token_cache.py \
  --config_yaml examples/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun.yaml \
  --stage1_artifact playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/epoch_027.ckpt \
  --output "${OUTPUT}" \
  --mode train \
  --device "${CACHE_DEVICE:-cuda}" \
  --batch_size "${CACHE_BATCH_SIZE:-256}" \
  --num_workers "${CACHE_NUM_WORKERS:-8}" \
  --max_batches "${CACHE_MAX_BATCHES:-0}"
