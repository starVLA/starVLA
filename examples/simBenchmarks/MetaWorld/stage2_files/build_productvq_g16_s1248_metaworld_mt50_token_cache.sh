#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE1_DIR="playground/Checkpoints/var_stage1_metaworld_mt50_e32_aeinit_productvq_g16_s1_2_4_8"
STAGE1_ARTIFACT="${STAGE1_ARTIFACT:-${STAGE1_DIR}/best_recon.ckpt}"
OUTPUT="${CACHE_OUTPUT:-${STAGE1_DIR}/stage2_token_cache_full.pt}"

if [[ ! -f "${STAGE1_ARTIFACT}" ]]; then
  echo "Missing ${STAGE1_ARTIFACT}; train Stage 1 product VQ first." >&2
  exit 1
fi

"${PYTHON_BIN}" starVLA/training/build_var_stage2_token_cache.py \
  --config_yaml examples/simBenchmarks/MetaWorld/train_files/train_var_stage1_metaworld_mt50_e32_aeinit_productvq_g16_s1_2_4_8.yaml \
  --stage1_artifact "${STAGE1_ARTIFACT}" \
  --output "${OUTPUT}" \
  --mode train \
  --device "${CACHE_DEVICE:-cuda}" \
  --batch_size "${CACHE_BATCH_SIZE:-256}" \
  --num_workers "${CACHE_NUM_WORKERS:-8}" \
  --max_batches "${CACHE_MAX_BATCHES:-0}"
