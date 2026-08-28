#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:${PATH}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

STAGE1_ARTIFACT="${STAGE1_ARTIFACT:-/root/nas/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_e128_productvq_resume_local_from_epoch016_mirror/best_recon.ckpt}"
OUTPUT="${OUTPUT:-/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_e128_productvq_resume_local_from_epoch016_mirror/stage2_token_cache_best_recon.pt}"

"${PYTHON_BIN}" starVLA/training/build_var_stage2_token_cache.py \
  --config_yaml examples/simBenchmarks/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_e128_aeinit_productvq_g16_s1_2_4_8_16_current_from_latest_epoch032.yaml \
  --stage1_artifact "${STAGE1_ARTIFACT}" \
  --output "${OUTPUT}" \
  --mode train \
  --device "${DEVICE:-cuda}" \
  --batch_size "${BATCH_SIZE:-512}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --max_batches "${MAX_BATCHES:-0}"
