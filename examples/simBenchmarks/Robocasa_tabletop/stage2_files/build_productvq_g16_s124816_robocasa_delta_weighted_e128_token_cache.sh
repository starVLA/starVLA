#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:${PATH}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1

CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_delta_productvq_g16_s1_2_4_8_16_weighted.yaml}"
STAGE1_ARTIFACT="${STAGE1_ARTIFACT:-/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_delta_productvq_g16_s124816_e128_weighted/best_recon.ckpt}"
OUTPUT="${OUTPUT:-/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_delta_productvq_g16_s124816_e128_weighted/stage2_token_cache_best_recon_delta.pt}"

if [[ ! -f "${STAGE1_ARTIFACT}" ]]; then
  echo "Missing delta productVQ stage1 artifact: ${STAGE1_ARTIFACT}" >&2
  exit 1
fi

"${PYTHON_BIN}" starVLA/training/build_var_stage2_token_cache.py   --config_yaml "${CONFIG_YAML}"   --stage1_artifact "${STAGE1_ARTIFACT}"   --output "${OUTPUT}"   --mode train   --device "${DEVICE:-cuda}"   --batch_size "${BATCH_SIZE:-512}"   --num_workers "${NUM_WORKERS:-0}"   --max_batches "${MAX_BATCHES:-0}"
