#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/stage2_token_cache_epoch027.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build the RoboCasa Stage 2 token cache first." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false

RUN_DIR="playground/Checkpoints/qwen_var_productvq_g16_s124816_robocasa_epoch027_100k_fullcache"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
"${ACCELERATE_BIN}" launch \
  --num_processes "${NUM_PROCESSES:-4}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29553}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_epoch027_100k.yaml \
  2>&1 | tee -a "${LOG_FILE}"
