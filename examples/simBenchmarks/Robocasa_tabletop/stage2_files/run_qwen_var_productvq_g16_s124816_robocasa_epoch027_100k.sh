#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_robocasa_gr1_e64_aeinit_productvq_g16_s1_2_4_8_16_batch256_rerun/stage2_token_cache_epoch027.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build the RoboCasa Stage 2 token cache first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="/root/nas/feihong/starVLA/Checkpoints/qwen_var_productvq_g16_s124816_robocasa_epoch027_100k_fullcache"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES:-8}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29553}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_epoch027_100k.yaml \
  2>&1 | tee -a "${LOG_FILE}"
