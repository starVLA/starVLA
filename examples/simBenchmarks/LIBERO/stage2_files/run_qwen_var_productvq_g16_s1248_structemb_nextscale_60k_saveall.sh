#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb/stage2_token_cache_full.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build it first with examples/simBenchmarks/LIBERO/stage2_files/build_productvq_g16_s1248_structemb_full_token_cache.sh" >&2
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

RUN_DIR="/root/nas/feihong/starVLA/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_60k_fullcache_saveall"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES:-8}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29542}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g16_s1248_structemb_nextscale_60k_fullcache_saveall.yaml \
  2>&1 | tee -a "${LOG_FILE}"
