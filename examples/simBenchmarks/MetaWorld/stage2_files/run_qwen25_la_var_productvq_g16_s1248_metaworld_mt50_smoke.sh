#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../../.."

TOKEN_CACHE="playground/Checkpoints/var_stage1_metaworld_mt50_e32_aeinit_productvq_g16_s1_2_4_8/stage2_token_cache_full.pt"
LA_CKPT="/root/nas/feihong/starVLA/Checkpoints/starvla_metaworld_qwenpiv3_la_finetune/checkpoints/steps_60000_pytorch_model.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build it first with examples/simBenchmarks/MetaWorld/stage2_files/build_productvq_g16_s1248_metaworld_mt50_token_cache.sh" >&2
  exit 1
fi
if [[ ! -f "${LA_CKPT}" ]]; then
  echo "Missing ${LA_CKPT}; download MINT-SJTU/starvla_metaworld_qwenpiv3_la_finetune first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="/root/nas/feihong/starVLA/Checkpoints/qwen25_la_var_productvq_g16_s1248_metaworld_mt50_smoke"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES:-1}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29553}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/MetaWorld/stage2_files/train_qwen25_la_var_productvq_g16_s1248_metaworld_mt50_smoke.yaml \
  2>&1 | tee -a "${LOG_FILE}"
