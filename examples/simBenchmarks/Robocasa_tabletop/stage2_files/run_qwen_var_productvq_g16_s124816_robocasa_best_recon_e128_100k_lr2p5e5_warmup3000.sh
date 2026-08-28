#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

TOKEN_CACHE="/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_e128_productvq_resume_local_from_epoch016_mirror/stage2_token_cache_best_recon.pt"
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; build the RoboCasa Stage 2 token cache first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REQUIRED_GPUS="${NUM_PROCESSES:-8}"
AVAILABLE_GPUS="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")"
if [[ "${AVAILABLE_GPUS}" -lt "${REQUIRED_GPUS}" ]]; then
  echo "Need ${REQUIRED_GPUS} visible CUDA devices, but only ${AVAILABLE_GPUS} detected. Check the 8x H100 allocation or set NUM_PROCESSES/CUDA_VISIBLE_DEVICES explicitly." >&2
  exit 1
fi


RUN_DIR="/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr2p5e5_warmup3000_fullcache"
mkdir -p "${RUN_DIR}"

LOG_FILE="${RUN_DIR}/train.log"

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES:-8}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29554}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/simBenchmarks/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr2p5e5_warmup3000.yaml \
  2>&1 | tee -a "${LOG_FILE}"
