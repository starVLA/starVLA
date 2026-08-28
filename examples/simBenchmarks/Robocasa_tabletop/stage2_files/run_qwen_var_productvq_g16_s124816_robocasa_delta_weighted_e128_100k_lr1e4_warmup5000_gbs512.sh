#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

CONFIG_YAML="examples/simBenchmarks/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_delta_weighted_e128_100k_lr1e4_warmup5000_gbs512.yaml"
STAGE1_ARTIFACT="/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_delta_productvq_g16_s124816_e128_weighted/best_recon.ckpt"
TOKEN_CACHE="/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_delta_productvq_g16_s124816_e128_weighted/stage2_token_cache_best_recon_delta.pt"
if [[ ! -f "${STAGE1_ARTIFACT}" ]]; then
  echo "Missing ${STAGE1_ARTIFACT}; finish delta productVQ Stage 1 first." >&2
  exit 1
fi
if [[ ! -f "${TOKEN_CACHE}" ]]; then
  echo "Missing ${TOKEN_CACHE}; run build_productvq_g16_s124816_robocasa_delta_weighted_e128_token_cache.sh first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
export PATH="$(pwd)/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DISTRIBUTED_TIMEOUT_SECONDS="${TORCH_DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-15}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export DEEPSPEED_REDUCE_BUCKET_SIZE="${DEEPSPEED_REDUCE_BUCKET_SIZE:-100000000}"
export DEEPSPEED_ALLGATHER_BUCKET_SIZE="${DEEPSPEED_ALLGATHER_BUCKET_SIZE:-100000000}"

if [[ "${ALLOW_SHARED_GPU:-0}" != "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  BUSY_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ' | sed '/^$/d' || true)"
  if [[ -n "${BUSY_PIDS}" ]]; then
    echo "CUDA device is busy with existing compute process(es): ${BUSY_PIDS}. Set ALLOW_SHARED_GPU=1 to run anyway." >&2
    exit 1
  fi
fi

REQUIRED_GPUS="${NUM_PROCESSES:-8}"
AVAILABLE_GPUS="$("${PYTHON_BIN}" -c "import torch; print(torch.cuda.device_count())")"
if [[ "${AVAILABLE_GPUS}" -lt "${REQUIRED_GPUS}" ]]; then
  echo "Need ${REQUIRED_GPUS} visible CUDA devices, but only ${AVAILABLE_GPUS} detected. Set NUM_PROCESSES/CUDA_VISIBLE_DEVICES explicitly or use an 8-GPU node." >&2
  exit 1
fi

RUN_DIR="/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_delta_weighted_e128_100k_lr1e4_warmup5000_gbs512_fullcache"
mkdir -p "${RUN_DIR}"
LOG_FILE="${RUN_DIR}/train.log"

"${PYTHON_BIN}" -m accelerate.commands.launch   --num_processes "${NUM_PROCESSES:-8}"   --mixed_precision "${MIXED_PRECISION:-bf16}"   --main_process_port "${MAIN_PROCESS_PORT:-29574}"   starVLA/training/train_starvla.py   --config_yaml "${CONFIG_YAML}"   2>&1 | tee -a "${LOG_FILE}"
