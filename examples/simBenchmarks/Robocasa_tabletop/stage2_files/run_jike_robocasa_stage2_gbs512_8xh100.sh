#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/feihong/starVLA"
cd "${REPO_DIR}"

export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29564}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="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"
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
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Keep runtime caches and W&B files off the system disk.
export HF_HOME="${HF_HOME:-/root/feihong/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/feihong/.cache/torch}"
export WANDB_DIR="${WANDB_DIR:-/root/feihong/starVLA/wandb}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}"

bash "examples/simBenchmarks/Robocasa_tabletop/stage2_files/run_qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs512.sh"
