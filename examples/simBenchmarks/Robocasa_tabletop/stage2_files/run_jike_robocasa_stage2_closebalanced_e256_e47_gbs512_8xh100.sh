#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/root/feihong/starVLA"
cd "${REPO_DIR}"

export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29564}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-/root/feihong/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/root/feihong/.cache/torch}"
export WANDB_DIR="${WANDB_DIR:-/root/feihong/starVLA/wandb}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}"

bash "examples/simBenchmarks/Robocasa_tabletop/stage2_files/run_qwen_var_productvq_g16_s124816_robocasa_closebalanced_e256_bestworst_e47_100k_lr1e4_warmup5000_gbs512.sh"
