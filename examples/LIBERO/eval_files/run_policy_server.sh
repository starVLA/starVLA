#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
CKPT="${CKPT:-${STARVLA_DIR}/playground/Checkpoints/libero_example/checkpoints/steps_50000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-6694}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
