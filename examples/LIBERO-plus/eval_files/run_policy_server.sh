#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
ABot_python="${ABot_python:-python}"
your_ckpt="${your_ckpt:-/path/to/checkpoint.pt}"
base_port="${base_port:-9883}"
gpu_id="${gpu_id:-0}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${ABot_python}" deployment/model_server/server_policy.py
  --ckpt_path "${your_ckpt}"
  --port "${base_port}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}"
