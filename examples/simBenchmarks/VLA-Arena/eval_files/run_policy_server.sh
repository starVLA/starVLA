#!/usr/bin/env bash
set -euo pipefail
# run_policy_server.sh
#
# Launches the starVLA WebSocket policy server for VLA-Arena evaluation.
# Run this script first, then launch eval_vla_arena.sh in a separate terminal.

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
starVLA_python="${starVLA_python:-python}"
your_ckpt="${your_ckpt:-/path/to/checkpoint.pt}"
gpu_id="${gpu_id:-0}"
port="${port:-10090}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${starVLA_python}" deployment/model_server/server_policy.py
  --ckpt_path "${your_ckpt}"
  --port "${port}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}"
