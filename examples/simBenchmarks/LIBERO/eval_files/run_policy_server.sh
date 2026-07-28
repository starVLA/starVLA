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

if [[ -n "${USE_CANONICAL_FORWARD:-}" ]]; then
  if [[ "${USE_CANONICAL_FORWARD}" != "true" && "${USE_CANONICAL_FORWARD}" != "false" ]]; then
    echo "USE_CANONICAL_FORWARD must be 'true' or 'false' when set; got '${USE_CANONICAL_FORWARD}'." >&2
    exit 2
  fi
  OVERRIDE="framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}"
  echo "Applying config override: ${OVERRIDE}"
  CMD+=(--config_override "${OVERRIDE}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
