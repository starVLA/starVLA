#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

: "${CKPT:?Set CKPT to a checkpoint produced by this WMH CALVIN baseline, for example results/Checkpoints/baseline_qwen3vl_action_gr00t_calvin_abc_smoke/checkpoints/steps_1_pytorch_model.pt}"

case "${CKPT}" in
  *Qwen3-VL-OFT-LIBERO*|*LIBERO*|*Robotwin*|*robotwin*|*Robocasa*|*robocasa*|*Behavior*|*BEHAVIOR*|*SimplerEnv*|*qwenpi_calvin_task_D_D*)
    echo "Refusing action-trained upstream checkpoint: ${CKPT}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 3
fi

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
PYTHON_BIN="${STARVLA_PYTHON:-python}"

export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

exec "${PYTHON_BIN}" deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT}" \
  --port "${PORT}" \
  --use_bf16
