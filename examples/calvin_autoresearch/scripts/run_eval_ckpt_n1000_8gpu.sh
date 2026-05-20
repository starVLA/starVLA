#!/usr/bin/env bash
set -euo pipefail

# Formal CALVIN ABC->D 1000-sequence evaluation for any local checkpoint.
# Usage:
#   CKPT=/path/to/steps_80000_pytorch_model.pt CALVIN_SEND_STATE=1 \
#     bash examples/calvin_autoresearch/scripts/run_eval_ckpt_n1000_8gpu.sh
#
# Set CALVIN_SEND_STATE=1 only for state-aware checkpoints with model_state_dim=8.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

: "${CKPT:?Set CKPT=/path/to/checkpoints/steps_80000_pytorch_model.pt}"

MEMBER="${MEMBER:-WMH}"
TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-1000}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-6700}"
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-28800}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-0}"
RUN_TAG="${RUN_TAG:-$(basename "$(dirname "$(dirname "${CKPT}")")")}"
TS="${TS:-$(date +%m%d_%H%M%S)}"

SAFE_TAG="$(printf '%s' "${RUN_TAG}" | tr -cs 'A-Za-z0-9_.-' '_')"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/${MEMBER}/reports/eval_${SAFE_TAG}_d_n${TOTAL_SEQUENCES}_${TS}}"

echo "[eval-ckpt-n1000] ckpt=${CKPT}"
echo "[eval-ckpt-n1000] send_state=${CALVIN_SEND_STATE}"
echo "[eval-ckpt-n1000] gpu_ids=${GPU_IDS}"
echo "[eval-ckpt-n1000] workers_per_gpu=${WORKERS_PER_GPU}"
echo "[eval-ckpt-n1000] eval_log_dir=${EVAL_LOG_DIR}"

TOTAL_SEQUENCES="${TOTAL_SEQUENCES}" \
GPU_IDS="${GPU_IDS}" \
WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
BASE_PORT="${BASE_PORT}" \
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT}" \
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" \
CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
CKPT="${CKPT}" \
EVAL_LOG_DIR="${EVAL_LOG_DIR}" \
bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
