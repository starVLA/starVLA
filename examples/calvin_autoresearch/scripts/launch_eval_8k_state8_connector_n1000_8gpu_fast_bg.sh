#!/usr/bin/env bash
set -euo pipefail

# Faster background launcher for WMH state8+connector steps_8000 n1000 8-GPU eval.
# It uses 2 workers per H200 by default: 16 CALVIN envs + 16 policy servers.
# DEBUG is explicitly disabled to avoid gif/video overhead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

export DEBUG="${DEBUG:-0}"
export CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-1}"
export TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-1000}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
export BASE_PORT="${BASE_PORT:-6800}"
export RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-28800}"
export SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
export RUN_TAG="${RUN_TAG:-8k_state8_connector_steps8000_fast_w${WORKERS_PER_GPU}x8}"

LOG_DIR="${LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/logs}"
mkdir -p "${LOG_DIR}"

TS="$(date +%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/eval_8k_state8_connector_n1000_8gpu_fast_w${WORKERS_PER_GPU}_${TS}.log}"

nohup setsid bash examples/calvin_autoresearch/scripts/run_eval_8k_state8_connector_n1000_8gpu.sh \
  </dev/null > "${LOG_FILE}" 2>&1 &
pid="$!"

echo "PID=${pid}"
echo "LOG=${LOG_FILE}"
echo "PARAMS: DEBUG=${DEBUG} TOTAL_SEQUENCES=${TOTAL_SEQUENCES} GPU_IDS=${GPU_IDS} WORKERS_PER_GPU=${WORKERS_PER_GPU} BASE_PORT=${BASE_PORT}"
echo "To monitor:"
echo "  tail -f ${LOG_FILE}"
