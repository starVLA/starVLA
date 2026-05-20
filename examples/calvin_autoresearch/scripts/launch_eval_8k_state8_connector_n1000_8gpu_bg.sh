#!/usr/bin/env bash
set -euo pipefail

# Background launcher for WMH state8+connector steps_8000 n1000 8-GPU eval.
# This avoids fragile shell line wrapping around `date` and redirection.
#
# Usage:
#   bash examples/calvin_autoresearch/scripts/launch_eval_8k_state8_connector_n1000_8gpu_bg.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

LOG_DIR="${LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/logs}"
mkdir -p "${LOG_DIR}"

TS="$(date +%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/eval_8k_state8_connector_n1000_8gpu_${TS}.log}"

nohup setsid bash examples/calvin_autoresearch/scripts/run_eval_8k_state8_connector_n1000_8gpu.sh \
  </dev/null > "${LOG_FILE}" 2>&1 &
pid="$!"

echo "PID=${pid}"
echo "LOG=${LOG_FILE}"
echo "To monitor:"
echo "  tail -f ${LOG_FILE}"
