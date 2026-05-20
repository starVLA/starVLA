#!/usr/bin/env bash
set -euo pipefail

# Run the current clean 8k state8+connector baseline eval without requiring the
# caller to paste a long checkpoint path.
#
# Defaults:
#   1. D n10 smoke on one GPU.
#   2. D n100 parallel eval if n10 succeeds.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
REPORT_ROOT="${REPORT_ROOT:-${SHARED_ROOT}/members/${MEMBER}/reports}"

RUN_ID="${RUN_ID:-abc_state8_connector_8h200_bs96_8k_0519_083200}"
CKPT_NAME="${CKPT_NAME:-steps_8000_pytorch_model.pt}"
RUN_DIR="${SHARED_ROOT}/members/${MEMBER}/runs/${RUN_ID}"
CKPT="${CKPT:-${RUN_DIR}/checkpoints/${CKPT_NAME}}"

CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-1}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

RUN_N10="${RUN_N10:-1}"
RUN_N100="${RUN_N100:-1}"
N10_SEQUENCES="${N10_SEQUENCES:-10}"
N100_SEQUENCES="${N100_SEQUENCES:-100}"

N10_GPU_ID="${N10_GPU_ID:-0}"
N10_PORT="${N10_PORT:-5694}"
# Default to a conservative 4-GPU n100 eval. Override N100_GPU_IDS for 8-GPU
# runs after confirming the target node is free.
N100_GPU_IDS="${N100_GPU_IDS:-${GPU_IDS:-0,1,2,3}}"
N100_WORKERS_PER_GPU="${N100_WORKERS_PER_GPU:-1}"
N100_BASE_PORT="${N100_BASE_PORT:-6500}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-14400}"

N10_EVAL_LOG_DIR="${N10_EVAL_LOG_DIR:-${REPORT_ROOT}/eval_8k_state8_connector_d_n10_${TS}}"
N100_EVAL_LOG_DIR="${N100_EVAL_LOG_DIR:-${REPORT_ROOT}/eval_8k_state8_connector_d_n100_${TS}}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  echo "Set RUN_ID/CKPT_NAME/CKPT if you want another checkpoint." >&2
  exit 3
fi

echo "[eval-8k] checkpoint: ${CKPT}"
echo "[eval-8k] send_state: ${CALVIN_SEND_STATE}"
echo "[eval-8k] n10_dir: ${N10_EVAL_LOG_DIR}"
echo "[eval-8k] n100_dir: ${N100_EVAL_LOG_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[eval-8k] DRY_RUN=1; no eval launched."
  echo "[eval-8k] n10 command:"
  printf '  CALVIN_SEND_STATE=%q NUM_SEQUENCES=%q GPU_ID=%q PORT=%q CKPT=%q EVAL_LOG_DIR=%q bash %q\n' \
    "${CALVIN_SEND_STATE}" "${N10_SEQUENCES}" "${N10_GPU_ID}" "${N10_PORT}" "${CKPT}" "${N10_EVAL_LOG_DIR}" \
    "examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh"
  echo "[eval-8k] n100 command:"
  printf '  CALVIN_SEND_STATE=%q TOTAL_SEQUENCES=%q GPU_IDS=%q WORKERS_PER_GPU=%q BASE_PORT=%q CKPT=%q EVAL_LOG_DIR=%q bash %q\n' \
    "${CALVIN_SEND_STATE}" "${N100_SEQUENCES}" "${N100_GPU_IDS}" "${N100_WORKERS_PER_GPU}" "${N100_BASE_PORT}" \
    "${CKPT}" "${N100_EVAL_LOG_DIR}" "examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh"
  exit 0
fi

if [[ "${RUN_N10}" == "1" ]]; then
  echo "[eval-8k] running n10 smoke..."
  CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
  NUM_SEQUENCES="${N10_SEQUENCES}" \
  GPU_ID="${N10_GPU_ID}" \
  PORT="${N10_PORT}" \
  UNNORM_KEY="${UNNORM_KEY}" \
  CKPT="${CKPT}" \
  EVAL_LOG_DIR="${N10_EVAL_LOG_DIR}" \
    bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_oneclick.sh

  if [[ ! -f "${N10_EVAL_LOG_DIR}/metrics.json" ]]; then
    echo "[eval-8k] n10 finished without metrics.json; refusing to start n100." >&2
    echo "[eval-8k] check: ${N10_EVAL_LOG_DIR}" >&2
    exit 4
  fi
  echo "[eval-8k] n10 metrics: ${N10_EVAL_LOG_DIR}/metrics.json"
else
  echo "[eval-8k] RUN_N10=0; skipping n10."
fi

if [[ "${RUN_N100}" == "1" ]]; then
  echo "[eval-8k] running n100 parallel eval..."
  CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
  TOTAL_SEQUENCES="${N100_SEQUENCES}" \
  GPU_IDS="${N100_GPU_IDS}" \
  WORKERS_PER_GPU="${N100_WORKERS_PER_GPU}" \
  BASE_PORT="${N100_BASE_PORT}" \
  SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" \
  RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT}" \
  UNNORM_KEY="${UNNORM_KEY}" \
  CKPT="${CKPT}" \
  EVAL_LOG_DIR="${N100_EVAL_LOG_DIR}" \
    bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh

  if [[ ! -f "${N100_EVAL_LOG_DIR}/metrics.json" ]]; then
    echo "[eval-8k] n100 finished without metrics.json; check worker logs." >&2
    echo "[eval-8k] check: ${N100_EVAL_LOG_DIR}" >&2
    exit 5
  fi
  echo "[eval-8k] n100 metrics: ${N100_EVAL_LOG_DIR}/metrics.json"
else
  echo "[eval-8k] RUN_N100=0; skipping n100."
fi

echo "[eval-8k] done"
