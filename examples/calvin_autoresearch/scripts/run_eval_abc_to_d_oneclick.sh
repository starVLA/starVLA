#!/usr/bin/env bash
set -euo pipefail

# One-command CALVIN ABC->D closed-loop evaluation.
# Starts a policy server from a WMH-trained checkpoint, waits until it is ready,
# runs the strict D evaluator, and stops the server on exit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

RUN_ID="${RUN_ID:-abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437}"
CKPT="${CKPT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/${RUN_ID}/checkpoints/steps_60000_pytorch_model.pt}"
CALVIN_D_DATASET="${CALVIN_D_DATASET:-/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d}"
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/calvin_models/conf}"
CALVIN_PYTHON="${CALVIN_PYTHON:-/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/calvin_venv/bin/python}"
NUM_SEQUENCES="${NUM_SEQUENCES:-10}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
HOST="${HOST:-127.0.0.1}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_formal_n${NUM_SEQUENCES}_${TS}}"
START_SERVER="${START_SERVER:-1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
DEBUG="${DEBUG:-0}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-0}"
CALVIN_STATE_MODE="${CALVIN_STATE_MODE:-normal}"
CALVIN_STATE_SHUFFLE_BUFFER="${CALVIN_STATE_SHUFFLE_BUFFER:-32}"
CALVIN_DEBUG_GIF_ROOT="${CALVIN_DEBUG_GIF_ROOT:-${EVAL_LOG_DIR}/debug_gifs}"
CALVIN_DEBUG_GIF_COUNTER_DIR="${CALVIN_DEBUG_GIF_COUNTER_DIR:-${EVAL_LOG_DIR}/debug_gif_counts}"

mkdir -p "${EVAL_LOG_DIR}"

case "${CKPT}" in
  *Qwen3-VL-OFT-LIBERO*|*LIBERO*|*Robotwin*|*robotwin*|*Robocasa*|*robocasa*|*Behavior*|*BEHAVIOR*|*SimplerEnv*|*qwenpi_calvin_task_D_D*)
    echo "Refusing action-trained upstream checkpoint: ${CKPT}" >&2
    exit 2
    ;;
esac

for required in \
  "${CKPT}" \
  "${CALVIN_D_DATASET}/validation/.hydra/merged_config.yaml" \
  "${CALVIN_CONFIG_PATH}/annotations/new_playtable_validation.yaml" \
  "${CALVIN_CONFIG_PATH}/callbacks/rollout/tasks/new_playtable_tasks.yaml" \
  "${CALVIN_PYTHON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required eval asset: ${required}" >&2
    exit 3
  fi
done

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "${START_SERVER}" == "1" ]]; then
  echo "[oneclick-eval] starting server on GPU ${GPU_ID}, port ${PORT}"
  GPU_ID="${GPU_ID}" PORT="${PORT}" CKPT="${CKPT}" STARVLA_STATE_SANITY_MODE="${CALVIN_STATE_MODE}" \
    bash examples/calvin_autoresearch/scripts/run_policy_server.sh \
    > "${EVAL_LOG_DIR}/server.log" 2>&1 &
  SERVER_PID="$!"
  echo "${SERVER_PID}" > "${EVAL_LOG_DIR}/server.pid"

  start_ts="$(date +%s)"
  while true; do
    if grep -q "server running" "${EVAL_LOG_DIR}/server.log" 2>/dev/null; then
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Policy server exited before becoming ready. Log:" >&2
      tail -100 "${EVAL_LOG_DIR}/server.log" >&2 || true
      exit 4
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > SERVER_READY_TIMEOUT )); then
      echo "Timed out waiting for policy server. Log:" >&2
      tail -100 "${EVAL_LOG_DIR}/server.log" >&2 || true
      exit 5
    fi
    sleep 2
  done
else
  echo "[oneclick-eval] START_SERVER=0; using existing server at ${HOST}:${PORT}"
fi

echo "[oneclick-eval] running NUM_SEQUENCES=${NUM_SEQUENCES}"
CALVIN_PYTHON="${CALVIN_PYTHON}" \
CALVIN_D_DATASET="${CALVIN_D_DATASET}" \
CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}" \
CKPT="${CKPT}" \
HOST="${HOST}" \
PORT="${PORT}" \
UNNORM_KEY="${UNNORM_KEY}" \
NUM_SEQUENCES="${NUM_SEQUENCES}" \
DEBUG="${DEBUG}" \
CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
CALVIN_STATE_MODE="${CALVIN_STATE_MODE}" \
CALVIN_STATE_SHUFFLE_BUFFER="${CALVIN_STATE_SHUFFLE_BUFFER}" \
CALVIN_DEBUG_GIF_ROOT="${CALVIN_DEBUG_GIF_ROOT}" \
CALVIN_DEBUG_GIF_COUNTER_DIR="${CALVIN_DEBUG_GIF_COUNTER_DIR}" \
EVAL_LOG_DIR="${EVAL_LOG_DIR}" \
bash examples/calvin_autoresearch/scripts/run_eval_d_formal.sh 2>&1 | tee "${EVAL_LOG_DIR}/eval.log"

echo "[oneclick-eval] results: ${EVAL_LOG_DIR}/results.json"
if [[ -f "${EVAL_LOG_DIR}/metrics.json" ]]; then
  echo "[oneclick-eval] metrics: ${EVAL_LOG_DIR}/metrics.json"
fi
