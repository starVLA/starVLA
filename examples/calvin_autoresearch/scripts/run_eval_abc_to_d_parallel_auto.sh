#!/usr/bin/env bash
set -euo pipefail

# Parallel CALVIN ABC->D evaluation.
# Starts one or more policy servers per visible GPU, shards eval_sequences.json
# by start/stride, then aggregates worker results.

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
TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-${NUM_SEQUENCES:-1000}}"
BASE_PORT="${BASE_PORT:-5694}"
HOST="${HOST:-127.0.0.1}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_abc_to_d_parallel_n${TOTAL_SEQUENCES}_${TS}}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-7200}"
DEBUG="${DEBUG:-0}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-0}"
CALVIN_STATE_MODE="${CALVIN_STATE_MODE:-normal}"
CALVIN_STATE_SHUFFLE_BUFFER="${CALVIN_STATE_SHUFFLE_BUFFER:-32}"
CALVIN_DEBUG_GIF_ROOT="${CALVIN_DEBUG_GIF_ROOT:-${EVAL_LOG_DIR}/debug_gifs}"
CALVIN_DEBUG_GIF_COUNTER_DIR="${CALVIN_DEBUG_GIF_COUNTER_DIR:-${EVAL_LOG_DIR}/debug_gif_counts}"
DRY_RUN="${DRY_RUN:-0}"
POLICY_SERVER_SCRIPT="${POLICY_SERVER_SCRIPT:-examples/calvin_autoresearch/scripts/run_policy_server.sh}"

# Auto mode is conservative: estimate one policy server as 36 GiB and keep 24 GiB free.
# On H200 this usually selects 2 workers/GPU. Lower MAX_WORKERS_PER_GPU to 1 if
# the node is shared or if you see OOM during startup.
GPU_IDS="${GPU_IDS:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-auto}"
PER_SERVER_MEM_MB="${PER_SERVER_MEM_MB:-36000}"
GPU_MEM_RESERVE_MB="${GPU_MEM_RESERVE_MB:-24000}"
MAX_WORKERS_PER_GPU="${MAX_WORKERS_PER_GPU:-2}"

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

if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

declare -a WORKER_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  gpu="$(echo "${gpu}" | xargs)"
  [[ -n "${gpu}" ]] || continue
  if [[ "${WORKERS_PER_GPU}" == "auto" ]]; then
    mem_line="$(nvidia-smi --id="${gpu}" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | head -1)"
    total_mb="$(echo "${mem_line}" | cut -d',' -f1 | xargs)"
    used_mb="$(echo "${mem_line}" | cut -d',' -f2 | xargs)"
    free_for_workers=$(( total_mb - used_mb - GPU_MEM_RESERVE_MB ))
    count=$(( free_for_workers / PER_SERVER_MEM_MB ))
    if (( count < 1 )); then
      count=1
    fi
    if (( count > MAX_WORKERS_PER_GPU )); then
      count="${MAX_WORKERS_PER_GPU}"
    fi
  else
    count="${WORKERS_PER_GPU}"
  fi
  for ((i = 0; i < count; i++)); do
    WORKER_GPUS+=("${gpu}")
  done
done

if (( ${#WORKER_GPUS[@]} == 0 )); then
  echo "No GPU workers selected. Set GPU_IDS=0,1,2,3 or check nvidia-smi." >&2
  exit 4
fi
if (( ${#WORKER_GPUS[@]} > TOTAL_SEQUENCES )); then
  WORKER_GPUS=("${WORKER_GPUS[@]:0:${TOTAL_SEQUENCES}}")
fi

NUM_WORKERS="${#WORKER_GPUS[@]}"
mkdir -p "${EVAL_LOG_DIR}"

echo "[parallel-eval] total_sequences=${TOTAL_SEQUENCES}"
echo "[parallel-eval] gpu_ids=${GPU_IDS}"
echo "[parallel-eval] workers=${NUM_WORKERS}"
echo "[parallel-eval] debug=${DEBUG}"
echo "[parallel-eval] send_state=${CALVIN_SEND_STATE}"
echo "[parallel-eval] state_mode=${CALVIN_STATE_MODE}"
echo "[parallel-eval] eval_log_dir=${EVAL_LOG_DIR}"
if [[ "${DEBUG}" == "1" || "${DEBUG}" == "true" || "${DEBUG}" == "True" ]]; then
  echo "[parallel-eval] debug_gif_root=${CALVIN_DEBUG_GIF_ROOT}"
fi
for worker_id in "${!WORKER_GPUS[@]}"; do
  count=$(( (TOTAL_SEQUENCES - worker_id + NUM_WORKERS - 1) / NUM_WORKERS ))
  port=$(( BASE_PORT + worker_id ))
  printf '[parallel-eval] worker=%02d gpu=%s port=%s start=%s stride=%s count=%s\n' \
    "${worker_id}" "${WORKER_GPUS[$worker_id]}" "${port}" "${worker_id}" "${NUM_WORKERS}" "${count}"
done | tee "${EVAL_LOG_DIR}/workers.tsv"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

declare -a SERVER_PIDS=()
declare -a EVAL_PIDS=()

cleanup() {
  for pid in "${EVAL_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup EXIT

for worker_id in "${!WORKER_GPUS[@]}"; do
  worker_dir="${EVAL_LOG_DIR}/worker_$(printf '%02d' "${worker_id}")"
  mkdir -p "${worker_dir}"
  port=$(( BASE_PORT + worker_id ))
  echo "[parallel-eval] starting server worker=${worker_id} gpu=${WORKER_GPUS[$worker_id]} port=${port}"
  env -u DEBUG \
    SERVER_DEBUG="${SERVER_DEBUG:-0}" \
    DEBUGPY="${DEBUGPY:-0}" \
    STARVLA_STATE_SANITY_MODE="${CALVIN_STATE_MODE}" \
    GPU_ID="${WORKER_GPUS[$worker_id]}" PORT="${port}" CKPT="${CKPT}" \
    bash "${POLICY_SERVER_SCRIPT}" \
    > "${worker_dir}/server.log" 2>&1 &
  SERVER_PIDS+=("$!")
  echo "${SERVER_PIDS[-1]}" > "${worker_dir}/server.pid"
done

start_ts="$(date +%s)"
for worker_id in "${!WORKER_GPUS[@]}"; do
  worker_dir="${EVAL_LOG_DIR}/worker_$(printf '%02d' "${worker_id}")"
  server_pid="${SERVER_PIDS[$worker_id]}"
  while true; do
    if grep -q "server running" "${worker_dir}/server.log" 2>/dev/null; then
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      echo "Policy server worker ${worker_id} exited before ready. Log:" >&2
      tail -100 "${worker_dir}/server.log" >&2 || true
      exit 5
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts > SERVER_READY_TIMEOUT )); then
      echo "Timed out waiting for policy server worker ${worker_id}. Log:" >&2
      tail -100 "${worker_dir}/server.log" >&2 || true
      exit 6
    fi
    sleep 2
  done
done

for worker_id in "${!WORKER_GPUS[@]}"; do
  count=$(( (TOTAL_SEQUENCES - worker_id + NUM_WORKERS - 1) / NUM_WORKERS ))
  if (( count <= 0 )); then
    continue
  fi
  worker_dir="${EVAL_LOG_DIR}/worker_$(printf '%02d' "${worker_id}")"
  port=$(( BASE_PORT + worker_id ))
  echo "[parallel-eval] running worker=${worker_id} count=${count}"
  CALVIN_PYTHON="${CALVIN_PYTHON}" \
  CALVIN_D_DATASET="${CALVIN_D_DATASET}" \
  CALVIN_CONFIG_PATH="${CALVIN_CONFIG_PATH}" \
  CKPT="${CKPT}" \
  HOST="${HOST}" \
  PORT="${port}" \
  UNNORM_KEY="${UNNORM_KEY}" \
  NUM_SEQUENCES="${count}" \
  SEQUENCE_START="${worker_id}" \
  SEQUENCE_STRIDE="${NUM_WORKERS}" \
  DEBUG="${DEBUG}" \
  CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
  CALVIN_STATE_MODE="${CALVIN_STATE_MODE}" \
  CALVIN_STATE_SHUFFLE_BUFFER="${CALVIN_STATE_SHUFFLE_BUFFER}" \
  CALVIN_DEBUG_GIF_ROOT="${CALVIN_DEBUG_GIF_ROOT}" \
  CALVIN_DEBUG_GIF_COUNTER_DIR="${CALVIN_DEBUG_GIF_COUNTER_DIR}" \
  EVAL_LOG_DIR="${worker_dir}" \
    bash examples/calvin_autoresearch/scripts/run_eval_d_formal.sh \
    > "${worker_dir}/eval.log" 2>&1 &
  EVAL_PIDS+=("$!")
done

expected_results="${#EVAL_PIDS[@]}"
result_start_ts="$(date +%s)"
while true; do
  result_count="$(find "${EVAL_LOG_DIR}" -path '*/worker_*/results.json' -type f 2>/dev/null | wc -l)"
  if (( result_count >= expected_results )); then
    echo "[parallel-eval] collected ${result_count}/${expected_results} worker result files"
    break
  fi

  for worker_id in "${!EVAL_PIDS[@]}"; do
    worker_dir="${EVAL_LOG_DIR}/worker_$(printf '%02d' "${worker_id}")"
    if [[ ! -f "${worker_dir}/results.json" ]] && ! kill -0 "${EVAL_PIDS[$worker_id]}" 2>/dev/null; then
      echo "Eval worker ${worker_id} exited without results. Log:" >&2
      tail -100 "${worker_dir}/eval.log" >&2 || true
      exit 7
    fi
  done

  now_ts="$(date +%s)"
  if (( now_ts - result_start_ts > RESULT_WAIT_TIMEOUT )); then
    echo "Timed out waiting for worker results (${result_count}/${expected_results})." >&2
    echo "Check ${EVAL_LOG_DIR}/worker_*/eval.log" >&2
    exit 8
  fi
  sleep 5
done

PYTHONDONTWRITEBYTECODE=1 python examples/calvin_autoresearch/scripts/aggregate_parallel_eval_dir.py "${EVAL_LOG_DIR}"

cleanup
trap - EXIT
echo "[parallel-eval] done: ${EVAL_LOG_DIR}/results.json"
if [[ -f "${EVAL_LOG_DIR}/metrics.json" ]]; then
  echo "[parallel-eval] metrics: ${EVAL_LOG_DIR}/metrics.json"
fi
