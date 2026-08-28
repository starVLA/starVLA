#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CKPT="${1:-playground/Checkpoints/qwen_var_productvq_g16_s124816_robocasa_epoch027_100k_fullcache/checkpoints/steps_82000_pytorch_model.pt}"
STARVLA_DIR="${STARVLA_DIR:-${DEFAULT_STARVLA_DIR}}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-python}"

TASKS_PRESET="${TASKS_PRESET:-gr1_5}"
TASKS_FILE="${TASKS_FILE:-}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-10}"
CHUNK_EPISODES="${CHUNK_EPISODES:-1}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
MAX_RETRIES="${MAX_RETRIES:-100000}"
BASE_PORT="${BASE_PORT:-6700}"
N_ENVS="${N_ENVS:-1}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
N_ACTION_STEPS="${N_ACTION_STEPS:-12}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
SERVER_IDLE_TIMEOUT="${SERVER_IDLE_TIMEOUT:-1800}"
SIM_TIMEOUT="${SIM_TIMEOUT:-3600}"
EVAL_USE_BF16="${EVAL_USE_BF16:-0}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
ACTION_STATS_EVERY="${ACTION_STATS_EVERY:-0}"
NORM_ACTION_STATS_EVERY="${NORM_ACTION_STATS_EVERY:-0}"
SESSION_PREFIX="${SESSION_PREFIX:-robocasa_stage2_eval}"

read -r -a GPUS <<< "${EVAL_GPUS:-0 1}"
WORKER_COUNT="${WORKER_COUNT:-${#GPUS[@]}}"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-robocasa_eval/$(basename "${CKPT}" .pt)_${TASKS_PRESET}_${TRIALS_PER_TASK}eps_chunk${CHUNK_EPISODES}_robust}"
OUTPUT_ROOT="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}"
LOG_ROOT="${OUTPUT_ROOT}/logs"
SUPERVISOR_LOG="${LOG_ROOT}/robocasa_parallel_eval_supervisor.log"
SUMMARY_PATH="${OUTPUT_ROOT}/summary.txt"

mkdir -p "${LOG_ROOT}"
cd "${STARVLA_DIR}"

extra_task_args=(--tasks-preset "${TASKS_PRESET}")
if [[ -n "${TASKS_FILE}" ]]; then
  extra_task_args=(--tasks-file "${TASKS_FILE}")
fi

common_args=(
  "${CKPT}"
  --output-root "${OUTPUT_ROOT}"
  --repo-root "${STARVLA_DIR}"
  --starvla-python "${STARVLA_PYTHON}"
  --robocasa-python "${ROBOCASA_PYTHON}"
  "${extra_task_args[@]}"
  --trials-per-task "${TRIALS_PER_TASK}"
  --chunk-episodes "${CHUNK_EPISODES}"
  --worker-count "${WORKER_COUNT}"
  --base-port "${BASE_PORT}"
  --max-retries "${MAX_RETRIES}"
  --server-ready-timeout "${SERVER_READY_TIMEOUT}"
  --server-idle-timeout "${SERVER_IDLE_TIMEOUT}"
  --sim-timeout "${SIM_TIMEOUT}"
  --n-envs "${N_ENVS}"
  --max-episode-steps "${MAX_EPISODE_STEPS}"
  --n-action-steps "${N_ACTION_STEPS}"
  --action-stats-every "${ACTION_STATS_EVERY}"
  --norm-action-stats-every "${NORM_ACTION_STATS_EVERY}"
)

if [[ "${EVAL_USE_BF16}" == "1" ]]; then
  common_args+=(--use-bf16)
fi
if [[ "${SAVE_VIDEOS}" != "1" ]]; then
  common_args+=(--no-video)
fi

launch_worker() {
  local worker="$1"
  local gpu="$2"
  local session="${SESSION_PREFIX}_w${worker}"
  local launch_log="${LOG_ROOT}/worker_${worker}.log"
  if tmux has-session -t "${session}" >/dev/null 2>&1; then
    return 0
  fi
  echo "[$(date)] launching worker=${worker}/${WORKER_COUNT} gpu=${gpu} session=${session}" | tee -a "${SUPERVISOR_LOG}"
  tmux new-session -d -s "${session}" \
    "cd ${STARVLA_DIR}; PYTHONPATH=${STARVLA_DIR} ${STARVLA_PYTHON} examples/simBenchmarks/Robocasa_tabletop/eval_files/run_robocasa_stage2_eval_chunked.py ${common_args[*]} --worker-index ${worker} --gpu ${gpu} >> ${launch_log} 2>&1"
}

summarize() {
  "${STARVLA_PYTHON}" examples/simBenchmarks/Robocasa_tabletop/eval_files/summarize_robocasa_success.py \
    "${OUTPUT_ROOT}" \
    "${extra_task_args[@]}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --chunk-episodes "${CHUNK_EPISODES}" \
    --expected-episodes-per-chunk "${CHUNK_EPISODES}" > "${SUMMARY_PATH}" || true
}

eval_complete() {
  "${STARVLA_PYTHON}" examples/simBenchmarks/Robocasa_tabletop/eval_files/summarize_robocasa_success.py \
    "${OUTPUT_ROOT}" \
    "${extra_task_args[@]}" \
    --trials-per-task "${TRIALS_PER_TASK}" \
    --chunk-episodes "${CHUNK_EPISODES}" \
    --expected-episodes-per-chunk "${CHUNK_EPISODES}" \
    --require-complete > "${SUMMARY_PATH}"
}

echo "[$(date)] RoboCasa stage2 parallel eval supervisor started" | tee -a "${SUPERVISOR_LOG}"
echo "[$(date)] ckpt=${CKPT}" | tee -a "${SUPERVISOR_LOG}"
echo "[$(date)] output=${OUTPUT_ROOT}" | tee -a "${SUPERVISOR_LOG}"
echo "[$(date)] tasks=${TASKS_PRESET} trials=${TRIALS_PER_TASK} chunk=${CHUNK_EPISODES} bf16=${EVAL_USE_BF16}" | tee -a "${SUPERVISOR_LOG}"

while true; do
  summarize
  if eval_complete; then
    echo "[$(date)] completed all expected chunks; summary=${SUMMARY_PATH}" | tee -a "${SUPERVISOR_LOG}"
    exit 0
  fi
  for worker in $(seq 0 $((WORKER_COUNT - 1))); do
    gpu="${GPUS[$((worker % ${#GPUS[@]}))]}"
    launch_worker "${worker}" "${gpu}"
  done
  sleep "${CHECK_INTERVAL_SECONDS}"
done
