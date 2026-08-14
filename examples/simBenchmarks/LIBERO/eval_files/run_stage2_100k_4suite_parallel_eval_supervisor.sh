#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_DIR="${STARVLA_DIR:-${DEFAULT_STARVLA_DIR}}"
CKPT="${1:-${STARVLA_DIR}/playground/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints/steps_100000_pytorch_model.pt}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_stage2_100k_40task_4suite}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
MAX_RETRIES="${MAX_RETRIES:-100000}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
EVAL_SEED="${EVAL_SEED:-7}"
EVAL_USE_BF16="${EVAL_USE_BF16:-1}"

MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
LOG_ROOT="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs"
SUMMARY_PATH="${LOG_ROOT}/libero_40task_summary.txt"
SUPERVISOR_LOG="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/stage2_100k_4suite_parallel_supervisor.log"

mkdir -p "${LOG_ROOT}"

SUITES=(libero_spatial libero_object libero_goal libero_10)
read -r -a GPUS <<< "${EVAL_GPUS:-2 3 4 5}"
read -r -a PORTS <<< "${EVAL_PORTS:-18830 18831 18832 18833}"
SESSION_SUFFIX="${SESSION_SUFFIX:-}"
SESSIONS=(
  "stage2_100k_spatial${SESSION_SUFFIX}"
  "stage2_100k_object${SESSION_SUFFIX}"
  "stage2_100k_goal${SESSION_SUFFIX}"
  "stage2_100k_10${SESSION_SUFFIX}"
)

if [[ "${#GPUS[@]}" -ne "${#SUITES[@]}" || "${#PORTS[@]}" -ne "${#SUITES[@]}" ]]; then
  echo "EVAL_GPUS and EVAL_PORTS must each provide ${#SUITES[@]} values" >&2
  exit 2
fi

# Full-suite ownership: one GPU/session per LIBERO suite.
# object/goal/spatial use single-trial chunks because they have shown
# intermittent environment/client aborts; completed chunks are skipped on retry.
suite_chunk_trials() {
  local suite="$1"
  if [[ "${suite}" == "libero_object" || "${suite}" == "libero_goal" ]]; then
    echo "${OBJECT_GOAL_CHUNK_TRIALS:-1}"
  elif [[ "${suite}" == "libero_spatial" ]]; then
    echo "${SPATIAL_CHUNK_TRIALS:-1}"
  elif [[ "${suite}" == "libero_10" ]]; then
    echo "${LIBERO_10_CHUNK_TRIALS:-5}"
  else
    echo "${SPATIAL_10_CHUNK_TRIALS:-5}"
  fi
}

chunk_completed() {
  local log_path="$1"
  local expected_episodes="$2"
  [[ -f "${log_path}" ]] || return 1
  grep -q "EVAL_CHUNK_OK" "${log_path}" || return 1
  grep -q "Total success rate:" "${log_path}" || return 1
  grep -q "Total episodes: ${expected_episodes}" "${log_path}" || return 1
}

trial_completed() {
  local suite="$1"
  local ckpt_base="$2"
  local task_id="$3"
  local trial_idx="$4"
  local log_path base start count end

  shopt -s nullglob
  for log_path in "${LOG_ROOT}/${suite}/${ckpt_base}"_stage2_chunked_t"${task_id}"_r*_n*.log; do
    base="$(basename "${log_path}")"
    if [[ "${base}" =~ _t${task_id}_r([0-9]+)_n([0-9]+)\.log$ ]]; then
      start="${BASH_REMATCH[1]}"
      count="${BASH_REMATCH[2]}"
      end=$((start + count))
      if [[ "${trial_idx}" -ge "${start}" && "${trial_idx}" -lt "${end}" ]]; then
        chunk_completed "${log_path}" "${count}" && return 0
      fi
    fi
  done
  return 1
}

suite_completed() {
  local suite="$1"
  local suite_chunk="$2"
  local ckpt_base
  ckpt_base="$(basename "${CKPT}" .pt)"
  local task_id trial_idx

  for task_id in $(seq 0 9); do
    for trial_idx in $(seq 0 $((TRIALS_PER_TASK - 1))); do
      trial_completed "${suite}" "${ckpt_base}" "${task_id}" "${trial_idx}" || return 1
    done
  done
  return 0
}

launch_suite() {
  local suite="$1"
  local gpu="$2"
  local port="$3"
  local session="$4"
  local suite_chunk="$5"
  local launch_log="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/launch_${suite}.log"

  if tmux has-session -t "${session}" >/dev/null 2>&1; then
    return 0
  fi

  echo "[$(date)] launching ${suite} on gpu=${gpu} port=${port} session=${session} chunk=${suite_chunk}" | tee -a "${SUPERVISOR_LOG}"
  tmux new-session -d -s "${session}" \
    "cd ${STARVLA_DIR}; USE_BF16=${EVAL_USE_BF16} SEED=${EVAL_SEED} TASK_SUITES_OVERRIDE=${suite} EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT} TRIALS_PER_TASK=${TRIALS_PER_TASK} CHUNK_TRIALS=${suite_chunk} MAX_RETRIES=${MAX_RETRIES} CHUNK_TIMEOUT_SECONDS=${CHUNK_TIMEOUT_SECONDS} SAVE_VIDEOS=0 IMAGE_VIEWS=primary+wrist POLICY_IMAGE_SIZE=224 CONSTRAIN_TO_ACTION_TOKENS=0 CLIP_NORMALIZED_ACTIONS=0 VALIDATE_INPUTS=1 STRICT_TRIAL_COUNT=1 bash examples/simBenchmarks/LIBERO/eval_files/run_stage2_eval_chunked.sh ${CKPT} ${gpu} ${port} >> ${launch_log} 2>&1"
}

cd "${STARVLA_DIR}"
echo "[$(date)] stage2 100k 4-suite parallel eval supervisor started" | tee -a "${SUPERVISOR_LOG}"
echo "[$(date)] ckpt=${CKPT}" | tee -a "${SUPERVISOR_LOG}"
echo "[$(date)] output=${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}" | tee -a "${SUPERVISOR_LOG}"
echo "[$(date)] eval_seed=${EVAL_SEED}" | tee -a "${SUPERVISOR_LOG}"
echo "[$(date)] eval_use_bf16=${EVAL_USE_BF16}" | tee -a "${SUPERVISOR_LOG}"

while true; do
  completed=0
  for idx in "${!SUITES[@]}"; do
    suite="${SUITES[$idx]}"
    suite_chunk="$(suite_chunk_trials "${suite}")"
    if suite_completed "${suite}" "${suite_chunk}"; then
      completed=$((completed + 1))
      continue
    fi
    launch_suite "${suite}" "${GPUS[$idx]}" "${PORTS[$idx]}" "${SESSIONS[$idx]}" "${suite_chunk}"
  done

  python examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${LOG_ROOT}" --chunked --require-ok-marker > "${LOG_ROOT}/libero_40task_progress.txt" || true

  if [[ "${completed}" -eq "${#SUITES[@]}" ]]; then
    python examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${LOG_ROOT}" --chunked --require-ok-marker | tee "${SUMMARY_PATH}"
    echo "[$(date)] completed all suites; summary=${SUMMARY_PATH}" | tee -a "${SUPERVISOR_LOG}"
    exit 0
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
