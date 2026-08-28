#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pt> [gpu_id] [base_port]"
  exit 2
fi

CKPT="$1"
GPU_ID="${2:-1}"
BASE_PORT="${3:-18300}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-5}"
MAX_RETRIES="${MAX_RETRIES:-100000}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-0}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_smoke}"
MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
SUMMARY_PATH="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs/libero_40task_summary.txt"
PROGRESS_PATH="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs/libero_40task_progress.txt"

mkdir -p "$(dirname "${SUMMARY_PATH}")"

for suite_idx in "${!TASK_SUITES[@]}"; do
  suite="${TASK_SUITES[$suite_idx]}"
  port=$((BASE_PORT + suite_idx))
  for task_id in $(seq 0 9); do
    trial_start=0
    while [[ "${trial_start}" -lt "${TRIALS_PER_TASK}" ]]; do
      remaining=$((TRIALS_PER_TASK - trial_start))
      chunk="${CHUNK_TRIALS}"
      if [[ "${remaining}" -lt "${chunk}" ]]; then
        chunk="${remaining}"
      fi
      log_suffix="_chunked_t${task_id}_r${trial_start}_n${chunk}"
      attempt=1
      while true; do
        echo "========== suite=${suite} task=${task_id} trials=${trial_start}..$((trial_start + chunk - 1)) attempt=${attempt} =========="
        if timeout --kill-after=60s "${CHUNK_TIMEOUT_SECONDS}" env \
          TASK_START="${task_id}" TASK_COUNT=1 TRIAL_START="${trial_start}" \
          NUM_TRIALS_PER_TASK="${chunk}" MAX_TASKS=-1 UNNORM_KEY="${UNNORM_KEY}" \
          SAVE_VIDEOS="${SAVE_VIDEOS}" IMAGE_VIEWS="${IMAGE_VIEWS}" \
          POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE}" EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
          LOG_SUFFIX="${log_suffix}" \
          "${SCRIPT_DIR}/run_local_eval_once.sh" "${CKPT}" "${suite}" "${GPU_ID}" "${port}"; then
          break
        fi
        if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
          echo "[chunked_eval] failed after ${MAX_RETRIES} attempts: suite=${suite} task=${task_id} trial_start=${trial_start} chunk=${chunk}" >&2
          exit 1
        fi
        attempt=$((attempt + 1))
        sleep 5
      done
      trial_start=$((trial_start + chunk))
    done
    {
      echo "========== progress after suite=${suite} task=${task_id} =========="
      date
      python examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs" --chunked
      echo
    } | tee "${PROGRESS_PATH}"
  done
done

echo "========== chunked LIBERO eval completed =========="
python examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs" --chunked | tee "${SUMMARY_PATH}"
echo "[chunked_eval] summary=${SUMMARY_PATH}"
