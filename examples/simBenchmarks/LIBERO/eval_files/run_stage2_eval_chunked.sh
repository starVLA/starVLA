#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <stage2_checkpoint.pt> [gpu_id] [base_port]"
  echo "Example smoke: TRIALS_PER_TASK=3 TASK_START=0 TASK_COUNT=1 $0 playground/Checkpoints/run/checkpoints/steps_50000_pytorch_model.pt 9 18620"
  echo "Example full:  TRIALS_PER_TASK=50 CHUNK_TRIALS=5 $0 playground/Checkpoints/run/checkpoints/steps_50000_pytorch_model.pt 9 18620"
  exit 2
fi

CKPT="$1"
GPU_ID="${2:-9}"
BASE_PORT="${3:-18620}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_HOME="${LIBERO_HOME:-/root/feihong/LIBERO}"
export STARVLA_PYTHON="${STARVLA_PYTHON:-/root/feihong/starVLA/.venv/bin/python}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-/root/feihong/LIBERO/.venv/bin/python}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}"
if [[ -n "${TASK_SUITES_OVERRIDE:-}" ]]; then
  read -r -a TASK_SUITES <<< "${TASK_SUITES_OVERRIDE}"
else
  TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)
fi

# Stage2 QwenVAR training/eval contract.
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-5}"
MAX_RETRIES="${MAX_RETRIES:-100000}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-224}"
# Match QwenVAR training config and FAST baseline eval by default. Hard
# constrained generation is available for debugging via env override, but it is
# not the training-aligned stage2 eval path.
CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-0}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_stage2}"
TASK_START_VALUE="${TASK_START:-0}"
TASK_COUNT_VALUE="${TASK_COUNT:--1}"
TRIAL_START_VALUE="${TRIAL_START:-0}"

MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
SUMMARY_PATH="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs/libero_40task_summary.txt"
PROGRESS_PATH="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs/libero_40task_progress.txt"

mkdir -p "$(dirname "${SUMMARY_PATH}")"

chunk_completed() {
  local log_path="$1"
  local expected_episodes="$2"

  [[ -f "${log_path}" ]] || return 1
  grep -q "EVAL_CHUNK_OK" "${log_path}" || return 1
  grep -q "Total success rate:" "${log_path}" || return 1
  grep -q "Total episodes: ${expected_episodes}" "${log_path}" || return 1
  return 0
}

chunk_covered_by_existing_log() {
  local suite="$1"
  local task_id="$2"
  local trial_start="$3"
  local chunk="$4"
  local ckpt_base log_path base start count end wanted_end

  ckpt_base="$(basename "${CKPT}" .pt)"
  wanted_end=$((trial_start + chunk))

  shopt -s nullglob
  for log_path in "${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs/${suite}/${ckpt_base}"_stage2_chunked_t"${task_id}"_r*_n*.log; do
    base="$(basename "${log_path}")"
    if [[ "${base}" =~ _t${task_id}_r([0-9]+)_n([0-9]+)\.log$ ]]; then
      start="${BASH_REMATCH[1]}"
      count="${BASH_REMATCH[2]}"
      end=$((start + count))
      if [[ "${trial_start}" -ge "${start}" && "${wanted_end}" -le "${end}" ]]; then
        chunk_completed "${log_path}" "${count}" && return 0
      fi
    fi
  done
  return 1
}

for suite_idx in "${!TASK_SUITES[@]}"; do
  suite="${TASK_SUITES[$suite_idx]}"
  port=$((BASE_PORT + suite_idx))
  for task_id in $(seq 0 9); do
    if [[ "${task_id}" -lt "${TASK_START_VALUE}" ]]; then
      continue
    fi
    if [[ "${TASK_COUNT_VALUE}" != "-1" ]]; then
      end_task=$((TASK_START_VALUE + TASK_COUNT_VALUE))
      if [[ "${task_id}" -ge "${end_task}" ]]; then
        continue
      fi
    fi

    trial_start="${TRIAL_START_VALUE}"
    while [[ "${trial_start}" -lt "${TRIALS_PER_TASK}" ]]; do
      remaining=$((TRIALS_PER_TASK - trial_start))
      chunk="${CHUNK_TRIALS}"
      if [[ "${remaining}" -lt "${chunk}" ]]; then
        chunk="${remaining}"
      fi

      log_suffix="_stage2_chunked_t${task_id}_r${trial_start}_n${chunk}"
      log_path="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs/${suite}/$(basename "${CKPT}" .pt)${log_suffix}.log"
      if chunk_completed "${log_path}" "${chunk}" || chunk_covered_by_existing_log "${suite}" "${task_id}" "${trial_start}" "${chunk}"; then
        echo "[stage2_eval] skip completed chunk: suite=${suite} task=${task_id} trial_start=${trial_start} chunk=${chunk} log=${log_path}"
        trial_start=$((trial_start + chunk))
        continue
      fi
      attempt=1
      while true; do
        echo "========== stage2 suite=${suite} task=${task_id} trials=${trial_start}..$((trial_start + chunk - 1)) attempt=${attempt} =========="
        if timeout --kill-after=60s "${CHUNK_TIMEOUT_SECONDS}" env \
          TASK_START="${task_id}" TASK_COUNT=1 TRIAL_START="${trial_start}" \
          NUM_TRIALS_PER_TASK="${chunk}" MAX_TASKS=-1 UNNORM_KEY="${UNNORM_KEY}" \
          SAVE_VIDEOS="${SAVE_VIDEOS}" IMAGE_VIEWS="${IMAGE_VIEWS}" \
          POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE}" \
          CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS}" \
          MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS}" \
          VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}" \
          STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}" \
          MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN:-2.0}" \
          MIN_IMAGE_STD="${MIN_IMAGE_STD:-1.0}" \
          EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
          LOG_SUFFIX="${log_suffix}" \
          "${SCRIPT_DIR}/run_local_eval_once.sh" "${CKPT}" "${suite}" "${GPU_ID}" "${port}"; then
          break
        fi
        if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
          echo "[stage2_eval] failed after ${MAX_RETRIES} attempts: suite=${suite} task=${task_id} trial_start=${trial_start} chunk=${chunk}" >&2
          exit 1
        fi
        attempt=$((attempt + 1))
        sleep 5
      done
      trial_start=$((trial_start + chunk))
    done

    {
      echo "========== stage2 progress after suite=${suite} task=${task_id} =========="
      date
      "${LIBERO_PYTHON:-python}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs" --chunked --require-ok-marker
      echo
    } | tee "${PROGRESS_PATH}"
  done
done

echo "========== stage2 LIBERO eval completed =========="
"${LIBERO_PYTHON:-python}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs" --chunked --require-ok-marker | tee "${SUMMARY_PATH}"
echo "[stage2_eval] summary=${SUMMARY_PATH}"
