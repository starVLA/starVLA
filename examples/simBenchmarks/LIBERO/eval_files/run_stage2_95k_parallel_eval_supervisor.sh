#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-/home/zhangfeihong/starVLA/playground/Checkpoints/qwen_var_productvq_g8_aligned_fast_100k_skipbad/checkpoints/steps_95000_pytorch_model.pt}"
STARVLA_DIR="${STARVLA_DIR:-/home/zhangfeihong/starVLA}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_stage2_95k_40task}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-5}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"

MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
LOG_ROOT="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs"
SUMMARY_PATH="${LOG_ROOT}/libero_40task_summary.txt"
SUPERVISOR_LOG="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/stage2_95k_parallel_supervisor.log"

mkdir -p "${LOG_ROOT}"

SUITES=(libero_spatial libero_object libero_goal libero_10)

# One supervisor job owns either a whole suite or a disjoint task range.
# object/goal have shown intermittent LIBERO client aborts before a 5-trial
# chunk completes. Single-trial chunks preserve completed episodes across
# retries while multiple task shards recover throughput.
JOB_SUITES=(libero_spatial libero_object libero_object libero_object libero_goal libero_goal libero_goal libero_10)
JOB_GPUS=(8 1 4 5 2 6 7 3)
JOB_PORTS=(18720 18730 18731 18732 18740 18741 18742 18750)
JOB_SESSIONS=(stage2_95k_spatial stage2_95k_object_t0_2 stage2_95k_object_t3_5 stage2_95k_object_t6_9 stage2_95k_goal_t0_2 stage2_95k_goal_t3_5 stage2_95k_goal_t6_9 stage2_95k_10)
JOB_TASK_STARTS=(0 0 3 6 0 3 6 0)
JOB_TASK_COUNTS=(-1 3 3 4 3 3 4 -1)
JOB_CHUNKS=(5 1 1 1 1 1 1 5)

chunk_completed() {
  local log_path="$1"
  local expected_episodes="$2"
  [[ -f "${log_path}" ]] || return 1
  grep -q "Total success rate:" "${log_path}" || return 1
  grep -q "Total episodes: ${expected_episodes}" "${log_path}" || return 1
}

suite_completed() {
  local suite="$1"
  local suite_chunk="$2"
  local ckpt_base
  ckpt_base="$(basename "${CKPT}" .pt)"
  local task_id trial_start remaining chunk log_path

  for task_id in $(seq 0 9); do
    trial_start=0
    while [[ "${trial_start}" -lt "${TRIALS_PER_TASK}" ]]; do
      remaining=$((TRIALS_PER_TASK - trial_start))
      chunk="${suite_chunk}"
      if [[ "${remaining}" -lt "${chunk}" ]]; then
        chunk="${remaining}"
      fi
      log_path="${LOG_ROOT}/${suite}/${ckpt_base}_stage2_chunked_t${task_id}_r${trial_start}_n${chunk}.log"
      chunk_completed "${log_path}" "${chunk}" || return 1
      trial_start=$((trial_start + chunk))
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
  local task_start="$6"
  local task_count="$7"
  local launch_log="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/launch_${suite}.log"

  if tmux has-session -t "${session}" >/dev/null 2>&1; then
    return 0
  fi

  echo "[$(date)] launching ${suite} tasks=${task_start}:${task_count} on gpu=${gpu} port=${port} session=${session}" | tee -a "${SUPERVISOR_LOG}"
  tmux new-session -d -s "${session}" \
    "cd ${STARVLA_DIR}; TASK_SUITES_OVERRIDE=${suite} TASK_START=${task_start} TASK_COUNT=${task_count} EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT} TRIALS_PER_TASK=${TRIALS_PER_TASK} CHUNK_TRIALS=${suite_chunk} MAX_RETRIES=100000 CHUNK_TIMEOUT_SECONDS=1800 SAVE_VIDEOS=0 IMAGE_VIEWS=primary+wrist POLICY_IMAGE_SIZE=224 CONSTRAIN_TO_ACTION_TOKENS=0 CLIP_NORMALIZED_ACTIONS=0 bash examples/simBenchmarks/LIBERO/eval_files/run_stage2_eval_chunked.sh ${CKPT} ${gpu} ${port} >> ${launch_log} 2>&1"
}

cd "${STARVLA_DIR}"
echo "[$(date)] stage2 95k parallel eval supervisor started" | tee -a "${SUPERVISOR_LOG}"

while true; do
  completed=0
  for suite in "${SUITES[@]}"; do
    if [[ "${suite}" == "libero_object" || "${suite}" == "libero_goal" ]]; then
      suite_chunk=1
    else
      suite_chunk=5
    fi
    if suite_completed "${suite}" "${suite_chunk}"; then
      completed=$((completed + 1))
    fi
  done

  for idx in "${!JOB_SUITES[@]}"; do
    suite="${JOB_SUITES[$idx]}"
    if [[ "${suite}" == "libero_object" || "${suite}" == "libero_goal" ]]; then
      suite_chunk=1
    else
      suite_chunk=5
    fi
    suite_completed "${suite}" "${suite_chunk}" && continue
    launch_suite "${suite}" "${JOB_GPUS[$idx]}" "${JOB_PORTS[$idx]}" "${JOB_SESSIONS[$idx]}" "${JOB_CHUNKS[$idx]}" "${JOB_TASK_STARTS[$idx]}" "${JOB_TASK_COUNTS[$idx]}"
  done

  python examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${LOG_ROOT}" --chunked > "${LOG_ROOT}/libero_40task_progress.txt" || true

  if [[ "${completed}" -eq "${#SUITES[@]}" ]]; then
    python examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${LOG_ROOT}" --chunked | tee "${SUMMARY_PATH}"
    echo "[$(date)] completed all suites; summary=${SUMMARY_PATH}" | tee -a "${SUPERVISOR_LOG}"
    exit 0
  fi

  sleep "${CHECK_INTERVAL_SECONDS}"
done
