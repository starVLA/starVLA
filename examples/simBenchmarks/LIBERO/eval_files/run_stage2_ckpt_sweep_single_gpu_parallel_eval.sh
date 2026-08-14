#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-/root/feihong/starVLA}"
LIBERO_HOME="${LIBERO_HOME:-/root/feihong/LIBERO}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${STARVLA_DIR}/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_HOME}/.venv/bin/python}"

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
SWEEP_STEPS="${SWEEP_STEPS:-26000 27000 28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000}"
EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-eval_sweep_26k_to_40k_40task_50ep_robust_seed7_20260709}"

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-19250}"
USE_BF16="${USE_BF16:-1}"
EVAL_SEED="${EVAL_SEED:-7}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-1}"
PARALLEL_CHUNKS="${PARALLEL_CHUNKS:-4}"
MAX_RETRIES="${MAX_RETRIES:-100000}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-900}"
SERVER_READY_POLL_SECONDS="${SERVER_READY_POLL_SECONDS:-2}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-224}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}"
MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN:-2.0}"
MIN_IMAGE_STD="${MIN_IMAGE_STD:-1.0}"
CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
PROGRESS_EVERY_WAVES="${PROGRESS_EVERY_WAVES:-1}"

SUITES=(libero_spatial libero_object libero_goal libero_10)

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CLIP_NORMALIZED_ACTIONS

SERVER_PID=""
SERVER_LOG=""

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT

server_alive() {
  [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1
}

start_server() {
  local ckpt="$1"
  local log_root="$2"
  stop_server
  mkdir -p "${log_root}"
  SERVER_LOG="${log_root}/policy_server.log"
  : > "${SERVER_LOG}"

  local server_args=(deployment/model_server/server_policy.py --ckpt_path "${ckpt}" --port "${PORT}" --idle_timeout -1)
  if [[ "${USE_BF16}" == "1" ]]; then
    server_args+=(--use_bf16)
  fi

  echo "[$(date)] starting policy server ckpt=${ckpt} gpu=${GPU_ID} port=${PORT} use_bf16=${USE_BF16}" | tee -a "${SERVER_LOG}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" "${server_args[@]}" >> "${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!

  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
  while [[ "${SECONDS}" -lt "${deadline}" ]]; do
    if grep -q "server running" "${SERVER_LOG}"; then
      echo "[$(date)] policy server ready pid=${SERVER_PID}" | tee -a "${SERVER_LOG}"
      return 0
    fi
    if ! server_alive; then
      echo "[$(date)] policy server exited before ready; tail follows" | tee -a "${SERVER_LOG}"
      tail -120 "${SERVER_LOG}" || true
      return 1
    fi
    sleep "${SERVER_READY_POLL_SECONDS}"
  done

  echo "[$(date)] policy server did not become ready within ${SERVER_READY_TIMEOUT_SECONDS}s" | tee -a "${SERVER_LOG}"
  tail -120 "${SERVER_LOG}" || true
  return 1
}

chunk_completed() {
  local log_path="$1"
  local expected_episodes="$2"
  [[ -f "${log_path}" ]] || return 1
  grep -q "EVAL_CHUNK_OK" "${log_path}" || return 1
  grep -q "Total success rate:" "${log_path}" || return 1
  grep -q "Total episodes: ${expected_episodes}" "${log_path}" || return 1
  return 0
}

run_eval_chunk_once() {
  local ckpt="$1"
  local suite="$2"
  local task_id="$3"
  local trial_start="$4"
  local chunk="$5"
  local log_path="$6"
  local video_out_path="$7"

  local eval_args=(
    examples/simBenchmarks/LIBERO/eval_files/eval_libero.py
    --args.pretrained-path "${ckpt}"
    --args.host 127.0.0.1
    --args.port "${PORT}"
    --args.task-suite-name "${suite}"
    --args.num-trials-per-task "${chunk}"
    --args.max-tasks -1
    --args.task-start "${task_id}"
    --args.task-count 1
    --args.trial-start "${trial_start}"
    --args.seed "${EVAL_SEED}"
    --args.unnorm-key "${UNNORM_KEY}"
    --args.video-out-path "${video_out_path}"
    --args.image-views "${IMAGE_VIEWS}"
    --args.min-image-mean "${MIN_IMAGE_MEAN}"
    --args.min-image-std "${MIN_IMAGE_STD}"
  )

  if [[ "${SAVE_VIDEOS}" == "1" ]]; then
    eval_args+=(--args.save-videos)
  else
    eval_args+=(--args.no-save-videos)
  fi
  if [[ "${VALIDATE_INPUTS}" != "1" ]]; then
    eval_args+=(--args.no-validate-inputs)
  fi
  if [[ "${STRICT_TRIAL_COUNT}" != "1" ]]; then
    eval_args+=(--args.no-strict-trial-count)
  fi
  if [[ "${POLICY_IMAGE_SIZE}" != "0" ]]; then
    eval_args+=(--args.policy-image-size "${POLICY_IMAGE_SIZE}")
  fi
  if [[ "${CONSTRAIN_TO_ACTION_TOKENS}" == "1" ]]; then
    eval_args+=(--args.constrain-to-action-tokens)
  fi
  if [[ -n "${MAX_NEW_TOKENS}" ]]; then
    eval_args+=(--args.max-new-tokens "${MAX_NEW_TOKENS}")
  fi

  timeout --kill-after=60s "${CHUNK_TIMEOUT_SECONDS}" \
    "${LIBERO_PYTHON}" "${eval_args[@]}" > "${log_path}" 2>&1
}

summarize_ckpt() {
  local log_root="$1"
  local output_path="$2"
  local episode_csv="$3"
  local episode_jsonl="$4"
  "${LIBERO_PYTHON}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py \
    "${log_root}" --chunked --require-ok-marker \
    --episode-csv "${episode_csv}" \
    --episode-jsonl "${episode_jsonl}" | tee "${output_path}"
}

count_completed_chunks() {
  local log_root="$1"
  local ckpt_base="$2"
  local total=0
  local suite task_id trial_start remaining chunk log_path
  for suite in "${SUITES[@]}"; do
    for task_id in $(seq 0 9); do
      trial_start=0
      while [[ "${trial_start}" -lt "${TRIALS_PER_TASK}" ]]; do
        remaining=$((TRIALS_PER_TASK - trial_start))
        chunk="${CHUNK_TRIALS}"
        if [[ "${remaining}" -lt "${chunk}" ]]; then
          chunk="${remaining}"
        fi
        log_path="${log_root}/${suite}/${ckpt_base}_stage2_chunked_t${task_id}_r${trial_start}_n${chunk}.log"
        if chunk_completed "${log_path}" "${chunk}"; then
          total=$((total + 1))
        fi
        trial_start=$((trial_start + chunk))
      done
    done
  done
  echo "${total}"
}

expected_chunks_per_ckpt() {
  local chunks_per_task=$(((TRIALS_PER_TASK + CHUNK_TRIALS - 1) / CHUNK_TRIALS))
  echo $(( ${#SUITES[@]} * 10 * chunks_per_task ))
}

run_one_ckpt() {
  local step="$1"
  local ckpt="${CHECKPOINT_DIR}/steps_${step}_pytorch_model.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[$(date)] missing checkpoint: ${ckpt}" >&2
    exit 1
  fi

  local ckpt_base eval_root log_root supervisor_log summary_path progress_path episode_csv episode_jsonl expected_total wave
  ckpt_base="$(basename "${ckpt}" .pt)"
  eval_root="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_steps_${step}"
  log_root="${eval_root}/logs"
  supervisor_log="${eval_root}/single_gpu_parallel_eval_supervisor.log"
  summary_path="${log_root}/libero_40task_summary.txt"
  progress_path="${log_root}/libero_40task_progress.txt"
  episode_csv="${log_root}/libero_40task_episodes.csv"
  episode_jsonl="${log_root}/libero_40task_episodes.jsonl"
  expected_total="$(expected_chunks_per_ckpt)"
  wave=0

  mkdir -p "${log_root}"
  echo "[$(date)] ===== start ckpt step=${step} output=${eval_root} expected_chunks=${expected_total} parallel=${PARALLEL_CHUNKS} =====" | tee -a "${supervisor_log}"
  echo "[$(date)] settings seed=${EVAL_SEED} trials=${TRIALS_PER_TASK} chunk=${CHUNK_TRIALS} gpu=${GPU_ID} port=${PORT} use_bf16=${USE_BF16}" | tee -a "${supervisor_log}"

  start_server "${ckpt}" "${log_root}" >> "${supervisor_log}" 2>&1

  declare -A attempts=()
  while true; do
    local completed
    completed="$(count_completed_chunks "${log_root}" "${ckpt_base}")"
    echo "[$(date)] progress ${ckpt_base}: completed_chunks=${completed}/${expected_total}" | tee -a "${supervisor_log}"
    if [[ "${completed}" -ge "${expected_total}" ]]; then
      break
    fi

    if ! server_alive; then
      echo "[$(date)] server not alive before scheduling; restarting" | tee -a "${supervisor_log}"
      start_server "${ckpt}" "${log_root}" >> "${supervisor_log}" 2>&1
    fi

    wave=$((wave + 1))
    local active=0
    local pids=()
    local labels=()
    local paths=()
    local chunks=()
    local suite task_id trial_start remaining chunk log_dir video_out_path log_path label attempt

    for suite in "${SUITES[@]}"; do
      for task_id in $(seq 0 9); do
        trial_start=0
        while [[ "${trial_start}" -lt "${TRIALS_PER_TASK}" ]]; do
          remaining=$((TRIALS_PER_TASK - trial_start))
          chunk="${CHUNK_TRIALS}"
          if [[ "${remaining}" -lt "${chunk}" ]]; then
            chunk="${remaining}"
          fi
          log_dir="${log_root}/${suite}"
          video_out_path="${eval_root}/videos/${suite}/${ckpt_base}"
          mkdir -p "${log_dir}" "${video_out_path}"
          log_path="${log_dir}/${ckpt_base}_stage2_chunked_t${task_id}_r${trial_start}_n${chunk}.log"

          if chunk_completed "${log_path}" "${chunk}"; then
            trial_start=$((trial_start + chunk))
            continue
          fi

          attempt="$(( ${attempts[${log_path}]:-0} + 1 ))"
          attempts["${log_path}"]="${attempt}"
          if [[ "${attempt}" -gt "${MAX_RETRIES}" ]]; then
            echo "[$(date)] failed after ${MAX_RETRIES} attempts: ${ckpt_base} ${suite} task=${task_id} trial=${trial_start}" | tee -a "${supervisor_log}" >&2
            exit 1
          fi

          label="${suite}:t${task_id}:r${trial_start}:n${chunk}:a${attempt}"
          echo "[$(date)] wave=${wave} launch ${ckpt_base} ${label}" | tee -a "${supervisor_log}"
          run_eval_chunk_once "${ckpt}" "${suite}" "${task_id}" "${trial_start}" "${chunk}" "${log_path}" "${video_out_path}" &
          pids+=("$!")
          labels+=("${label}")
          paths+=("${log_path}")
          chunks+=("${chunk}")
          active=$((active + 1))

          if [[ "${active}" -ge "${PARALLEL_CHUNKS}" ]]; then
            break 3
          fi
          trial_start=$((trial_start + chunk))
        done
      done
    done

    if [[ "${#pids[@]}" -eq 0 ]]; then
      echo "[$(date)] no launchable pending chunks but completion not reached; sleeping" | tee -a "${supervisor_log}"
      sleep 5
      continue
    fi

    local idx status failures=0
    for idx in "${!pids[@]}"; do
      status=0
      wait "${pids[$idx]}" || status=$?
      if [[ "${status}" -eq 0 ]] && chunk_completed "${paths[$idx]}" "${chunks[$idx]}"; then
        echo "[$(date)] wave=${wave} ok ${labels[$idx]}" | tee -a "${supervisor_log}"
      else
        failures=$((failures + 1))
        echo "[$(date)] wave=${wave} incomplete/failed status=${status} ${labels[$idx]} log=${paths[$idx]}" | tee -a "${supervisor_log}"
      fi
    done

    if ! server_alive; then
      echo "[$(date)] server died during wave=${wave}; restarting" | tee -a "${supervisor_log}"
      start_server "${ckpt}" "${log_root}" >> "${supervisor_log}" 2>&1
    fi

    if [[ $((wave % PROGRESS_EVERY_WAVES)) -eq 0 ]]; then
      {
        echo "========== progress ${ckpt_base} wave=${wave} =========="
        date
        "${LIBERO_PYTHON}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${log_root}" --chunked --require-ok-marker || true
        echo
      } | tee "${progress_path}" >> "${supervisor_log}"
    fi
  done

  summarize_ckpt "${log_root}" "${summary_path}" "${episode_csv}" "${episode_jsonl}"
  echo "[$(date)] ===== completed ckpt step=${step}; summary=${summary_path} =====" | tee -a "${supervisor_log}"
  stop_server
}

for step in ${SWEEP_STEPS}; do
  run_one_ckpt "${step}"
done

echo "[$(date)] completed all requested checkpoints: ${SWEEP_STEPS}"
