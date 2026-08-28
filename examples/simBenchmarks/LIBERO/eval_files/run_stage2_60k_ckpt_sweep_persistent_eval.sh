#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-/root/feihong/starVLA}"
LIBERO_HOME="${LIBERO_HOME:-/root/feihong/LIBERO}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${STARVLA_DIR}/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_HOME}/.venv/bin/python}"

RUN_ROOT="${RUN_ROOT:-/root/nas/feihong/starVLA/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_60k_fullcache_saveall}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
SWEEP_STEPS="${SWEEP_STEPS:-50000 48000 46000 44000 42000 40000}"
EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-eval_sweep_40k_to_50k_40task_50ep_robust_seed7_20260628}"

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-19250}"
USE_BF16="${USE_BF16:-0}"
EVAL_SEED="${EVAL_SEED:-7}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-1}"
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

SUITES=(libero_spatial libero_object libero_goal libero_10)

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/.libero_config"
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

run_eval_chunk() {
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
  "${LIBERO_PYTHON}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${log_root}" --chunked --require-ok-marker | tee "${output_path}"
}

for step in ${SWEEP_STEPS}; do
  ckpt="${CHECKPOINT_DIR}/steps_${step}_pytorch_model.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[$(date)] missing checkpoint: ${ckpt}" >&2
    exit 1
  fi

  ckpt_base="$(basename "${ckpt}" .pt)"
  eval_output_root="${EVAL_OUTPUT_PREFIX}_steps_${step}"
  eval_root="${RUN_ROOT}/${eval_output_root}"
  log_root="${eval_root}/logs"
  supervisor_log="${eval_root}/persistent_eval_supervisor.log"
  summary_path="${log_root}/libero_40task_summary.txt"
  progress_path="${log_root}/libero_40task_progress.txt"

  mkdir -p "${log_root}"
  echo "[$(date)] ===== start ckpt step=${step} output=${eval_root} =====" | tee -a "${supervisor_log}"
  echo "[$(date)] settings seed=${EVAL_SEED} trials=${TRIALS_PER_TASK} chunk=${CHUNK_TRIALS} gpu=${GPU_ID} port=${PORT} use_bf16=${USE_BF16}" | tee -a "${supervisor_log}"

  start_server "${ckpt}" "${log_root}" >> "${supervisor_log}" 2>&1

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
          echo "[$(date)] skip completed ${ckpt_base} ${suite} task=${task_id} trial=${trial_start} n=${chunk}" | tee -a "${supervisor_log}"
          trial_start=$((trial_start + chunk))
          continue
        fi

        attempt=1
        while true; do
          if ! server_alive; then
            echo "[$(date)] server not alive before chunk; restarting" | tee -a "${supervisor_log}"
            start_server "${ckpt}" "${log_root}" >> "${supervisor_log}" 2>&1
          fi

          echo "[$(date)] eval ${ckpt_base} ${suite} task=${task_id} trial=${trial_start} n=${chunk} attempt=${attempt}" | tee -a "${supervisor_log}"
          if run_eval_chunk "${ckpt}" "${suite}" "${task_id}" "${trial_start}" "${chunk}" "${log_path}" "${video_out_path}" && chunk_completed "${log_path}" "${chunk}"; then
            break
          fi

          echo "[$(date)] chunk failed or incomplete: ${log_path}" | tee -a "${supervisor_log}"
          if ! server_alive; then
            echo "[$(date)] server died after failed chunk; restarting" | tee -a "${supervisor_log}"
            start_server "${ckpt}" "${log_root}" >> "${supervisor_log}" 2>&1
          fi
          if [[ "${attempt}" -ge "${MAX_RETRIES}" ]]; then
            echo "[$(date)] failed after ${MAX_RETRIES} attempts: ${ckpt_base} ${suite} task=${task_id} trial=${trial_start}" | tee -a "${supervisor_log}" >&2
            exit 1
          fi
          attempt=$((attempt + 1))
          sleep 5
        done

        trial_start=$((trial_start + chunk))
      done

      {
        echo "========== progress ${ckpt_base} after ${suite} task=${task_id} =========="
        date
        summarize_ckpt "${log_root}" "${progress_path}"
        echo
      } | tee -a "${supervisor_log}"
    done
  done

  summarize_ckpt "${log_root}" "${summary_path}"
  echo "[$(date)] ===== completed ckpt step=${step}; summary=${summary_path} =====" | tee -a "${supervisor_log}"
  stop_server
done

echo "[$(date)] completed all requested checkpoints: ${SWEEP_STEPS}"
