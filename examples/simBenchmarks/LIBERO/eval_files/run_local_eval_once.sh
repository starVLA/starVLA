#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint.pt> <task_suite> [gpu_id] [port]"
  echo "Example: $0 playground/Checkpoints/run/checkpoints/steps_32000_pytorch_model.pt libero_spatial 2 18080"
  exit 2
fi

CKPT="$1"
TASK_SUITE="$2"
GPU_ID="${3:-2}"
PORT="${4:-18080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STARVLA_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_DIR="${STARVLA_DIR:-${DEFAULT_STARVLA_DIR}}"
LIBERO_HOME="${LIBERO_HOME:-/root/feihong/LIBERO}"
STARVLA_PYTHON="${STARVLA_PYTHON:-/root/feihong/starVLA/.venv/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/feihong/LIBERO/.venv/bin/python}"

NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
MAX_TASKS="${MAX_TASKS:-1}"
TASK_START="${TASK_START:-0}"
TASK_COUNT="${TASK_COUNT:--1}"
TRIAL_START="${TRIAL_START:-0}"
UNNORM_KEY="${UNNORM_KEY:-franka}"
USE_BF16="${USE_BF16:-1}"
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"
LOG_SUFFIX="${LOG_SUFFIX:-}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_smoke}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-0}"
CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-0}"
SEED="${SEED:-7}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}"
MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN:-2.0}"
MIN_IMAGE_STD="${MIN_IMAGE_STD:-1.0}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-360}"
SERVER_READY_POLL_SECONDS="${SERVER_READY_POLL_SECONDS:-2}"

export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}"
export TOKENIZERS_PARALLELISM=false
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CLIP_NORMALIZED_ACTIONS

cd "${STARVLA_DIR}"

MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
CKPT_BASENAME="$(basename "${CKPT}" .pt)"
RUN_NAME="$(basename "${MODEL_ROOT}")"
VIDEO_OUT_PATH="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/videos/${TASK_SUITE}/${CKPT_BASENAME}"
LOG_DIR="${MODEL_ROOT}/${EVAL_OUTPUT_ROOT}/logs/${TASK_SUITE}"
LOG_STEM="${CKPT_BASENAME}${LOG_SUFFIX}"
LOG_PATH="${LOG_DIR}/${LOG_STEM}.log"

mkdir -p "${VIDEO_OUT_PATH}" "${LOG_DIR}"

SERVER_ARGS=(deployment/model_server/server_policy.py --ckpt_path "${CKPT}" --port "${PORT}" --idle_timeout -1)
if [[ "${USE_BF16}" == "1" ]]; then
  SERVER_ARGS+=(--use_bf16)
fi
EVAL_EXTRA_ARGS=()
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.save-videos)
else
  EVAL_EXTRA_ARGS+=(--args.no-save-videos)
fi
if [[ "${CONSTRAIN_TO_ACTION_TOKENS}" == "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.constrain-to-action-tokens)
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  EVAL_EXTRA_ARGS+=(--args.max-new-tokens "${MAX_NEW_TOKENS}")
fi
if [[ "${POLICY_IMAGE_SIZE}" != "0" ]]; then
  EVAL_EXTRA_ARGS+=(--args.policy-image-size "${POLICY_IMAGE_SIZE}")
fi
if [[ "${VALIDATE_INPUTS}" != "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.no-validate-inputs)
fi
if [[ "${STRICT_TRIAL_COUNT}" != "1" ]]; then
  EVAL_EXTRA_ARGS+=(--args.no-strict-trial-count)
fi
EVAL_EXTRA_ARGS+=(--args.min-image-mean "${MIN_IMAGE_MEAN}")
EVAL_EXTRA_ARGS+=(--args.min-image-std "${MIN_IMAGE_STD}")

echo "[eval] run=${RUN_NAME}"
echo "[eval] ckpt=${CKPT}"
echo "[eval] suite=${TASK_SUITE} trials_per_task=${NUM_TRIALS_PER_TASK} max_tasks=${MAX_TASKS} task_start=${TASK_START} task_count=${TASK_COUNT} trial_start=${TRIAL_START} seed=${SEED} image_views=${IMAGE_VIEWS} policy_image_size=${POLICY_IMAGE_SIZE} constrain_to_action_tokens=${CONSTRAIN_TO_ACTION_TOKENS} max_new_tokens=${MAX_NEW_TOKENS} clip_normalized_actions=${CLIP_NORMALIZED_ACTIONS} validate_inputs=${VALIDATE_INPUTS} strict_trial_count=${STRICT_TRIAL_COUNT} output_root=${EVAL_OUTPUT_ROOT}"
echo "[eval] gpu=${GPU_ID} port=${PORT} mujoco_egl_device_id=${MUJOCO_EGL_DEVICE_ID}"
echo "[eval] videos=${VIDEO_OUT_PATH}"
echo "[eval] log=${LOG_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${STARVLA_PYTHON}" "${SERVER_ARGS[@]}" >"${LOG_DIR}/${CKPT_BASENAME}.server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[eval] waiting for policy server pid=${SERVER_PID} timeout=${SERVER_READY_TIMEOUT_SECONDS}s"
SERVER_READY_DEADLINE=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
while [[ "${SECONDS}" -lt "${SERVER_READY_DEADLINE}" ]]; do
  if grep -q "server running" "${LOG_DIR}/${CKPT_BASENAME}.server.log"; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[eval] policy server exited early; server log:"
    tail -120 "${LOG_DIR}/${CKPT_BASENAME}.server.log" || true
    exit 1
  fi
  sleep "${SERVER_READY_POLL_SECONDS}"
done

if ! grep -q "server running" "${LOG_DIR}/${CKPT_BASENAME}.server.log"; then
  echo "[eval] policy server did not become ready; server log:"
  tail -120 "${LOG_DIR}/${CKPT_BASENAME}.server.log" || true
  exit 1
fi

"${LIBERO_PYTHON}" examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host 127.0.0.1 \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.task-start "${TASK_START}" \
  --args.task-count "${TASK_COUNT}" \
  --args.trial-start "${TRIAL_START}" \
  --args.seed "${SEED}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  --args.image-views "${IMAGE_VIEWS}" \
  "${EVAL_EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_PATH}"

echo "[eval] completed ${TASK_SUITE}; result log=${LOG_PATH}"
