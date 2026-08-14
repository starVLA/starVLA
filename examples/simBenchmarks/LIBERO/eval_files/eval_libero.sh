#!/usr/bin/env bash
set -euo pipefail

# Default feihong LIBERO eval client. Start run_policy_server.sh first, or point
# HOST/PORT at an already-running persistent policy server.
STARVLA_DIR=${STARVLA_DIR:-/root/feihong/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/root/feihong/LIBERO}
LIBERO_PYTHON=${LIBERO_PYTHON:-/root/feihong/LIBERO/.venv/bin/python}
CKPT=${CKPT:-/root/nas/feihong/starVLA/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints/steps_100000_pytorch_model.pt}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-6694}
UNNORM_KEY=${UNNORM_KEY:-franka}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_goal}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
MAX_TASKS=${MAX_TASKS:--1}
TASK_START=${TASK_START:-0}
TASK_COUNT=${TASK_COUNT:--1}
TRIAL_START=${TRIAL_START:-0}
IMAGE_HISTORY=${IMAGE_HISTORY:--1}
MULTIVIEW_PACK=${MULTIVIEW_PACK:-auto}
IMAGE_VIEWS=${IMAGE_VIEWS:-primary+wrist}
POLICY_IMAGE_SIZE=${POLICY_IMAGE_SIZE:-0}
SAVE_VIDEOS=${SAVE_VIDEOS:-1}
VALIDATE_INPUTS=${VALIDATE_INPUTS:-1}
MUJOCO_GL=${MUJOCO_GL:-egl}
PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL PYOPENGL_PLATFORM
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

if [[ "${CKPT}" == *"/checkpoints/"* ]]; then
  MODEL_ROOT=$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')
elif [[ "$(basename "$(dirname "${CKPT}")")" == "final_model" ]]; then
  MODEL_ROOT=$(dirname "$(dirname "${CKPT}")")
else
  MODEL_ROOT=$(dirname "${CKPT}")
fi
FOLDER_NAME=$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-${MODEL_ROOT}/results/${TASK_SUITE_NAME}/${FOLDER_NAME}}

exec "${LIBERO_PYTHON}" ./examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "${CKPT}" \
  --args.host "${HOST}" \
  --args.port "${PORT}" \
  --args.task-suite-name "${TASK_SUITE_NAME}" \
  --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}" \
  --args.max-tasks "${MAX_TASKS}" \
  --args.task-start "${TASK_START}" \
  --args.task-count "${TASK_COUNT}" \
  --args.trial-start "${TRIAL_START}" \
  --args.unnorm-key "${UNNORM_KEY}" \
  --args.video-out-path "${VIDEO_OUT_PATH}" \
  --args.image-history "${IMAGE_HISTORY}" \
  --args.multiview-pack "${MULTIVIEW_PACK}" \
  --args.image-views "${IMAGE_VIEWS}" \
  --args.policy-image-size "${POLICY_IMAGE_SIZE}" \
  --args.save-videos "${SAVE_VIDEOS}" \
  --args.validate-inputs "${VALIDATE_INPUTS}"
