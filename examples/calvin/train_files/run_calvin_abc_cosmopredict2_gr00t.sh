#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Single-node launcher for Cosmos-Predict2-2B + GR00T on CALVIN ABC LeRobot data.
# Run from the StarVLA repository root.

if [[ ! -f "starVLA/training/train_starvla.py" ]]; then
  echo "Run this script from the StarVLA repository root." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
CONFIG_YAML="${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_abc_cosmopredict2_gr00t.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
FRAMEWORK_NAME="${FRAMEWORK_NAME:-CosmoPredict2GR00T}"
DATA_MIX="${DATA_MIX:-calvin_task_ABC_D}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-cosmopredict2_gr00t_calvin_abc}"
BASE_WM="${BASE_WM:-}"
BASE_WM_CHECKPOINT="${BASE_WM_CHECKPOINT:-}"

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_ID_PREFIX}_${RUN_TIMESTAMP}}"

MODEL_ARGS=()
if [[ -n "${BASE_WM}" ]]; then
  if [[ "${BASE_WM}" == /* || "${BASE_WM}" == ./* ]] && [[ ! -d "${BASE_WM}" ]]; then
    echo "BASE_WM does not exist: ${BASE_WM}" >&2
    exit 1
  fi
  MODEL_ARGS+=(--framework.world_model.base_wm "${BASE_WM}")
fi
if [[ -n "${BASE_WM_CHECKPOINT}" ]]; then
  if [[ ! -f "${BASE_WM_CHECKPOINT}" ]]; then
    echo "BASE_WM_CHECKPOINT does not exist: ${BASE_WM_CHECKPOINT}" >&2
    exit 1
  fi
  MODEL_ARGS+=(--framework.world_model.transformer_checkpoint "${BASE_WM_CHECKPOINT}")
fi

echo "===== StarVLA single-node launch ====="
echo "CONFIG_YAML=${CONFIG_YAML}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "FRAMEWORK_NAME=${FRAMEWORK_NAME}"
echo "BASE_WM=${BASE_WM}"
echo "BASE_WM_CHECKPOINT=${BASE_WM_CHECKPOINT}"
echo "DATA_MIX=${DATA_MIX}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "======================================"

accelerate launch \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  "${MODEL_ARGS[@]}" \
  --datasets.vla_data.data_root_dir . \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
