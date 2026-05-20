#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

if [[ ! -f "starVLA/training/train_starvla.py" ]]; then
  echo "Run this script from the StarVLA repository root." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No Python interpreter found. Set PYTHON_BIN to the training environment python." >&2
  exit 1
fi

ACCELERATE_LAUNCH=()
if [[ -n "${ACCELERATE_BIN:-}" && -x "${ACCELERATE_BIN}" ]]; then
  ACCELERATE_LAUNCH=("${ACCELERATE_BIN}" launch)
elif command -v accelerate >/dev/null 2>&1; then
  ACCELERATE_LAUNCH=("$(command -v accelerate)" launch)
else
  if ! "${PYTHON_BIN}" -c "import accelerate" >/dev/null 2>&1; then
    echo "accelerate is not available in PYTHON_BIN=${PYTHON_BIN}." >&2
    echo "Set PYTHON_BIN to the env python that has accelerate installed, or set ACCELERATE_BIN." >&2
    exit 127
  fi
  ACCELERATE_LAUNCH=("${PYTHON_BIN}" -m accelerate.commands.launch)
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
CONFIG_YAML="${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_abc_qwen35_2b_gr00t_multiview.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenGR00T}"
DATA_MIX="${DATA_MIX:-calvin_task_ABC_D_multiview}"
BASE_VLM="${BASE_VLM:-./playground/Pretrained_models/Qwen3.5-2B}"
BASE_WM="${BASE_WM:-}"
BASE_WM_CHECKPOINT="${BASE_WM_CHECKPOINT:-}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
IS_RESUME="${IS_RESUME:-false}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"
RELOAD_MODULES="${RELOAD_MODULES:-}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-qwen35_2b_gr00t_calvin_abc_multiview}"

if [[ "${BASE_VLM}" == ./* || "${BASE_VLM}" == /* ]]; then
  if [[ ! -e "${BASE_VLM}" ]]; then
    echo "BASE_VLM points to a local path that does not exist: ${BASE_VLM}" >&2
    echo "Set BASE_VLM to the actual Qwen3.5-2B checkpoint directory." >&2
    exit 1
  fi
fi

if [[ -n "${BASE_WM}" && ( "${BASE_WM}" == ./* || "${BASE_WM}" == /* ) ]]; then
  if [[ ! -e "${BASE_WM}" ]]; then
    echo "BASE_WM points to a local path that does not exist: ${BASE_WM}" >&2
    echo "Set BASE_WM to the actual Cosmos-Predict2 checkpoint directory." >&2
    exit 1
  fi
fi

if [[ -n "${BASE_WM_CHECKPOINT}" && ! -f "${BASE_WM_CHECKPOINT}" ]]; then
  echo "BASE_WM_CHECKPOINT points to a file that does not exist: ${BASE_WM_CHECKPOINT}" >&2
  exit 1
fi

RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${RUN_ID_PREFIX}_${RUN_TIMESTAMP}}"

MODEL_ARGS=(--framework.qwenvl.base_vlm "${BASE_VLM}")
if [[ -n "${BASE_WM}" ]]; then
  MODEL_ARGS+=(--framework.world_model.base_wm "${BASE_WM}")
fi
if [[ -n "${BASE_WM_CHECKPOINT}" ]]; then
  MODEL_ARGS+=(--framework.world_model.transformer_checkpoint "${BASE_WM_CHECKPOINT}")
fi

TRAINER_ARGS=(--trainer.is_resume "${IS_RESUME}")
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  TRAINER_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi
if [[ -n "${RELOAD_MODULES}" ]]; then
  TRAINER_ARGS+=(--trainer.reload_modules "${RELOAD_MODULES}")
fi

echo "===== StarVLA Qwen3.5 multiview single-node launch ====="
echo "CONFIG_YAML=${CONFIG_YAML}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "FRAMEWORK_NAME=${FRAMEWORK_NAME}"
echo "DATA_MIX=${DATA_MIX}"
echo "BASE_VLM=${BASE_VLM}"
echo "BASE_WM=${BASE_WM}"
echo "BASE_WM_CHECKPOINT=${BASE_WM_CHECKPOINT}"
echo "FREEZE_MODULES=${FREEZE_MODULES}"
echo "IS_RESUME=${IS_RESUME}"
echo "PRETRAINED_CHECKPOINT=${PRETRAINED_CHECKPOINT}"
echo "RELOAD_MODULES=${RELOAD_MODULES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ACCELERATE_LAUNCH=${ACCELERATE_LAUNCH[*]}"
echo "========================================================="

"${ACCELERATE_LAUNCH[@]}" \
  --config_file "${DEEPSPEED_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK_NAME}" \
  "${MODEL_ARGS[@]}" \
  --datasets.vla_data.data_root_dir . \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  "${TRAINER_ARGS[@]}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
