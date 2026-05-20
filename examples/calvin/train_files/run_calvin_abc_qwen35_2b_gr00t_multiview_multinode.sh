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

MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-29500}}"
NNODES="${PET_NNODES:-}"
NPROC_PER_NODE="${PET_NPROC_PER_NODE:-}"
NODE_RANK="${PET_NODE_RANK:-${RANK:-0}}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "MASTER_ADDR or PET_MASTER_ADDR must be set for multi-node training." >&2
  exit 1
fi

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    echo "PET_NPROC_PER_NODE is not set and nvidia-smi is unavailable." >&2
    exit 1
  fi
fi

if [[ -z "${NNODES}" ]]; then
  if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" -ge "${NPROC_PER_NODE}" ]]; then
    if (( WORLD_SIZE % NPROC_PER_NODE == 0 && WORLD_SIZE > NPROC_PER_NODE )); then
      NNODES="$((WORLD_SIZE / NPROC_PER_NODE))"
    else
      NNODES="${WORLD_SIZE}"
    fi
  else
    echo "PET_NNODES is not set and WORLD_SIZE cannot determine node count." >&2
    exit 1
  fi
fi

TOTAL_PROCESSES="$((NNODES * NPROC_PER_NODE))"
if [[ -n "${WORLD_SIZE:-}" && "${WORLD_SIZE}" -ne "${NNODES}" && "${WORLD_SIZE}" -ne "${TOTAL_PROCESSES}" ]]; then
  echo "Warning: WORLD_SIZE=${WORLD_SIZE} is neither node count (${NNODES}) nor process count (${TOTAL_PROCESSES}); using ${TOTAL_PROCESSES} for accelerate." >&2
fi

JOB_SUFFIX="${TRAIN_JOB_ID:-job}"
ROUND_SUFFIX="${RUNNING_ROUND:-0}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-qwen35_2b_gr00t_calvin_abc_multiview}"

mkdir -p "${RUN_ROOT_DIR}/.run_timestamps"
RUN_STAMP_FILE="${RUN_STAMP_FILE:-${RUN_ROOT_DIR}/.run_timestamps/${JOB_SUFFIX}_round${ROUND_SUFFIX}.txt}"
if [[ -z "${RUN_TIMESTAMP:-}" ]]; then
  if [[ "${NODE_RANK}" == "0" ]]; then
    if [[ ! -s "${RUN_STAMP_FILE}" ]]; then
      date +%Y%m%d_%H%M%S > "${RUN_STAMP_FILE}"
    fi
  else
    for _ in $(seq 1 120); do
      [[ -s "${RUN_STAMP_FILE}" ]] && break
      sleep 1
    done
  fi

  if [[ ! -s "${RUN_STAMP_FILE}" ]]; then
    echo "Failed to read shared run timestamp file: ${RUN_STAMP_FILE}" >&2
    echo "Set RUN_TIMESTAMP explicitly if RUN_ROOT_DIR is not shared across nodes." >&2
    exit 1
  fi
  RUN_TIMESTAMP="$(tr -d '[:space:]' < "${RUN_STAMP_FILE}")"
fi

RUN_ID="${RUN_ID:-${RUN_ID_PREFIX}_${JOB_SUFFIX}_round${ROUND_SUFFIX}_${RUN_TIMESTAMP}}"

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

export MASTER_ADDR
export MASTER_PORT
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "===== StarVLA Qwen3.5 multiview multi-node launch ====="
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
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "ACCELERATE_LAUNCH=${ACCELERATE_LAUNCH[*]}"
echo "RUN_TIMESTAMP=${RUN_TIMESTAMP}"
echo "RUN_STAMP_FILE=${RUN_STAMP_FILE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "TOTAL_PROCESSES=${TOTAL_PROCESSES}"
echo "NODE_RANK=${NODE_RANK}"
echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA:-}"
echo "========================================================"

"${ACCELERATE_LAUNCH[@]}" \
  --config_file "${DEEPSPEED_CONFIG}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --machine_rank "${NODE_RANK}" \
  --num_machines "${NNODES}" \
  --num_processes "${TOTAL_PROCESSES}" \
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
