#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

# Multi-node launcher for Cosmos-Predict2-2B + GR00T on CALVIN ABC LeRobot data.
# Expected to run from the StarVLA repository root.
#
# Scheduler-provided variables supported by this script:
#   NCCL_IB_QPS_PER_CONNECTION, NCCL_GDR_LEVEL, NCCL_IB_PCI_RELAXED_ORDERING,
#   NCCL_IB_TC, NCCL_NVLS_ENABLE, NCCL_IB_GID_INDEX, GLOO_SOCKET_IFNAME,
#   NCCL_SOCKET_IFNAME, NCCL_DEBUG, NCCL_IB_TIMEOUT, NCCL_IB_RETRY_CNT,
#   NCCL_IB_HCA, MASTER_PORT, MASTER_ADDR, PET_MASTER_PORT, PET_MASTER_ADDR,
#   WORLD_SIZE, RANK, PET_NPROC_PER_NODE, PET_NNODES, PET_NODE_RANK,
#   TRAIN_JOB_ID, RUNNING_ROUND.

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

CONFIG_YAML="${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_abc_cosmopredict2_gr00t_online.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"

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
      # Some schedulers expose WORLD_SIZE as node count, not process count.
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

RUN_ID="${RUN_ID:-cosmopredict2_gr00t_calvin_abc_${JOB_SUFFIX}_round${ROUND_SUFFIX}_${RUN_TIMESTAMP}}"

# Preserve scheduler-provided NCCL/Gloo values. Set only conservative defaults
# for variables that are often absent.
export MASTER_ADDR
export MASTER_PORT
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="offline"

echo "===== StarVLA multi-node launch ====="
echo "CONFIG_YAML=${CONFIG_YAML}"
echo "DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"
echo "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "RUN_ID=${RUN_ID}"
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
echo "====================================="

"${ACCELERATE_LAUNCH[@]}" \
  --config_file "${DEEPSPEED_CONFIG}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --machine_rank "${NODE_RANK}" \
  --num_machines "${NNODES}" \
  --num_processes "${TOTAL_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name CosmoPredict2GR00T \
  --datasets.vla_data.data_root_dir . \
  --datasets.vla_data.data_mix calvin_task_ABC_D \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"