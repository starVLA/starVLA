#!/usr/bin/env bash
set -euo pipefail

# Time-budgeted continuation for MoE95k + fresh LoRA branches.
# It waits for the current short branch to produce a target checkpoint, then
# continues from the latest checkpoint with a larger batch and fewer checkpoint
# writes. This is meant for a limited remaining GPU window.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VARIANT="${VARIANT:-aug}"
case "${VARIANT}" in
  aug|mirror) ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected aug or mirror" >&2
    exit 2
    ;;
esac

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/members/${MEMBER}/logs}"
SOURCE_RUN_PATTERN="${SOURCE_RUN_PATTERN:-abc_moe95k_lora_${VARIANT}_2500_*}"
CURRENT_TARGET_STEPS="${CURRENT_TARGET_STEPS:-2500}"
WAIT_FOR_CURRENT="${WAIT_FOR_CURRENT:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-7200}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
QWEN_LORA_LR="${QWEN_LORA_LR:-5.0e-06}"
ACTION_LR="${ACTION_LR:-5.0e-05}"
TS="${TS:-$(date +%m%d_%H%M%S)}"

latest_run() {
  find "${RUN_ROOT_DIR}" -maxdepth 1 -type d -name "${SOURCE_RUN_PATTERN}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

latest_checkpoint() {
  local run="$1"
  [[ -n "${run}" && -d "${run}/checkpoints" ]] || return 0
  find "${run}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

max_checkpoint_step() {
  local run="$1"
  [[ -n "${run}" && -d "${run}/checkpoints" ]] || {
    echo 0
    return 0
  }
  find "${run}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%f\n' 2>/dev/null \
    | sed -n 's/^steps_\([0-9][0-9]*\)_pytorch_model\.pt$/\1/p' \
    | sort -n \
    | tail -1
}

mkdir -p "${LOG_DIR}"

source_run="$(latest_run)"
if [[ -z "${source_run}" ]]; then
  echo "[moe-lora-3h] no source run found for pattern ${SOURCE_RUN_PATTERN}" >&2
  exit 3
fi

echo "[moe-lora-3h] variant=${VARIANT}"
echo "[moe-lora-3h] source_run=${source_run}"
echo "[moe-lora-3h] target checkpoint step before continue=${CURRENT_TARGET_STEPS}"

if [[ "${WAIT_FOR_CURRENT}" == "1" ]]; then
  start_ts="$(date +%s)"
  while true; do
    current_step="$(max_checkpoint_step "${source_run}")"
    current_step="${current_step:-0}"
    ckpt="$(latest_checkpoint "${source_run}")"
    echo "[moe-lora-3h] current max checkpoint step=${current_step} ckpt=${ckpt:-none}"
    if (( current_step >= CURRENT_TARGET_STEPS )); then
      break
    fi
    if [[ -f "${source_run}/summary.jsonl" ]] && grep -q "\"steps\": ${CURRENT_TARGET_STEPS}" "${source_run}/summary.jsonl"; then
      break
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= WAIT_TIMEOUT_SECONDS )); then
      if [[ -n "${ckpt:-}" ]]; then
        echo "[moe-lora-3h] wait timeout reached; continuing from latest available checkpoint"
        break
      fi
      echo "[moe-lora-3h] wait timeout reached and no checkpoint is available" >&2
      exit 4
    fi
    sleep "${WAIT_POLL_SECONDS}"
  done
fi

ckpt="$(latest_checkpoint "${source_run}")"
if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
  echo "[moe-lora-3h] no checkpoint found under ${source_run}/checkpoints" >&2
  exit 4
fi

RUN_ID="${RUN_ID:-abc_moe95k_lora_${VARIANT}_3h_bs${BATCH_SIZE}_${TS}}"

echo "[moe-lora-3h] continuing from ${ckpt}"
echo "[moe-lora-3h] run_id=${RUN_ID}"
echo "[moe-lora-3h] max_train_steps=${MAX_TRAIN_STEPS} save_interval=${SAVE_INTERVAL} batch_size=${BATCH_SIZE}"
echo "[moe-lora-3h] lr qwen_lora=${QWEN_LORA_LR} action=${ACTION_LR}"

export VARIANT
export PRETRAINED_CHECKPOINT="${ckpt}"
export RUN_ID
export MAX_TRAIN_STEPS
export SAVE_INTERVAL
export BATCH_SIZE
export DATALOADER_NUM_WORKERS
export DATALOADER_PREFETCH_FACTOR
export QWEN_LORA_LR
export ACTION_LR
export RUN_ROOT_DIR

exec bash "${SCRIPT_DIR}/run_train_moe95k_lora_h200.sh"
