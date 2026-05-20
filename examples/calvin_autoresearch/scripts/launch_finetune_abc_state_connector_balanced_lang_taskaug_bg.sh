#!/usr/bin/env bash
set -euo pipefail

# Background launcher for ABC-only balanced+language+task-aware image augmentation
# post-training. This wrapper exists to avoid brittle multi-line nohup commands.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/members/${MEMBER}/logs}"
mkdir -p "${LOG_DIR}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-abc_state8_connector_balanced_lang_taskaug_ft${MAX_TRAIN_STEPS}_${TS}}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_ID}.log}"

export RUN_ID
export MAX_TRAIN_STEPS
export SAVE_INTERVAL
export NUM_PROCESSES="${NUM_PROCESSES:-4}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export LOG_GRAD_NORMS="${LOG_GRAD_NORMS:-1}"
export SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-1}"

echo "[launch-finetune-bg] run_id=${RUN_ID}"
echo "[launch-finetune-bg] log=${LOG_PATH}"
echo "[launch-finetune-bg] gpu_ids=${GPU_IDS} num_processes=${NUM_PROCESSES} batch_size=${BATCH_SIZE} steps=${MAX_TRAIN_STEPS}"

nohup bash "${SCRIPT_DIR}/run_finetune_abc_state_connector_balanced_lang_taskaug_h200.sh" \
  > "${LOG_PATH}" 2>&1 &

echo "[launch-finetune-bg] pid=$!"
echo "[launch-finetune-bg] tail -f ${LOG_PATH}"
