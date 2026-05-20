#!/usr/bin/env bash
set -euo pipefail

# One-command legal CALVIN ABC training entrypoint for the final image.
# It uses only the allowed Qwen base model + CALVIN ABC data and refuses
# upstream action-trained checkpoints in the underlying launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

TS="${TS:-$(date +%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-abc_pretrain_qwen3vl_gr00t_headonly_h200_${TS}}"
LOG_DIR="${LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/logs}"
mkdir -p "${LOG_DIR}"

export STRICT_ASSETS="${STRICT_ASSETS:-1}"
export GPU_IDS="${GPU_IDS:-0,1,2}"
export NUM_PROCESSES="${NUM_PROCESSES:-3}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-60000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
export DRY_RUN="${DRY_RUN:-0}"
export RUN_ID

LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
echo "[oneclick-train] RUN_ID=${RUN_ID}"
echo "[oneclick-train] LOG=${LOG_PATH}"

exec bash examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh 2>&1 | tee "${LOG_PATH}"
