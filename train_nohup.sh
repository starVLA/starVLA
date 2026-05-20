#!/usr/bin/env bash
set -euo pipefail

# One-command WMH training launcher.
#
# Default behavior:
#   - nohup-launches ./wmh night-race8
#   - starts two ABC-only 4-GPU branches on one 8-GPU node:
#       1. hard-sampler-v2 non-mirror augmentation
#       2. hard-sampler-v2 + left/right mirror augmentation
#
# Common overrides:
#   MAX_TRAIN_STEPS=6000 ./train_nohup.sh
#   MODE=mirror-night8 MAX_TRAIN_STEPS=10000 ./train_nohup.sh
#   MODE=lora-ft8 MAX_TRAIN_STEPS=2000 ./train_nohup.sh
#   MODE=lora-after-race LORA_MAX_TRAIN_STEPS=2000 ./train_nohup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/starvla_env.sh}"

if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${ENV_FILE}"
fi

cd "${SCRIPT_DIR}"

MODE="${MODE:-night-race8}"
MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/members/${MEMBER}/logs}"
TS="${TS:-$(date +%m%d_%H%M%S)}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-8000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}"
if [[ -z "${BATCH_SIZE+x}" ]]; then
  case "${MODE}" in
    lora-*) BATCH_SIZE=64 ;;
    *) BATCH_SIZE=96 ;;
  esac
fi
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
CONNECTOR_LR="${CONNECTOR_LR:-3.0e-05}"
ACTION_LR="${ACTION_LR:-1.0e-04}"
MIRROR_PROBABILITY="${MIRROR_PROBABILITY:-0.25}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_LATEST="${CONTINUE_LATEST:-0}"

RUN_TAG="${RUN_TAG:-nohup_${MODE}_${MAX_TRAIN_STEPS}_${TS}}"
LOG_PATH="${LOG_PATH:-${LOG_DIR}/${RUN_TAG}.log}"

mkdir -p "${LOG_DIR}" wmh_links

case "${MODE}" in
  night-race8|mirror-night8|mirror-ft8|aug-ft8|lora-ft8|lora-probe8|lora-smoke8|lora-after-race)
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    echo "Allowed: night-race8, mirror-night8, mirror-ft8, aug-ft8, lora-ft8, lora-probe8, lora-smoke8, lora-after-race" >&2
    exit 2
    ;;
esac

echo "[train-nohup] mode=${MODE}"
echo "[train-nohup] steps=${MAX_TRAIN_STEPS} save_interval=${SAVE_INTERVAL}"
echo "[train-nohup] batch_size=${BATCH_SIZE}"
echo "[train-nohup] connector_lr=${CONNECTOR_LR} action_lr=${ACTION_LR}"
echo "[train-nohup] mirror_probability=${MIRROR_PROBABILITY}"
echo "[train-nohup] launcher_log=${LOG_PATH}"
echo "[train-nohup] dry_run=${DRY_RUN}"
echo "[train-nohup] continue_latest=${CONTINUE_LATEST}"

nohup env \
  DRY_RUN="${DRY_RUN}" \
  CONTINUE_LATEST="${CONTINUE_LATEST}" \
  LOG_DIR="${LOG_DIR}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
  DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR}" \
  CONNECTOR_LR="${CONNECTOR_LR}" \
  ACTION_LR="${ACTION_LR}" \
  MIRROR_PROBABILITY="${MIRROR_PROBABILITY}" \
  LORA_MAX_TRAIN_STEPS="${LORA_MAX_TRAIN_STEPS:-2000}" \
  LORA_SAVE_INTERVAL="${LORA_SAVE_INTERVAL:-1000}" \
  LORA_SOURCE="${LORA_SOURCE:-base}" \
  ./wmh "${MODE}" > "${LOG_PATH}" 2>&1 &

PID="$!"
ln -sfn "${LOG_PATH}" wmh_links/latest_train_nohup.log

echo "[train-nohup] launcher_pid=${PID}"
echo "[train-nohup] tail launcher: tail -f wmh_links/latest_train_nohup.log"
echo "[train-nohup] tail aug branch: ./wmh tail-race-aug"
echo "[train-nohup] tail mirror branch: ./wmh tail-race-mirror"
