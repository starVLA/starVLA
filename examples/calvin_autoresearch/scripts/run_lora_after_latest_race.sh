#!/usr/bin/env bash
set -euo pipefail

# Wait for the latest night-race branches to finish, then launch a LoRA
# exploration finetune. This lets the current 8-GPU race fully use the node
# first and automatically schedules LoRA afterwards.

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
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
RACE_DIR="${RACE_DIR:-}"

POLL_INTERVAL="${POLL_INTERVAL:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
LORA_MAX_TRAIN_STEPS="${LORA_MAX_TRAIN_STEPS:-2000}"
LORA_SAVE_INTERVAL="${LORA_SAVE_INTERVAL:-1000}"
LORA_BATCH_SIZE="${LORA_BATCH_SIZE:-64}"
LORA_NUM_PROCESSES="${LORA_NUM_PROCESSES:-8}"
LORA_GPU_IDS="${LORA_GPU_IDS:-0,1,2,3,4,5,6,7}"
LORA_SOURCE="${LORA_SOURCE:-base}"
DRY_RUN="${DRY_RUN:-0}"

latest_race_dir() {
  find "${LOG_DIR}" -maxdepth 1 -type d -name 'night_race_hardv2_*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

latest_log_in_race() {
  local pattern="$1"
  find "${RACE_DIR}" -maxdepth 1 -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

run_id_from_log() {
  basename "$1" .log
}

latest_ckpt_for_run() {
  local run_id="$1"
  find "${RUN_ROOT_DIR}/${run_id}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

log_failed() {
  local log="$1"
  grep -Eq 'Traceback|ChildFailed|out of memory|CUDA error|RuntimeError|Killed|Exception' "${log}" 2>/dev/null
}

log_done() {
  local log="$1"
  grep -Eq 'Training complete|skip_final_save enabled|and that.s all|Checkpoint saved' "${log}" 2>/dev/null
}

if [[ -z "${RACE_DIR}" ]]; then
  RACE_DIR="$(latest_race_dir)"
fi
if [[ -z "${RACE_DIR}" || ! -d "${RACE_DIR}" ]]; then
  echo "[lora-after-race] no night_race_hardv2_* directory found under ${LOG_DIR}" >&2
  exit 2
fi

AUG_LOG="$(latest_log_in_race 'abc_aug_hardv2_*.log')"
MIRROR_LOG="$(latest_log_in_race 'abc_mirror_hardv2_*.log')"
if [[ -z "${AUG_LOG}" || -z "${MIRROR_LOG}" ]]; then
  echo "[lora-after-race] missing branch logs in ${RACE_DIR}" >&2
  echo "[lora-after-race] aug_log=${AUG_LOG:-<missing>}" >&2
  echo "[lora-after-race] mirror_log=${MIRROR_LOG:-<missing>}" >&2
  exit 2
fi

AUG_RUN_ID="$(run_id_from_log "${AUG_LOG}")"
MIRROR_RUN_ID="$(run_id_from_log "${MIRROR_LOG}")"

cat <<EOF
[lora-after-race] race_dir=${RACE_DIR}
[lora-after-race] aug_run=${AUG_RUN_ID}
[lora-after-race] mirror_run=${MIRROR_RUN_ID}
[lora-after-race] poll_interval=${POLL_INTERVAL}s max_wait=${MAX_WAIT_SECONDS}s
[lora-after-race] lora_source=${LORA_SOURCE}
[lora-after-race] lora_steps=${LORA_MAX_TRAIN_STEPS} save_interval=${LORA_SAVE_INTERVAL}
EOF

elapsed=0
while true; do
  if log_failed "${AUG_LOG}"; then
    echo "[lora-after-race] aug branch appears failed; refusing to launch LoRA" >&2
    tail -80 "${AUG_LOG}" >&2 || true
    exit 3
  fi
  if log_failed "${MIRROR_LOG}"; then
    echo "[lora-after-race] mirror branch appears failed; refusing to launch LoRA" >&2
    tail -80 "${MIRROR_LOG}" >&2 || true
    exit 3
  fi

  AUG_CKPT="$(latest_ckpt_for_run "${AUG_RUN_ID}")"
  MIRROR_CKPT="$(latest_ckpt_for_run "${MIRROR_RUN_ID}")"

  if [[ -n "${AUG_CKPT}" && -n "${MIRROR_CKPT}" ]] && log_done "${AUG_LOG}" && log_done "${MIRROR_LOG}"; then
    echo "[lora-after-race] both branches finished"
    echo "[lora-after-race] aug_ckpt=${AUG_CKPT}"
    echo "[lora-after-race] mirror_ckpt=${MIRROR_CKPT}"
    break
  fi

  if (( elapsed >= MAX_WAIT_SECONDS )); then
    echo "[lora-after-race] timed out after ${elapsed}s waiting for race completion" >&2
    exit 4
  fi

  echo "[lora-after-race] waiting elapsed=${elapsed}s aug_ckpt=${AUG_CKPT:-no} mirror_ckpt=${MIRROR_CKPT:-no}"
  sleep "${POLL_INTERVAL}"
  elapsed=$((elapsed + POLL_INTERVAL))
done

case "${LORA_SOURCE}" in
  base)
    unset PRETRAINED_CHECKPOINT
    ;;
  aug)
    export PRETRAINED_CHECKPOINT="${AUG_CKPT}"
    ;;
  mirror)
    export PRETRAINED_CHECKPOINT="${MIRROR_CKPT}"
    ;;
  *)
    echo "[lora-after-race] invalid LORA_SOURCE=${LORA_SOURCE}; expected base, aug, or mirror" >&2
    exit 2
    ;;
esac

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[lora-after-race] DRY_RUN would launch:"
  printf '  MAX_TRAIN_STEPS=%q SAVE_INTERVAL=%q NUM_PROCESSES=%q GPU_IDS=%q BATCH_SIZE=%q ./wmh lora-ft8\n' \
    "${LORA_MAX_TRAIN_STEPS}" "${LORA_SAVE_INTERVAL}" "${LORA_NUM_PROCESSES}" "${LORA_GPU_IDS}" "${LORA_BATCH_SIZE}"
  exit 0
fi

echo "[lora-after-race] launching LoRA exploration"
MAX_TRAIN_STEPS="${LORA_MAX_TRAIN_STEPS}" \
SAVE_INTERVAL="${LORA_SAVE_INTERVAL}" \
NUM_PROCESSES="${LORA_NUM_PROCESSES}" \
GPU_IDS="${LORA_GPU_IDS}" \
BATCH_SIZE="${LORA_BATCH_SIZE}" \
  ./wmh lora-ft8

echo "[lora-after-race] LoRA launcher submitted"
