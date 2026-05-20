#!/usr/bin/env bash
set -euo pipefail

# Evaluate the latest WMH MoE95k+LoRA checkpoint on CALVIN D.
#
# Usage:
#   VARIANT=aug    bash examples/calvin_autoresearch/scripts/run_eval_moe_lora_latest_n300.sh
#   VARIANT=mirror bash examples/calvin_autoresearch/scripts/run_eval_moe_lora_latest_n300.sh
#
# The checkpoint is resolved at launch time from the newest complete
# steps_*_pytorch_model.pt under the matching 3h run.

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
RUN_DIR="${RUN_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
REPORT_DIR="${REPORT_DIR:-${SHARED_ROOT}/members/${MEMBER}/reports}"

VARIANT="${VARIANT:-aug}"
case "${VARIANT}" in
  aug)
    RUN_PATTERN="${RUN_PATTERN:-abc_moe95k_lora_aug_3h_bs96_*}"
    RUN_TAG="${RUN_TAG:-moe95k_lora_aug_latest}"
    ;;
  mirror)
    RUN_PATTERN="${RUN_PATTERN:-abc_moe95k_lora_mirror_3h_bs96_*}"
    RUN_TAG="${RUN_TAG:-moe95k_lora_mirror_latest}"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected aug or mirror" >&2
    exit 2
    ;;
esac

latest_checkpoint() {
  local run_pattern="$1"
  find "${RUN_DIR}" -maxdepth 1 -type d -name "${run_pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2- \
    | while read -r run; do
        [[ -n "${run}" ]] || exit 0
        find "${run}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%T@ %p\n' 2>/dev/null \
          | sort -nr \
          | head -1 \
          | cut -d' ' -f2-
      done
}

CKPT="${CKPT:-$(latest_checkpoint "${RUN_PATTERN}")}"
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "No complete checkpoint found for pattern ${RUN_PATTERN} under ${RUN_DIR}" >&2
  exit 3
fi

TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-300}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-8300}"
UNNORM_KEY="${UNNORM_KEY:-new_embodiment}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-0}"
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-14400}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
DEBUG="${DEBUG:-0}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${REPORT_DIR}/eval_${RUN_TAG}_d_n${TOTAL_SEQUENCES}_${TS}}"

echo "[eval-moe-lora] variant=${VARIANT}"
echo "[eval-moe-lora] ckpt=${CKPT}"
echo "[eval-moe-lora] unnorm_key=${UNNORM_KEY}"
echo "[eval-moe-lora] send_state=${CALVIN_SEND_STATE}"
echo "[eval-moe-lora] gpu_ids=${GPU_IDS}"
echo "[eval-moe-lora] workers_per_gpu=${WORKERS_PER_GPU}"
echo "[eval-moe-lora] eval_log_dir=${EVAL_LOG_DIR}"

CKPT="${CKPT}" \
TOTAL_SEQUENCES="${TOTAL_SEQUENCES}" \
GPU_IDS="${GPU_IDS}" \
WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
BASE_PORT="${BASE_PORT}" \
UNNORM_KEY="${UNNORM_KEY}" \
CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT}" \
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" \
DEBUG="${DEBUG}" \
EVAL_LOG_DIR="${EVAL_LOG_DIR}" \
POLICY_SERVER_SCRIPT="examples/calvin_autoresearch/scripts/run_policy_server_moe_lora.sh" \
bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
