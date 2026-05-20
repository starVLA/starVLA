#!/usr/bin/env bash
set -euo pipefail

# Evaluate the WMH/GTY QwenGR00T_MoE_Adaptive checkpoint on CALVIN D.
# Defaults to the completed WMH adaptive run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

ADAPTIVE_RUN_DIR="${ADAPTIVE_RUN_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_augmented_moe_adaptive_WMH_bs64_s15k_0519_121249}"
CKPT="${CKPT:-${ADAPTIVE_RUN_DIR}/checkpoints/steps_15000_pytorch_model.pt}"
TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-300}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
BASE_PORT="${BASE_PORT:-8000}"
UNNORM_KEY="${UNNORM_KEY:-new_embodiment}"
CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-0}"
RUN_TAG="${RUN_TAG:-moe_adaptive_s15k}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/eval_${RUN_TAG}_d_n${TOTAL_SEQUENCES}_${TS}}"

echo "[eval-moe-adaptive] ckpt=${CKPT}"
echo "[eval-moe-adaptive] unnorm_key=${UNNORM_KEY}"
echo "[eval-moe-adaptive] send_state=${CALVIN_SEND_STATE}"
echo "[eval-moe-adaptive] gpu_ids=${GPU_IDS}"
echo "[eval-moe-adaptive] workers_per_gpu=${WORKERS_PER_GPU}"
echo "[eval-moe-adaptive] eval_log_dir=${EVAL_LOG_DIR}"

CKPT="${CKPT}" \
TOTAL_SEQUENCES="${TOTAL_SEQUENCES}" \
GPU_IDS="${GPU_IDS}" \
WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
BASE_PORT="${BASE_PORT}" \
UNNORM_KEY="${UNNORM_KEY}" \
CALVIN_SEND_STATE="${CALVIN_SEND_STATE}" \
EVAL_LOG_DIR="${EVAL_LOG_DIR}" \
POLICY_SERVER_SCRIPT="examples/calvin_autoresearch/scripts/run_policy_server_moe_adaptive.sh" \
bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
