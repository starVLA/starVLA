#!/usr/bin/env bash
set -euo pipefail

# Run CALVIN D closed-loop eval under state sanity modes:
#   normal  - send the real 8-D robot state
#   zero    - zero the normalized state before policy inference
#   shuffle - send temporally mismatched state in the eval client; the server
#             also supports batch shuffling for batched callers.
#
# This is intended for state-aware checkpoints only. It never trains on D data.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

CKPT="${CKPT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt}"
TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-100}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-7000}"
MODES="${MODES:-normal zero shuffle}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
REPORT_ROOT="${REPORT_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/reports/state_ablation_${TS}}"

mkdir -p "${REPORT_ROOT}"
echo "[state-ablation] ckpt=${CKPT}"
echo "[state-ablation] report_root=${REPORT_ROOT}"
echo "[state-ablation] modes=${MODES}"

mode_idx=0
for mode in ${MODES}; do
  mode_dir="${REPORT_ROOT}/${mode}_n${TOTAL_SEQUENCES}"
  port=$(( BASE_PORT + mode_idx * 100 ))
  echo "[state-ablation] running mode=${mode} port_base=${port} dir=${mode_dir}"
  CALVIN_SEND_STATE=1 \
  CALVIN_STATE_MODE="${mode}" \
  STARVLA_STATE_SANITY_MODE="${mode}" \
  CKPT="${CKPT}" \
  TOTAL_SEQUENCES="${TOTAL_SEQUENCES}" \
  GPU_IDS="${GPU_IDS}" \
  WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
  BASE_PORT="${port}" \
  EVAL_LOG_DIR="${mode_dir}" \
  DEBUG=0 \
  bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh

  if [[ -f "${mode_dir}/metrics.json" ]]; then
    PYTHONDONTWRITEBYTECODE=1 python examples/calvin_autoresearch/scripts/summarize_eval_metrics.py "${mode_dir}/metrics.json" \
      > "${mode_dir}/summary.txt" || true
  fi
  mode_idx=$(( mode_idx + 1 ))
done

echo "[state-ablation] done. Compare metrics under ${REPORT_ROOT}/{normal,zero,shuffle}_n${TOTAL_SEQUENCES}/metrics.json"
