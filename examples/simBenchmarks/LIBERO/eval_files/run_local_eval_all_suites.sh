#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint.pt> [gpu_id] [base_port]"
  echo "Example smoke: MAX_TASKS=1 NUM_TRIALS_PER_TASK=1 $0 playground/Checkpoints/run/checkpoints/steps_32000_pytorch_model.pt 2 18080"
  echo "Example full:  MAX_TASKS=-1 NUM_TRIALS_PER_TASK=50 $0 playground/Checkpoints/run/checkpoints/steps_32000_pytorch_model.pt 2 18080"
  exit 2
fi

CKPT="$1"
GPU_ID="${2:-2}"
BASE_PORT="${3:-18080}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SUITES=(libero_spatial libero_object libero_goal libero_10)

for idx in "${!TASK_SUITES[@]}"; do
  suite="${TASK_SUITES[$idx]}"
  port=$((BASE_PORT + idx))
  echo "========== ${suite} on GPU ${GPU_ID}, port ${port} =========="
  bash "${SCRIPT_DIR}/run_local_eval_once.sh" "${CKPT}" "${suite}" "${GPU_ID}" "${port}"
done

echo "========== all LIBERO suites completed =========="
