#!/usr/bin/env bash
set -euo pipefail

# One-click formal 8-GPU CALVIN ABC->D eval for WMH state8+connector steps_8000.
# Defaults:
#   checkpoint: WMH abc_state8_connector_8h200_bs96_8k_0519_083200 steps_8000
#   sequences: 1000
#   GPUs:      0,1,2,3,4,5,6,7
#   state:     enabled, because this checkpoint has model_state_dim=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CKPT="${CKPT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt}"
export CALVIN_SEND_STATE="${CALVIN_SEND_STATE:-1}"
export TOTAL_SEQUENCES="${TOTAL_SEQUENCES:-1000}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
export BASE_PORT="${BASE_PORT:-6700}"
export RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-28800}"
export SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
export RUN_TAG="${RUN_TAG:-8k_state8_connector_steps8000}"

exec bash "${SCRIPT_DIR}/run_eval_ckpt_n1000_8gpu.sh"
