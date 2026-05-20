#!/usr/bin/env bash
set -euo pipefail

# Short wrapper for the 8k state8+connector D n100 eval. This avoids long
# pasted commands where line wrapping can split the script path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_N10="${RUN_N10:-0}"
export RUN_N100="${RUN_N100:-1}"
export N100_SEQUENCES="${N100_SEQUENCES:-100}"
export N100_GPU_IDS="${N100_GPU_IDS:-0,1,2,3}"
export N100_WORKERS_PER_GPU="${N100_WORKERS_PER_GPU:-1}"
export N100_BASE_PORT="${N100_BASE_PORT:-6600}"

exec bash "${SCRIPT_DIR}/run_eval_8k_n10_n100.sh"
