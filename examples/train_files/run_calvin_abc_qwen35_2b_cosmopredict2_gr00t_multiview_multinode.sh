#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_abc_qwen35_2b_cosmopredict2_gr00t_multiview.yaml}"
export FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenCosmosGR00T}"
export DATA_MIX="${DATA_MIX:-calvin_task_ABC_D_multiview}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview}"

exec "${SCRIPT_DIR}/run_calvin_abc_qwen35_2b_gr00t_multiview_multinode.sh" "$@"
