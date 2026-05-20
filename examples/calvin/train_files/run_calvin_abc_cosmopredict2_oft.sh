#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_abc_cosmopredict2_oft.yaml}"
export FRAMEWORK_NAME="${FRAMEWORK_NAME:-CosmoPredict2OFT}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-cosmopredict2_oft_calvin_abc}"
exec "${SCRIPT_DIR}/run_calvin_abc_cosmopredict2_gr00t.sh" "$@"
