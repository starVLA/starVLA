#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# === Thin shim: mirrors the cosmopredict2_gr00t_multiview wrapper, but    ===
# ===   * points at the d_style fine-tune yaml                              ===
# ===   * continues from the steps_10000 cosmopredict2 ckpt by default      ===
# ===   * tags run_id so it does NOT collide with the original from-scratch ===
# === All other knobs (NUM_PROCESSES, BASE_VLM, ...) inherit from base bash ===
export CONFIG_YAML="${CONFIG_YAML:-examples/calvin/train_files/starvla_train_calvin_abc_qwen35_2b_cosmopredict2_gr00t_multiview_d_style_ft.yaml}"
export FRAMEWORK_NAME="${FRAMEWORK_NAME:-QwenCosmosGR00T}"
export DATA_MIX="${DATA_MIX:-calvin_task_ABC_D_multiview}"
export RUN_ID_PREFIX="${RUN_ID_PREFIX:-qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview_d_style_ft}"
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/inspire/qb-ilm2/project/26summer-camp-10/public/ten/qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview_20260519_112310/checkpoints/steps_10000_pytorch_model.pt}"

# Keep cosmos backbone frozen even though base bash defaults to empty:
# the base script forwards FREEZE_MODULES verbatim to --trainer.freeze_modules,
# and an empty string there would override the yaml's "cosmos_backbone".
export FREEZE_MODULES="${FREEZE_MODULES:-cosmos_backbone}"

exec "${SCRIPT_DIR}/run_calvin_abc_qwen35_2b_gr00t_multiview.sh" "$@"
