#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_aug.yaml}" \
DATA_MIX="${DATA_MIX:-calvin_abc_train_state_v3.0}" \
RUN_ID="${RUN_ID:-abc_pretrain_qwen3vl_gr00t_state8_connector_balanced_aug_h200}" \
bash "${SCRIPT_DIR}/run_train_abc_pretrain_h200.sh"
