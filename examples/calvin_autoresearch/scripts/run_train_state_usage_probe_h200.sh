#!/usr/bin/env bash
set -euo pipefail

# Short ABC-only training probes for state/connector attribution.
#
# VARIANT=state_only:
#   include_state=true, no vl_connector. Trains action_model, including the
#   GR00T state_encoder ("state_projector"). Checks whether proprio alone can
#   improve over the old no-state baseline.
#
# VARIANT=connector_no_state:
#   include_state=false, vl_connector=true. Trains connector + no-state action
#   head. Checks whether the connector has independent value.
#
# VARIANT=connector_only_no_state:
#   include_state=false, vl_connector=true, reloads the WMH no-state ABC baseline
#   and freezes Qwen + action_model. Trains only the new connector.
#
# VARIANT=state_connector:
#   include_state=true, vl_connector=true. Current 5.1+5.2 recipe.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

VARIANT="${VARIANT:-state_connector}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
TS="${TS:-$(date +%m%d_%H%M%S)}"

case "${VARIANT}" in
  state_only)
    CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8.yaml}"
    DATA_MIX="${DATA_MIX:-calvin_abc_train_state_v3.0}"
    RUN_ID="${RUN_ID:-abc_state_usage_state_only_probe${MAX_TRAIN_STEPS}_${TS}}"
    ;;
  connector_no_state)
    CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_connector_nostate.yaml}"
    DATA_MIX="${DATA_MIX:-calvin_abc_train_v3.0}"
    RUN_ID="${RUN_ID:-abc_state_usage_connector_nostate_probe${MAX_TRAIN_STEPS}_${TS}}"
    ;;
  connector_only_no_state)
    CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_connector_nostate.yaml}"
    DATA_MIX="${DATA_MIX:-calvin_abc_train_v3.0}"
    BASELINE_CKPT="${BASELINE_CKPT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/WMH/runs/abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437/checkpoints/steps_60000_pytorch_model.pt}"
    if [[ ! -f "${BASELINE_CKPT}" ]]; then
      echo "BASELINE_CKPT not found: ${BASELINE_CKPT}" >&2
      exit 3
    fi
    RUN_ID="${RUN_ID:-abc_state_usage_connector_only_nostate_probe${MAX_TRAIN_STEPS}_${TS}}"
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-} --trainer.pretrained_checkpoint ${BASELINE_CKPT} --trainer.freeze_modules qwen_vl_interface,action_model"
    ;;
  state_connector)
    CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector.yaml}"
    DATA_MIX="${DATA_MIX:-calvin_abc_train_state_v3.0}"
    RUN_ID="${RUN_ID:-abc_state_usage_state_connector_probe${MAX_TRAIN_STEPS}_${TS}}"
    ;;
  *)
    echo "Unsupported VARIANT=${VARIANT}; expected state_only, connector_no_state, connector_only_no_state, or state_connector." >&2
    exit 2
    ;;
esac

echo "[state-usage-probe] variant=${VARIANT}"
echo "[state-usage-probe] config=${CONFIG_YAML}"
echo "[state-usage-probe] run_id=${RUN_ID}"
echo "[state-usage-probe] steps=${MAX_TRAIN_STEPS}"

CONFIG_YAML="${CONFIG_YAML}" \
DATA_MIX="${DATA_MIX}" \
RUN_ID="${RUN_ID}" \
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
LOG_GRAD_NORMS=1 \
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-1}" \
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}" \
bash "${SCRIPT_DIR}/run_train_abc_pretrain_h200.sh"
