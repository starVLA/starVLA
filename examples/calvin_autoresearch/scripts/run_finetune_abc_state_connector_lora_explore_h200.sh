#!/usr/bin/env bash
set -euo pipefail

# ABC-only LoRA exploration branch.
#
# This branch keeps the Qwen backbone frozen except for dependency-free LoRA
# adapters injected into selected late Qwen attention projections. It is meant
# for exploration credit and controlled ablation, not as the default baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-abc_lora_explore_ft${MAX_TRAIN_STEPS:-2000}_${TS}}"

QWEN_LORA_RANK="${QWEN_LORA_RANK:-8}"
QWEN_LORA_ALPHA="${QWEN_LORA_ALPHA:-16}"
QWEN_LORA_DROPOUT="${QWEN_LORA_DROPOUT:-0.05}"
QWEN_LORA_LAST_N_LAYERS="${QWEN_LORA_LAST_N_LAYERS:-4}"
QWEN_LORA_TARGET_MODULES="${QWEN_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
QWEN_LORA_LR="${QWEN_LORA_LR:-5.0e-06}"
CONNECTOR_LR="${CONNECTOR_LR:-3.0e-05}"
ACTION_LR="${ACTION_LR:-1.0e-04}"

HARD_V2_ARGS=(
  --datasets.vla_data.sampler.oversample_tasks.turn_on_lightbulb 5.0
  --datasets.vla_data.sampler.oversample_tasks.stack_block 4.0
  --datasets.vla_data.sampler.oversample_tasks.move_slider_right 3.0
  --datasets.vla_data.sampler.oversample_tasks.push_red_block_left 2.0
  --datasets.vla_data.sampler.oversample_tasks.push_blue_block_left 2.0
  --datasets.vla_data.sampler.oversample_tasks.push_pink_block_left 2.0
)

LORA_ARGS=(
  --framework.qwen_lora.enabled true
  --framework.qwen_lora.rank "${QWEN_LORA_RANK}"
  --framework.qwen_lora.alpha "${QWEN_LORA_ALPHA}"
  --framework.qwen_lora.dropout "${QWEN_LORA_DROPOUT}"
  --framework.qwen_lora.last_n_layers "${QWEN_LORA_LAST_N_LAYERS}"
  --framework.qwen_lora.target_modules "${QWEN_LORA_TARGET_MODULES}"
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LORA_LR}"
  --trainer.learning_rate.vl_connector "${CONNECTOR_LR}"
  --trainer.learning_rate.action_model "${ACTION_LR}"
)

join_args() {
  local out="" arg
  for arg in "$@"; do
    out+="${arg} "
  done
  printf '%s' "${out% }"
}

USER_EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
export EXTRA_TRAIN_ARGS="$(join_args "${LORA_ARGS[@]}" "${HARD_V2_ARGS[@]}") ${USER_EXTRA_TRAIN_ARGS}"
export RUN_ID
export CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug.yaml}"
export NUM_PROCESSES="${NUM_PROCESSES:-8}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export LOG_GRAD_NORMS="${LOG_GRAD_NORMS:-1}"
export SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-1}"

cat <<EOF
[lora-explore] run_id=${RUN_ID}
[lora-explore] qwen_lora rank=${QWEN_LORA_RANK} alpha=${QWEN_LORA_ALPHA} dropout=${QWEN_LORA_DROPOUT}
[lora-explore] targets=${QWEN_LORA_TARGET_MODULES} last_n_layers=${QWEN_LORA_LAST_N_LAYERS}
[lora-explore] lr qwen_lora=${QWEN_LORA_LR} connector=${CONNECTOR_LR} action=${ACTION_LR}
[lora-explore] train_data=CALVIN ABC only
EOF

bash "${SCRIPT_DIR}/run_finetune_abc_state_connector_balanced_lang_taskaug_h200.sh"
