#!/usr/bin/env bash
set -euo pipefail

# Continue the no-mirror MoE95k+LoRA branch with:
#   1. action_horizon=4 for tighter closed-loop replanning;
#   2. progress-biased within-trajectory sampling for p2-p5-like states;
#   3. extra task balancing on long-horizon failure-heavy tasks.
#
# This is ABC-only.  It resolves the latest no-mirror WMH MoE95k+LoRA checkpoint
# by default and starts a new run, leaving the source checkpoint untouched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${STARVLA_ROOT}"

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/members/${MEMBER}/logs}"

# Prefer the current no-mirror 3h continuation branch; allow explicit SOURCE_CKPT
# when the user wants to pin a checkpoint.
SOURCE_RUN_PATTERNS="${SOURCE_RUN_PATTERNS:-abc_moe95k_lora_aug_3h_bs*_* abc_moe95k_lora_aug_2500_* abc_moe95k_lora_aug_5k_*}"
SOURCE_CKPT="${SOURCE_CKPT:-}"

TS="${TS:-$(date +%m%d_%H%M%S)}"
ACTION_HORIZON="${ACTION_HORIZON:-4}"
FUTURE_ACTION_WINDOW_SIZE="$((ACTION_HORIZON - 1))"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-6000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
BATCH_SIZE="${BATCH_SIZE:-96}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
QWEN_LORA_LR="${QWEN_LORA_LR:-2.0e-06}"
ACTION_LR="${ACTION_LR:-2.0e-05}"
RUN_ID="${RUN_ID:-abc_moe95k_lora_aug_p2p5_h${ACTION_HORIZON}_bs${BATCH_SIZE}_${MAX_TRAIN_STEPS}_${TS}}"

latest_checkpoint_from_patterns() {
  local pattern run
  for pattern in ${SOURCE_RUN_PATTERNS}; do
    find "${RUN_ROOT_DIR}" -maxdepth 1 -type d -name "${pattern}" -printf '%p\n' 2>/dev/null || true
  done \
    | sort -u \
    | while read -r run; do
        [[ -n "${run}" && -d "${run}/checkpoints" ]] || continue
        find "${run}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%T@ %p\n' 2>/dev/null
      done \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

mkdir -p "${LOG_DIR}"

if [[ -z "${SOURCE_CKPT}" ]]; then
  SOURCE_CKPT="$(latest_checkpoint_from_patterns)"
fi

if [[ -z "${SOURCE_CKPT}" || ! -f "${SOURCE_CKPT}" ]]; then
  echo "[p2p5-h4] no source checkpoint found." >&2
  echo "[p2p5-h4] searched RUN_ROOT_DIR=${RUN_ROOT_DIR}" >&2
  echo "[p2p5-h4] searched SOURCE_RUN_PATTERNS=${SOURCE_RUN_PATTERNS}" >&2
  echo "[p2p5-h4] set SOURCE_CKPT=/path/to/steps_xxx_pytorch_model.pt to override." >&2
  exit 3
fi

read -r -d '' P2P5_ARGS <<EOF || true
--framework.action_model.action_horizon ${ACTION_HORIZON}
--framework.action_model.future_action_window_size ${FUTURE_ACTION_WINDOW_SIZE}
--datasets.vla_data.action_horizon ${ACTION_HORIZON}
--datasets.vla_data.step_sampling.enabled true
--datasets.vla_data.step_sampling.type progress_curriculum
--datasets.vla_data.step_sampling.early_end 0.30
--datasets.vla_data.step_sampling.middle_start 0.30
--datasets.vla_data.step_sampling.middle_end 0.75
--datasets.vla_data.step_sampling.late_start 0.75
--datasets.vla_data.step_sampling.late_end 0.95
--datasets.vla_data.step_sampling.early_weight 0.45
--datasets.vla_data.step_sampling.middle_weight 1.80
--datasets.vla_data.step_sampling.late_weight 1.30
--datasets.vla_data.sampler.oversample_tasks.open_drawer 2.0
--datasets.vla_data.sampler.oversample_tasks.close_drawer 3.0
--datasets.vla_data.sampler.oversample_tasks.move_slider_left 5.0
--datasets.vla_data.sampler.oversample_tasks.move_slider_right 5.0
--datasets.vla_data.sampler.oversample_tasks.turn_off_lightbulb 5.0
--datasets.vla_data.sampler.oversample_tasks.turn_on_lightbulb 5.0
--datasets.vla_data.sampler.oversample_tasks.turn_off_led 3.0
--datasets.vla_data.sampler.oversample_tasks.turn_on_led 3.0
--datasets.vla_data.sampler.oversample_tasks.stack_block 4.0
--datasets.vla_data.sampler.oversample_tasks.unstack_block 3.0
--datasets.vla_data.sampler.oversample_tasks.place_in_drawer 2.5
--datasets.vla_data.sampler.oversample_tasks.place_in_slider 3.0
--datasets.vla_data.sampler.oversample_tasks.push_into_drawer 2.0
--datasets.vla_data.sampler.oversample_tasks.lift_red_block_slider 3.0
--datasets.vla_data.sampler.oversample_tasks.lift_blue_block_slider 3.0
--datasets.vla_data.sampler.oversample_tasks.lift_pink_block_slider 3.0
EOF

USER_EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
EXTRA_TRAIN_ARGS="${P2P5_ARGS} ${USER_EXTRA_TRAIN_ARGS}"

cat <<EOF
[p2p5-h4] source_ckpt=${SOURCE_CKPT}
[p2p5-h4] run_id=${RUN_ID}
[p2p5-h4] action_horizon=${ACTION_HORIZON}
[p2p5-h4] max_train_steps=${MAX_TRAIN_STEPS} save_interval=${SAVE_INTERVAL}
[p2p5-h4] batch_size=${BATCH_SIZE} qwen_lora_lr=${QWEN_LORA_LR} action_lr=${ACTION_LR}
[p2p5-h4] step_sampling=early[0,.30]x0.45 middle[.30,.75]x1.80 late[.75,.95]x1.30
EOF

export VARIANT=aug
export PRETRAINED_CHECKPOINT="${SOURCE_CKPT}"
export RUN_ID
export MAX_TRAIN_STEPS
export SAVE_INTERVAL
export BATCH_SIZE
export DATALOADER_NUM_WORKERS
export DATALOADER_PREFETCH_FACTOR
export QWEN_LORA_LR
export ACTION_LR
export RUN_ROOT_DIR
export EXTRA_TRAIN_ARGS

exec bash "${SCRIPT_DIR}/run_train_moe95k_lora_h200.sh"
