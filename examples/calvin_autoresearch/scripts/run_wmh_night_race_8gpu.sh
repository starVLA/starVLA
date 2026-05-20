#!/usr/bin/env bash
set -euo pipefail

# Launch two ABC-only post-training branches in parallel on one 8-GPU node.
# Branch A: hard-sampler-v2 + language/task-aware image augmentation, no mirror.
# Branch B: same data recipe plus left/right mirror augmentation.

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
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/members/${MEMBER}/logs}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
RACE_ID="${RACE_ID:-night_race_hardv2_${TS}}"
RACE_DIR="${LOG_DIR}/${RACE_ID}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-6000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-${MAX_TRAIN_STEPS}}"
BATCH_SIZE="${BATCH_SIZE:-96}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
CONNECTOR_LR="${CONNECTOR_LR:-3.0e-05}"
ACTION_LR="${ACTION_LR:-1.0e-04}"
MIRROR_PROBABILITY="${MIRROR_PROBABILITY:-0.25}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${SHARED_ROOT}/members/WMH/runs/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt}"
CONTINUE_LATEST="${CONTINUE_LATEST:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${RACE_DIR}" "${RUN_ROOT_DIR}" wmh_links

HARD_V2_ARGS=(
  --trainer.learning_rate.vl_connector "${CONNECTOR_LR}"
  --trainer.learning_rate.action_model "${ACTION_LR}"
  --datasets.vla_data.sampler.oversample_tasks.turn_on_lightbulb 5.0
  --datasets.vla_data.sampler.oversample_tasks.stack_block 4.0
  --datasets.vla_data.sampler.oversample_tasks.move_slider_right 3.0
  --datasets.vla_data.sampler.oversample_tasks.push_red_block_left 2.0
  --datasets.vla_data.sampler.oversample_tasks.push_blue_block_left 2.0
  --datasets.vla_data.sampler.oversample_tasks.push_pink_block_left 2.0
)

join_args() {
  local out="" arg
  for arg in "$@"; do
    out+="${arg} "
  done
  printf '%s' "${out% }"
}

latest_checkpoint() {
  local pattern="$1"
  find "${RUN_ROOT_DIR}" -maxdepth 3 -type f -path "*/${pattern}/checkpoints/steps_*_pytorch_model.pt" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

if [[ "${CONTINUE_LATEST}" == "1" ]]; then
  AUG_PRETRAINED_CHECKPOINT="${AUG_PRETRAINED_CHECKPOINT:-$(latest_checkpoint 'abc_aug_hardv2_*')}"
  MIRROR_PRETRAINED_CHECKPOINT="${MIRROR_PRETRAINED_CHECKPOINT:-$(latest_checkpoint 'abc_mirror_hardv2_*')}"
  if [[ -z "${AUG_PRETRAINED_CHECKPOINT}" || -z "${MIRROR_PRETRAINED_CHECKPOINT}" ]]; then
    echo "CONTINUE_LATEST=1 requested, but latest branch checkpoints were not found." >&2
    echo "Wait for the current run to save checkpoints, or set AUG_PRETRAINED_CHECKPOINT and MIRROR_PRETRAINED_CHECKPOINT." >&2
    exit 3
  fi
else
  AUG_PRETRAINED_CHECKPOINT="${AUG_PRETRAINED_CHECKPOINT:-${PRETRAINED_CHECKPOINT}}"
  MIRROR_PRETRAINED_CHECKPOINT="${MIRROR_PRETRAINED_CHECKPOINT:-${PRETRAINED_CHECKPOINT}}"
fi

launch_branch() {
  local branch="$1"
  local gpu_ids="$2"
  local run_script="$3"
  local branch_ckpt="$4"
  local run_id log_path extra_args pid

  run_id="abc_${branch}_${MAX_TRAIN_STEPS}_${TS}"
  log_path="${RACE_DIR}/${run_id}.log"
  extra_args="$(join_args "${HARD_V2_ARGS[@]}")"

  echo "[night-race] branch=${branch}"
  echo "[night-race]   gpu_ids=${gpu_ids}"
  echo "[night-race]   run_id=${run_id}"
  echo "[night-race]   log=${log_path}"
  echo "[night-race]   ckpt=${branch_ckpt}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[night-race] DRY_RUN command for ${branch}:"
    printf '  env CUDA_VISIBLE_DEVICES=%q GPU_IDS=%q NUM_PROCESSES=4 RUN_ID=%q MAX_TRAIN_STEPS=%q SAVE_INTERVAL=%q BATCH_SIZE=%q PRETRAINED_CHECKPOINT=%q EXTRA_TRAIN_ARGS=%q bash %q\n' \
      "${gpu_ids}" "${gpu_ids}" "${run_id}" "${MAX_TRAIN_STEPS}" "${SAVE_INTERVAL}" "${BATCH_SIZE}" "${branch_ckpt}" "${extra_args}" "${run_script}"
    return 0
  fi

  if [[ "${branch}" == *mirror* ]]; then
    nohup env \
      CUDA_VISIBLE_DEVICES="${gpu_ids}" \
      GPU_IDS="${gpu_ids}" \
      NUM_PROCESSES=4 \
      RUN_ID="${run_id}" \
      RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
      MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
      SAVE_INTERVAL="${SAVE_INTERVAL}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
      DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR}" \
      PRETRAINED_CHECKPOINT="${branch_ckpt}" \
      LR_MIRROR_PROBABILITY="${MIRROR_PROBABILITY}" \
      EXTRA_TRAIN_ARGS="${extra_args}" \
      LOG_GRAD_NORMS=1 \
      SKIP_FINAL_SAVE=1 \
      bash "${run_script}" > "${log_path}" 2>&1 &
  else
    nohup env \
      CUDA_VISIBLE_DEVICES="${gpu_ids}" \
      GPU_IDS="${gpu_ids}" \
      NUM_PROCESSES=4 \
      RUN_ID="${run_id}" \
      RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
      MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
      SAVE_INTERVAL="${SAVE_INTERVAL}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
      DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR}" \
      PRETRAINED_CHECKPOINT="${branch_ckpt}" \
      EXTRA_TRAIN_ARGS="${extra_args}" \
      LOG_GRAD_NORMS=1 \
      SKIP_FINAL_SAVE=1 \
      bash "${run_script}" > "${log_path}" 2>&1 &
  fi
  pid="$!"
  echo "${pid}" > "${RACE_DIR}/${branch}.pid"
}

cat > "${RACE_DIR}/manifest.txt" <<EOF
race_id=${RACE_ID}
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
source_checkpoint_default=${PRETRAINED_CHECKPOINT}
continue_latest=${CONTINUE_LATEST}
source_checkpoint_aug=${AUG_PRETRAINED_CHECKPOINT}
source_checkpoint_mirror=${MIRROR_PRETRAINED_CHECKPOINT}
train_data=CALVIN ABC only
calvin_d_training=forbidden
qwen_update=no
branches=aug_hardv2,mirror_hardv2
gpus_aug_hardv2=0,1,2,3
gpus_mirror_hardv2=4,5,6,7
max_train_steps=${MAX_TRAIN_STEPS}
save_interval=${SAVE_INTERVAL}
batch_size_per_gpu=${BATCH_SIZE}
connector_lr=${CONNECTOR_LR}
action_lr=${ACTION_LR}
mirror_probability=${MIRROR_PROBABILITY}
hard_sampler_v2=turn_on_lightbulb:5,stack_block:4,move_slider_right:3,left_push_tasks:2
EOF

launch_branch \
  "aug_hardv2" \
  "0,1,2,3" \
  "${SCRIPT_DIR}/run_finetune_abc_state_connector_balanced_lang_taskaug_h200.sh" \
  "${AUG_PRETRAINED_CHECKPOINT}"

launch_branch \
  "mirror_hardv2" \
  "4,5,6,7" \
  "${SCRIPT_DIR}/run_finetune_abc_state_connector_balanced_lang_taskaug_lrmirror_h200.sh" \
  "${MIRROR_PRETRAINED_CHECKPOINT}"

ln -sfn "${RACE_DIR}/manifest.txt" "wmh_links/latest_night_race_manifest.txt"
ln -sfn "${RACE_DIR}/abc_aug_hardv2_${MAX_TRAIN_STEPS}_${TS}.log" "wmh_links/latest_race_aug.log"
ln -sfn "${RACE_DIR}/abc_mirror_hardv2_${MAX_TRAIN_STEPS}_${TS}.log" "wmh_links/latest_race_mirror.log"

echo "[night-race] manifest=./wmh_links/latest_night_race_manifest.txt"
echo "[night-race] aug log=./wmh_links/latest_race_aug.log"
echo "[night-race] mirror log=./wmh_links/latest_race_mirror.log"
echo "[night-race] launched"
