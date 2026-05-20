#!/usr/bin/env bash
set -euo pipefail

# Overnight mirror branch pipeline:
#   1. continue from the fixed WMH 8k state8+connector checkpoint
#   2. train mirror augmentation branch on ABC only
#   3. optionally run D n10 and n100 eval after training
#
# The source checkpoint is never modified. The run directory records the source
# checkpoint path in night_manifest.txt before training starts.

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
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
REPORT_ROOT="${REPORT_ROOT:-${SHARED_ROOT}/members/${MEMBER}/reports}"
TS="${TS:-$(date +%m%d_%H%M%S)}"

MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
RUN_ID="${RUN_ID:-abc_state8_connector_balanced_lang_taskaug_lrmirror_night8_${MAX_TRAIN_STEPS}_${TS}}"
RUN_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${SHARED_ROOT}/members/WMH/runs/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt}"
MIRROR_CONNECTOR_LR="${MIRROR_CONNECTOR_LR:-3.0e-05}"
MIRROR_ACTION_LR="${MIRROR_ACTION_LR:-1.0e-04}"

RUN_EVAL_N10="${RUN_EVAL_N10:-1}"
RUN_EVAL_N100="${RUN_EVAL_N100:-1}"
ALLOW_EVAL_FAILURE="${ALLOW_EVAL_FAILURE:-1}"

mkdir -p "${RUN_DIR}" "${REPORT_ROOT}"

cat > "${RUN_DIR}/night_manifest.txt" <<EOF
run_id=${RUN_ID}
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
pretrained_checkpoint=${PRETRAINED_CHECKPOINT}
config=examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug_lrmirror.yaml
data_mix=calvin_abc_train_state_v3.0
train_data=ABC only
calvin_d_training=forbidden
max_train_steps=${MAX_TRAIN_STEPS}
save_interval=${SAVE_INTERVAL}
num_processes=8
gpu_ids=0,1,2,3,4,5,6,7
batch_size=${BATCH_SIZE:-96}
freeze_modules=qwen_vl_interface
train_modules=vl_connector,action_model
lr_vl_connector=${MIRROR_CONNECTOR_LR}
lr_action_model=${MIRROR_ACTION_LR}
lr_scheduler=cosine_with_min_lr
min_lr=1e-6
qwen_update=no
analysis2_absorbed=freeze_qwen_keep_stage_a_lower_connector_lr
EOF

export RUN_ID
export RUN_ROOT_DIR
export PRETRAINED_CHECKPOINT
export MAX_TRAIN_STEPS
export SAVE_INTERVAL
export NUM_PROCESSES=8
export GPU_IDS=0,1,2,3,4,5,6,7
export BATCH_SIZE="${BATCH_SIZE:-96}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
export LOG_GRAD_NORMS="${LOG_GRAD_NORMS:-1}"
export SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-1}"
export EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-} --trainer.learning_rate.vl_connector ${MIRROR_CONNECTOR_LR} --trainer.learning_rate.action_model ${MIRROR_ACTION_LR}"

echo "[night-mirror] run_id=${RUN_ID}"
echo "[night-mirror] source_ckpt=${PRETRAINED_CHECKPOINT}"
echo "[night-mirror] steps=${MAX_TRAIN_STEPS} save_interval=${SAVE_INTERVAL}"
echo "[night-mirror] qwen_update=no; train=vl_connector+action_model"
echo "[night-mirror] lr: vl_connector=${MIRROR_CONNECTOR_LR}, action_model=${MIRROR_ACTION_LR}"

bash examples/calvin_autoresearch/scripts/run_finetune_abc_state_connector_balanced_lang_taskaug_lrmirror_h200.sh

CKPT="${RUN_DIR}/checkpoints/steps_${MAX_TRAIN_STEPS}_pytorch_model.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="$(
    find "${RUN_DIR}/checkpoints" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -1 \
      | cut -d' ' -f2-
  )"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[night-mirror] no checkpoint found after training" >&2
  exit 4
fi
echo "${CKPT}" > "${RUN_DIR}/latest_ckpt.txt"
echo "[night-mirror] latest_ckpt=${CKPT}"

run_eval() {
  local total="$1"
  local eval_ts eval_dir status
  eval_ts="$(date +%m%d_%H%M%S)"
  eval_dir="${REPORT_ROOT}/eval_mirror_${RUN_ID}_d_n${total}_${eval_ts}"
  echo "[night-mirror] eval n=${total}: ${eval_dir}"
  set +e
  CKPT="${CKPT}" \
  TOTAL_SEQUENCES="${total}" \
  CALVIN_SEND_STATE=1 \
  GPU_IDS=0,1,2,3,4,5,6,7 \
  WORKERS_PER_GPU="${EVAL_WORKERS_PER_GPU:-1}" \
  BASE_PORT="$((7600 + RANDOM % 300))" \
  SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1200}" \
  RESULT_WAIT_TIMEOUT="${RESULT_WAIT_TIMEOUT:-7200}" \
  DEBUG=0 \
  EVAL_LOG_DIR="${eval_dir}" \
    bash examples/calvin_autoresearch/scripts/run_eval_abc_to_d_parallel_auto.sh
  status="$?"
  EVAL_DIR="${eval_dir}" KILL_SERVERS=1 bash examples/calvin_autoresearch/scripts/finalize_parallel_eval_dir.sh || true
  set -e
  if [[ "${status}" != "0" && "${ALLOW_EVAL_FAILURE}" != "1" ]]; then
    return "${status}"
  fi
  echo "[night-mirror] eval n=${total} status=${status}"
}

if [[ "${RUN_EVAL_N10}" == "1" ]]; then
  run_eval 10
fi
if [[ "${RUN_EVAL_N100}" == "1" ]]; then
  run_eval 100
fi

echo "[night-mirror] done"
