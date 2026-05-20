#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc.yaml}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/calvin_lerobot}"
DATA_MIX="${DATA_MIX:-calvin_abc_train_v3.0}"
RUN_ID="${RUN_ID:-baseline_qwen3vl_action_gr00t_calvin_abc_smoke}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
GPU_ID="${GPU_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  echo "PRETRAINED_CHECKPOINT is not allowed for this baseline smoke path." >&2
  exit 2
fi

STRICT_ASSETS="${STRICT_ASSETS:-1}" \
BASE_VLM="${BASE_VLM}" \
DATA_ROOT="${DATA_ROOT}" \
"${SCRIPT_DIR}/verify_assets.sh"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

cmd=(
  accelerate launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  --run_id "${RUN_ID}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
)

printf '[train] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
