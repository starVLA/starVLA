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
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-}"

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUN_ID="${RUN_ID:-abc_pretrain_qwen3vl_gr00t_headonly_h200}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"

NUM_PROCESSES="${NUM_PROCESSES:-3}"
GPU_IDS="${GPU_IDS:-0,1,2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-auto}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
LOG_GRAD_NORMS="${LOG_GRAD_NORMS:-0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${PRETRAINED_CHECKPOINT:-}" ]]; then
  echo "PRETRAINED_CHECKPOINT is not allowed for this CALVIN ABC pretrain path." >&2
  exit 2
fi

if [[ "${MAIN_PROCESS_PORT}" == "auto" || "${MAIN_PROCESS_PORT}" == "0" ]]; then
  MAIN_PROCESS_PORT="$("${STARVLA_PYTHON:-python}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
fi
echo "[train-abc-h200] main_process_port: ${MAIN_PROCESS_PORT}"

if [[ "${DRY_RUN}" != "1" ]]; then
  "${STARVLA_PYTHON:-python}" - "${NUM_PROCESSES}" <<'PY'
import sys
import torch

required = int(sys.argv[1])
available = torch.cuda.device_count() if torch.cuda.is_available() else 0
if available < required:
    raise SystemExit(
        f"This H200 launcher requires at least {required} visible CUDA devices, "
        f"but PyTorch sees {available}. Run inside a GPU allocation and check nvidia-smi."
    )
print(f"[train-abc-h200] visible CUDA devices: {available}")
PY
fi

case "${DATA_MIX}" in
  calvin_abc_train_v3.0)
    TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-calvin_abc_train_v3.0}"
    ;;
  calvin_abc_train_state_v3.0)
    TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-calvin_abc_train_v3.0}"
    ;;
  *)
  echo "This H200 pretrain entry is ABC-only. Refusing DATA_MIX=${DATA_MIX}" >&2
  exit 2
    ;;
esac

case "${DATA_ROOT}/${TRAIN_DATASET_DIR}" in
  *task_D_D*|*ABCD-D*|*abcd-d*|*calvin-task-D-D*|*calvin-task-ABCD-D*)
    echo "This H200 pretrain entry must not train on CALVIN D or ABCD-D data." >&2
    echo "Refusing dataset path: ${DATA_ROOT}/${TRAIN_DATASET_DIR}" >&2
    exit 2
    ;;
esac

STRICT_ASSETS="${STRICT_ASSETS:-1}" \
BASE_VLM="${BASE_VLM}" \
DATA_ROOT="${DATA_ROOT}" \
TRAIN_DATASET="${TRAIN_DATASET_DIR}" \
CONFIG_YAML="${CONFIG_YAML}" \
"${SCRIPT_DIR}/verify_assets.sh"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"

mkdir -p "${RUN_ROOT_DIR}"

cmd=(
  accelerate launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
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
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --trainer.log_grad_norms "${LOG_GRAD_NORMS}"
  --datasets.vla_data.num_workers "${DATALOADER_NUM_WORKERS}"
  --datasets.vla_data.prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --datasets.vla_data.pin_memory "${DATALOADER_PIN_MEMORY}"
  --datasets.vla_data.persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
  --trainer.skip_final_save "${SKIP_FINAL_SAVE}"
)

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_TRAIN_ARGS} )
  cmd+=("${extra_args[@]}")
fi

printf '[train-abc-h200] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
