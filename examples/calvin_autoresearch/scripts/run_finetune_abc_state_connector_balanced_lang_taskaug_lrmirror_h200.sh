#!/usr/bin/env bash
set -euo pipefail

# ABC-only post-training launcher for the left/right mirror branch.
#
# Purpose:
#   Continue from a WMH-produced state8+connector checkpoint using:
#     - task-balanced sampling
#     - hard-task language paraphrases
#     - task-aware light image augmentation
#     - left/right mirror augmentation for LR tasks only
#
# It intentionally does NOT train on CALVIN D.
# It intentionally refuses common upstream action-trained checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_gr00t_qwen3vl_action_calvin_abc_state8_connector_balanced_lang_taskaug_lrmirror.yaml}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
DATA_ROOT="${DATA_ROOT:-playground/Datasets/calvin_lerobot}"
DATA_MIX="${DATA_MIX:-calvin_abc_train_state_v3.0}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-}"

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-abc_state8_connector_balanced_lang_taskaug_lrmirror_ft2k_${TS}}"

PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${SHARED_ROOT}/members/WMH/runs/abc_state8_connector_8h200_bs96_8k_0519_083200/checkpoints/steps_8000_pytorch_model.pt}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-auto}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-96}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
LOG_GRAD_NORMS="${LOG_GRAD_NORMS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-1}"
LR_MIRROR_PROBABILITY="${LR_MIRROR_PROBABILITY:-}"
DRY_RUN="${DRY_RUN:-0}"

case "${PRETRAINED_CHECKPOINT}" in
  *Qwen3-VL-OFT-LIBERO*|*LIBERO*|*Robotwin*|*robotwin*|*Robocasa*|*robocasa*|*Behavior*|*BEHAVIOR*|*SimplerEnv*|*qwenpi_calvin_task_D_D*)
    echo "Refusing action-trained upstream checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
    exit 2
    ;;
esac

case "${DATA_MIX}" in
  calvin_abc_train_state_v3.0)
    TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-calvin_abc_train_v3.0}"
    ;;
  *)
    echo "This finetune entry is ABC-only and state-aware. Refusing DATA_MIX=${DATA_MIX}" >&2
    exit 2
    ;;
esac

case "${DATA_ROOT}/${TRAIN_DATASET_DIR}" in
  *task_D_D*|*ABCD-D*|*abcd-d*|*calvin-task-D-D*|*calvin-task-ABCD-D*)
    echo "This finetune entry must not train on CALVIN D or ABCD-D data." >&2
    echo "Refusing dataset path: ${DATA_ROOT}/${TRAIN_DATASET_DIR}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "PRETRAINED_CHECKPOINT not found: ${PRETRAINED_CHECKPOINT}" >&2
  exit 3
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
echo "[finetune-abc-lrmirror] main_process_port: ${MAIN_PROCESS_PORT}"

"${STARVLA_PYTHON:-python}" - "${PRETRAINED_CHECKPOINT}" "${CONFIG_YAML}" "${LR_MIRROR_PROBABILITY}" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf

ckpt = Path(sys.argv[1])
target_cfg_path = Path(sys.argv[2])
probability_override = sys.argv[3].strip()

run_dir = ckpt.parent.parent
source_cfg_path = run_dir / "config.full.yaml"
if not source_cfg_path.exists():
    source_cfg_path = run_dir / "config.yaml"
if not source_cfg_path.exists():
    raise SystemExit(f"Checkpoint run config not found next to {ckpt}")

source = OmegaConf.load(source_cfg_path)
target = OmegaConf.load(target_cfg_path)
src_action = source.framework.action_model
tgt_action = target.framework.action_model
mirror = (
    target.datasets.vla_data.get("spatial_augmentation", {})
    .get("left_right_mirror", {})
)
tasks = mirror.get("tasks", {})
probability = float(probability_override) if probability_override else float(mirror.get("probability", 0.0))

checks = {
    "source_state_dim": int(src_action.state_dim) == 8,
    "target_state_dim": int(tgt_action.state_dim) == 8,
    "source_connector": bool(source.framework.get("vl_connector", {}).get("enabled", False)),
    "target_connector": bool(target.framework.get("vl_connector", {}).get("enabled", False)),
    "target_include_state": bool(target.datasets.vla_data.include_state),
    "target_has_balanced_sampler": target.datasets.vla_data.get("sampler", {}).get("type") == "task_balanced",
    "target_has_language_aug": bool(target.datasets.vla_data.get("language_augmentation", {}).get("enabled", False)),
    "target_has_image_aug": bool(target.datasets.vla_data.get("image_augmentation", {}).get("enabled", False)),
    "target_has_lr_mirror": bool(mirror.get("enabled", False)),
    "target_lr_mirror_lr_tasks_only": str(mirror.get("apply_to", "")) == "lr_tasks",
    "target_lr_mirror_probability_safe": 0.0 < probability <= 0.5,
    "target_lr_mirror_flips_primary": bool(mirror.get("flip_primary_image", False)),
    "target_lr_mirror_flips_wrist": bool(mirror.get("flip_wrist_image", False)),
    "target_lr_mirror_has_tasks": len(tasks) >= 8,
}
bad = [name for name, ok in checks.items() if not ok]
if bad:
    for name, ok in checks.items():
        print(f"[finetune-abc-lrmirror] check {name}: {ok}")
    raise SystemExit(f"Config/checkpoint sanity checks failed: {bad}")
print(f"[finetune-abc-lrmirror] checkpoint sanity OK: {ckpt}")
print(f"[finetune-abc-lrmirror] target config sanity OK: {target_cfg_path}")
print(f"[finetune-abc-lrmirror] mirror probability: {probability}")
PY

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
print(f"[finetune-abc-lrmirror] visible CUDA devices: {available}")
PY
fi

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
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
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

if [[ -n "${LR_MIRROR_PROBABILITY}" ]]; then
  cmd+=(--datasets.vla_data.spatial_augmentation.left_right_mirror.probability "${LR_MIRROR_PROBABILITY}")
fi

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=( ${EXTRA_TRAIN_ARGS} )
  cmd+=("${extra_args[@]}")
fi

printf '[finetune-abc-lrmirror] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
