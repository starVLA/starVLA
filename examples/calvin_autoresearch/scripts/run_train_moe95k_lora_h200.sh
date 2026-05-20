#!/usr/bin/env bash
set -euo pipefail

# Continue from the GTY MoE95k ABC checkpoint with fresh Qwen LoRA adapters.
# Variants:
#   VARIANT=aug     GTY calvin_abc_augmented + WMH hard-task sampler/lang/image aug
#   VARIANT=mirror  same, plus left/right mirror aug for LR tasks
#
# This script is ABC-only and intentionally uses include_state=false/state_dim=7
# to stay compatible with the GTY MoE95k checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

VARIANT="${VARIANT:-aug}"
case "${VARIANT}" in
  aug)
    CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_qwen3vl_calvin_abc_augmented_moe_lora.yaml}"
    DEFAULT_RUN_PREFIX="abc_moe95k_lora_aug"
    ;;
  mirror)
    CONFIG_YAML="${CONFIG_YAML:-examples/calvin_autoresearch/train_files/starvla_qwen3vl_calvin_abc_augmented_moe_lora_lrmirror.yaml}"
    DEFAULT_RUN_PREFIX="abc_moe95k_lora_mirror"
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}; expected aug or mirror" >&2
    exit 2
    ;;
esac

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
GTY_ROOT="${GTY_ROOT:-${SHARED_ROOT}/members/GTY}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${SHARED_ROOT}/members/${MEMBER}/runs}"
TS="${TS:-$(date +%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${DEFAULT_RUN_PREFIX}_5k_${TS}}"

BASE_VLM="${BASE_VLM:-${SHARED_ROOT}/shared/models/base/Qwen3-VL-4B-Instruct-Action}"
DATA_ROOT="${DATA_ROOT:-${SHARED_ROOT}/shared/datasets/calvin_lerobot}"
DATA_MIX="${DATA_MIX:-calvin_abc_augmented}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-calvin_abc_train_v3.0}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${GTY_ROOT}/runs/gty_moe_posttrain_8h_GTY_0519_182014/checkpoints/steps_95000_pytorch_model.pt}"

detect_visible_gpu_ids() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    echo "${CUDA_VISIBLE_DEVICES}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
      | awk 'NF {gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}' \
      | paste -sd, -
  fi
}

count_csv_items() {
  local csv="$1"
  if [[ -z "${csv}" ]]; then
    echo 0
    return 0
  fi
  awk -F',' '{count=0; for (i=1; i<=NF; i++) if ($i != "") count++; print count}' <<<"${csv}"
}

DEFAULT_GPU_IDS="$(detect_visible_gpu_ids)"
DEFAULT_GPU_IDS="${DEFAULT_GPU_IDS:-0,1,2,3,4,5,6,7}"
DEFAULT_NUM_PROCESSES="$(count_csv_items "${DEFAULT_GPU_IDS}")"
if (( DEFAULT_NUM_PROCESSES <= 0 )); then
  DEFAULT_NUM_PROCESSES=8
fi

GPU_IDS="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
NUM_PROCESSES="${NUM_PROCESSES:-${DEFAULT_NUM_PROCESSES}}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-auto}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
LOG_GRAD_NORMS="${LOG_GRAD_NORMS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-1}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-1}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-1}"
DRY_RUN="${DRY_RUN:-0}"

QWEN_LORA_RANK="${QWEN_LORA_RANK:-8}"
QWEN_LORA_ALPHA="${QWEN_LORA_ALPHA:-16}"
QWEN_LORA_DROPOUT="${QWEN_LORA_DROPOUT:-0.05}"
QWEN_LORA_LAST_N_LAYERS="${QWEN_LORA_LAST_N_LAYERS:-4}"
QWEN_LORA_TARGET_MODULES="${QWEN_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
QWEN_LORA_LR="${QWEN_LORA_LR:-5.0e-06}"
ACTION_LR="${ACTION_LR:-5.0e-05}"

case "${PRETRAINED_CHECKPOINT}" in
  *Qwen3-VL-OFT-LIBERO*|*LIBERO*|*Robotwin*|*robotwin*|*Robocasa*|*robocasa*|*Behavior*|*BEHAVIOR*|*SimplerEnv*|*qwenpi_calvin_task_D_D*)
    echo "Refusing upstream action-trained checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
    exit 2
    ;;
esac

if [[ "${DATA_MIX}" != "calvin_abc_augmented" ]]; then
  echo "This launcher is ABC-only and expects DATA_MIX=calvin_abc_augmented, got ${DATA_MIX}" >&2
  exit 2
fi
case "${DATA_ROOT}/${TRAIN_DATASET_DIR}" in
  *task_D_D*|*ABCD-D*|*abcd-d*|*calvin-task-D-D*|*calvin-task-ABCD-D*)
    echo "Refusing training dataset path containing D split: ${DATA_ROOT}/${TRAIN_DATASET_DIR}" >&2
    exit 2
    ;;
esac
for required in "${PRETRAINED_CHECKPOINT}" "${BASE_VLM}/config.json" "${DATA_ROOT}/${TRAIN_DATASET_DIR}/meta/modality.json" "${CONFIG_YAML}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required asset: ${required}" >&2
    exit 3
  fi
done

if [[ "${MAIN_PROCESS_PORT}" == "auto" || "${MAIN_PROCESS_PORT}" == "0" ]]; then
  MAIN_PROCESS_PORT="$("${STARVLA_PYTHON:-python}" - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("", 0))
    print(sock.getsockname()[1])
PY
)"
fi

export PYTHONPATH="${STARVLA_ROOT}/examples/calvin_autoresearch/train_files:${GTY_ROOT}/train_files:${STARVLA_ROOT}:${PYTHONPATH:-}"
export GTY_ROOT
export NO_ALBUMENTATIONS_UPDATE=1

GTY_EXAMPLE_LINK="${STARVLA_ROOT}/examples/GTY_calvin"
if [[ ! -e "${GTY_EXAMPLE_LINK}" ]]; then
  ln -sfn "${GTY_ROOT}" "${GTY_EXAMPLE_LINK}"
fi

"${STARVLA_PYTHON:-python}" - "${PRETRAINED_CHECKPOINT}" "${CONFIG_YAML}" "${VARIANT}" <<'PY'
import sys
from pathlib import Path

from omegaconf import OmegaConf
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

ckpt = Path(sys.argv[1])
cfg_path = Path(sys.argv[2])
variant = sys.argv[3]

run_dir = ckpt.parent.parent
src_cfg_path = run_dir / "config.full.yaml"
if not src_cfg_path.exists():
    src_cfg_path = run_dir / "config.yaml"
source = OmegaConf.load(src_cfg_path)
target = OmegaConf.load(cfg_path)

mirror_enabled = bool(
    target.datasets.vla_data.get("spatial_augmentation", {})
    .get("left_right_mirror", {})
    .get("enabled", False)
)

checks = {
    "registry_has_calvin_abc_augmented": "calvin_abc_augmented" in DATASET_NAMED_MIXTURES,
    "source_framework_moe_or_moe_lora": source.framework.name in {"QwenGR00T_MoE", "QwenGR00T_MoE_LoRA"},
    "source_state_dim_7": int(source.framework.action_model.state_dim) == 7,
    "source_include_state_false": not bool(source.datasets.vla_data.include_state),
    "target_framework_moe_lora": target.framework.name == "QwenGR00T_MoE_LoRA",
    "target_qwen_lora_enabled": bool(target.framework.get("qwen_lora", {}).get("enabled", False)),
    "target_state_dim_7": int(target.framework.action_model.state_dim) == 7,
    "target_include_state_false": not bool(target.datasets.vla_data.include_state),
    "target_data_mix_augmented": target.datasets.vla_data.data_mix == "calvin_abc_augmented",
    "target_has_balanced_sampler": target.datasets.vla_data.get("sampler", {}).get("type") == "task_balanced",
    "target_has_language_aug": bool(target.datasets.vla_data.get("language_augmentation", {}).get("enabled", False)),
    "target_has_image_aug": bool(target.datasets.vla_data.get("image_augmentation", {}).get("enabled", False)),
    "mirror_flag_matches_variant": mirror_enabled == (variant == "mirror"),
}
bad = [name for name, ok in checks.items() if not ok]
if bad:
    for name, ok in checks.items():
        print(f"[moe95k-lora] check {name}: {ok}")
    raise SystemExit(f"MoE95k LoRA sanity checks failed: {bad}")
print(f"[moe95k-lora] source checkpoint OK: {ckpt}")
print(f"[moe95k-lora] target config OK: {cfg_path}")
print(f"[moe95k-lora] variant={variant} mirror_enabled={mirror_enabled}")
PY

if [[ "${DRY_RUN}" != "1" ]]; then
  "${STARVLA_PYTHON:-python}" - "${NUM_PROCESSES}" <<'PY'
import sys
import torch

required = int(sys.argv[1])
available = torch.cuda.device_count() if torch.cuda.is_available() else 0
if available < required:
    raise SystemExit(
        f"This launcher requires {required} visible CUDA devices, but PyTorch sees {available}."
    )
print(f"[moe95k-lora] visible CUDA devices: {available}")
PY
fi

export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${RUN_ROOT_DIR}"

cmd=(
  accelerate launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  examples/calvin_autoresearch/train_files/run_train_moe_lora_entry.py
  --config_yaml "${CONFIG_YAML}"
  --run_id "${RUN_ID}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --framework.qwen_lora.rank "${QWEN_LORA_RANK}"
  --framework.qwen_lora.alpha "${QWEN_LORA_ALPHA}"
  --framework.qwen_lora.dropout "${QWEN_LORA_DROPOUT}"
  --framework.qwen_lora.last_n_layers "${QWEN_LORA_LAST_N_LAYERS}"
  --framework.qwen_lora.target_modules "${QWEN_LORA_TARGET_MODULES}"
  --datasets.vla_data.data_root_dir "${DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}"
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --trainer.log_grad_norms "${LOG_GRAD_NORMS}"
  --trainer.learning_rate.qwen_vl_interface "${QWEN_LORA_LR}"
  --trainer.learning_rate.action_model "${ACTION_LR}"
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

cat <<EOF
[moe95k-lora] variant=${VARIANT}
[moe95k-lora] run_id=${RUN_ID}
[moe95k-lora] ckpt=${PRETRAINED_CHECKPOINT}
[moe95k-lora] data_mix=${DATA_MIX} train_dataset=${TRAIN_DATASET_DIR}
[moe95k-lora] steps=${MAX_TRAIN_STEPS} save_interval=${SAVE_INTERVAL}
[moe95k-lora] batch=${BATCH_SIZE} num_processes=${NUM_PROCESSES}
[moe95k-lora] lora rank=${QWEN_LORA_RANK} alpha=${QWEN_LORA_ALPHA} lr=${QWEN_LORA_LR}
[moe95k-lora] main_process_port=${MAIN_PROCESS_PORT}
EOF

printf '[moe95k-lora] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
