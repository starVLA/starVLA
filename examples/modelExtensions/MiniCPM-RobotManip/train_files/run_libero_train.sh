#!/usr/bin/env bash
# Fine-tune the released MiniCPM-RobotManip on LIBERO (80-D EE6D). 8-GPU node.
set -euo pipefail
cd "$(dirname "$0")/../../../.."   # -> repo root

export VLM_PATH="${VLM_PATH:-openbmb/MiniCPM-RobotManip}"
export LIBERO_EE6D_ROOT="${LIBERO_EE6D_ROOT:-playground/Datasets/LIBERO_EE6D}"
export TOKENIZERS_PARALLELISM=false

# Keep host thread use bounded across eight ranks.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_id="${RUN_ID:-minicpm_robotmanip_libero_full_finetune}"

# Optional runtime overrides. Learning rates and scheduler settings stay in the
# YAML so the published recipe has a single source of truth.
MAX_STEPS="${MAX_STEPS:-1500}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
PER_DEVICE_BS="${PER_DEVICE_BS:-12}"
SAVE_INTERVAL="${SAVE_INTERVAL:-250}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000000}"
LOG_FREQ="${LOG_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CONFIG_YAML="${CONFIG_YAML:-examples/modelExtensions/MiniCPM-RobotManip/train_files/minicpm_robotmanip_libero.yaml}"

python examples/modelExtensions/MiniCPM-RobotManip/train_files/install_modality.py \
  "${LIBERO_EE6D_ROOT}"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-examples/modelExtensions/MiniCPM-RobotManip/train_files/deepspeed_zero2_mixed_dtype.yaml}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.base_vlm "${VLM_PATH}" \
  --datasets.vla_data.data_root_dir "${LIBERO_EE6D_ROOT}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BS}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.num_warmup_steps "${WARMUP_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.logging_frequency "${LOG_FREQ}" \
  --run_id "${run_id}" \
  --run_root_dir playground/Checkpoints
