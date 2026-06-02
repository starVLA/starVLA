#!/bin/bash
#SBATCH --job-name=minicpm-vla
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=logs/minicpm_vla_%j.log
#
# Slurm submission for MiniCPM-V 4.6 LIBERO training (8×GPU).
# Aligned with upstream starVLA run_libero_train.sh:
#   - Uses static deepspeed_zero2.yaml (GA hard-coded to 1 in ds_config.yaml)
#   - Per-device BS=16, num_processes=8 -> effective BS = 128
#   - Note: upstream's `trainer.gradient_accumulation_steps` in YAML is dead config;
#     real GA is whatever ds_config.yaml says (1). We do NOT pass --grad-accum.
#
# Usage:
#   sbatch examples/MiniCPM/submit_hpc3_libero.sh
#   FRAMEWORK=MiniCPMGR00T sbatch examples/MiniCPM/submit_hpc3_libero.sh
#
# Environment overrides:
#   FRAMEWORK     - MiniCPMPI (default) or MiniCPMGR00T
#   BASE_VLM      - HF model id or local path (default openbmb/MiniCPM-V-4.6)
#   DATA_MIX      - libero_all / libero_spatial / ...
#   MAX_STEPS     - default 80000 (upstream guideline)
#   PER_DEVICE_BS - default 16  (matches upstream run_libero_train.sh)
#   ATTN_IMPL     - default sdpa
#   FREEZE_MODULES - default '' (unfreeze VLM, matches upstream)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/starVLA"

FRAMEWORK="${FRAMEWORK:-MiniCPMPI}"
BASE_VLM="${BASE_VLM:-openbmb/MiniCPM-V-4.6}"
DATA_MIX="${DATA_MIX:-libero_all}"
MAX_STEPS="${MAX_STEPS:-80000}"
PER_DEVICE_BS="${PER_DEVICE_BS:-16}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-true}"
FREEZE_MODULES="${FREEZE_MODULES:-}"
RUN_ID="${RUN_ID:-minicpm_${FRAMEWORK}_${DATA_MIX}_upstream_${SLURM_JOB_ID:-local}}"

LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
CONFIG_YAML="examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
ACCEL_CONFIG="starVLA/config/deepseeds/deepspeed_zero2.yaml"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}" logs
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/" || true

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[minicpm-vla] FRAMEWORK=${FRAMEWORK}  BASE_VLM=${BASE_VLM}"
echo "[minicpm-vla] DATA_MIX=${DATA_MIX}  STEPS=${MAX_STEPS}  PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "[minicpm-vla] effective BS = ${PER_DEVICE_BS}×8×1 (GA=1 from ds_config.yaml)"
echo "[minicpm-vla] FREEZE_MODULES='${FREEZE_MODULES}'  RUN_ID=${RUN_ID}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes 8 \
  --num_machines 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name "${FRAMEWORK}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.qwenvl.attn_implementation "${ATTN_IMPL}" \
  --framework.qwenvl.enable_gradient_checkpointing "${ENABLE_GRAD_CKPT}" \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim 1024 \
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BS}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
