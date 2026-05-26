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
#
# Usage:
#   sbatch examples/MiniCPM/submit_hpc3_libero.sh
#   FRAMEWORK=MiniCPMGR00T sbatch examples/MiniCPM/submit_hpc3_libero.sh
#   DATA_MIX=libero_spatial sbatch examples/MiniCPM/submit_hpc3_libero.sh
#
# Environment overrides:
#   FRAMEWORK     - MiniCPMPI (default) or MiniCPMGR00T
#   BASE_VLM      - HF model id or local path (default openbmb/MiniCPM-V-4.6)
#   DATA_MIX      - libero_all / libero_spatial / libero_object / libero_goal / libero_10
#   MAX_STEPS     - default 100000
#   PER_DEVICE_BS - default 4
#   GRAD_ACCUM    - default 4 (effective BS = 4×8×4 = 128)
#   ATTN_IMPL     - default sdpa
#   ZERO_STAGE    - default 2

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/starVLA"

FRAMEWORK="${FRAMEWORK:-MiniCPMPI}"
BASE_VLM="${BASE_VLM:-openbmb/MiniCPM-V-4.6}"
DATA_MIX="${DATA_MIX:-libero_all}"
MAX_STEPS="${MAX_STEPS:-100000}"
PER_DEVICE_BS="${PER_DEVICE_BS:-4}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
ENABLE_GRAD_CKPT="${ENABLE_GRAD_CKPT:-true}"
ZERO_STAGE="${ZERO_STAGE:-2}"
RUN_ID="${RUN_ID:-minicpm_${FRAMEWORK}_${DATA_MIX}_${SLURM_JOB_ID:-local}}"

LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
CONFIG_YAML="examples/LIBERO/train_files/starvla_cotrain_libero.yaml"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-results/Checkpoints}"

mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}" logs
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/" || true

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ACCEL_CONFIG=$(python3 examples/MiniCPM/_make_accelerate_config.py \
    --grad-accum "${GRAD_ACCUM}" \
    --num-processes 8 \
    --zero-stage "${ZERO_STAGE}")
echo "[minicpm-vla] generated accelerate config: ${ACCEL_CONFIG}"

echo "[minicpm-vla] FRAMEWORK=${FRAMEWORK}  BASE_VLM=${BASE_VLM}"
echo "[minicpm-vla] DATA_MIX=${DATA_MIX}  STEPS=${MAX_STEPS}  PER_DEVICE_BS=${PER_DEVICE_BS}"
echo "[minicpm-vla] GRAD_ACCUM=${GRAD_ACCUM}  effective BS = ${PER_DEVICE_BS}×8×${GRAD_ACCUM}"
echo "[minicpm-vla] ENABLE_GRAD_CKPT=${ENABLE_GRAD_CKPT}  ZERO_STAGE=${ZERO_STAGE}"
echo "[minicpm-vla] RUN_ID=${RUN_ID}"

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
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
