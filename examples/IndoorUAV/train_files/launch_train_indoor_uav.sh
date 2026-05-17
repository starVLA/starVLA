#!/bin/bash
# Launch QwenPI training on IndoorUAV-Replica with DeepSpeed ZeRO-2.
#
# Usage:
#   bash examples/IndoorUAV/train_files/launch_train_indoor_uav.sh
#   NUM_GPUS=4 GPU_IDS=4,5,6,7 bash examples/IndoorUAV/train_files/launch_train_indoor_uav.sh
#   ZERO_STAGE=3 GRAD_ACCUM=16 bash examples/IndoorUAV/train_files/launch_train_indoor_uav.sh
#
# Environment overrides:
#   NUM_GPUS      - default 4
#   GPU_IDS       - default 4,5,6,7  (CUDA_VISIBLE_DEVICES)
#   ZERO_STAGE    - default 2 (use 3 if even ZeRO-2 OOMs)
#   GRAD_ACCUM    - default 8
#   PER_DEVICE_BS - default 2
#   MAX_STEPS     - default 5000
#   RUN_ID        - default auto
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/starVLA"

NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
ZERO_STAGE="${ZERO_STAGE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
MAX_STEPS="${MAX_STEPS:-5000}"
RUN_ID="${RUN_ID:-indoor_uav_qwenpi_$(date +%Y%m%d_%H%M%S)}"

CONFIG_YAML="examples/IndoorUAV/train_files/starvla_train_indoor_uav.yaml"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-playground/Checkpoints}"
mkdir -p "${RUN_ROOT_DIR}/${RUN_ID}" logs
cp "$0" "${RUN_ROOT_DIR}/${RUN_ID}/" || true

# NCCL / CUDA env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Generate accelerate + DeepSpeed config (reuses Gemma4 helper)
ACCEL_CONFIG=$(python3 examples/Gemma4/_make_accelerate_config.py \
    --grad-accum "${GRAD_ACCUM}" \
    --num-processes "${NUM_GPUS}" \
    --zero-stage "${ZERO_STAGE}")
echo "[indoor-uav] generated accelerate config: ${ACCEL_CONFIG}"

echo "[indoor-uav] GPU_IDS=${GPU_IDS}  NUM_GPUS=${NUM_GPUS}  ZERO_STAGE=${ZERO_STAGE}"
echo "[indoor-uav] PER_DEVICE_BS=${PER_DEVICE_BS}  GRAD_ACCUM=${GRAD_ACCUM}"
echo "[indoor-uav] effective BS = ${PER_DEVICE_BS} × ${NUM_GPUS} × ${GRAD_ACCUM} = $((PER_DEVICE_BS * NUM_GPUS * GRAD_ACCUM))"
echo "[indoor-uav] MAX_STEPS=${MAX_STEPS}  RUN_ID=${RUN_ID}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BS}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval 1000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}"
