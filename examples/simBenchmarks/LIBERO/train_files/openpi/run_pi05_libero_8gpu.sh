#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/LIBERO/train_files/openpi/pi05_libero_8gpu.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
DATA_ROOT="${DATA_ROOT:-data/starvla_lerobot_root/libero}"
DATA_MIX="${DATA_MIX:-openpi_libero_all}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-paligemma_tokenizer.model}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-openpi_converted_protocol/pi05_base_starvla/fp32/model.safetensors}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
RUN_ID="${RUN_ID:-starvla_pi05_libero_multitask}"
WANDB_PROJECT="${WANDB_PROJECT:-starvla_pi}"
WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name PI05 \
  --framework.tokenizer.model_path "${TOKENIZER_MODEL}" \
  --framework.action_horizon 10 \
  --framework.max_token_len 200 \
  --framework.discrete_state_input false \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR}" \
  --datasets.vla_data.persistent_workers true \
  --datasets.vla_data.pin_memory true \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 120000 \
  --trainer.num_warmup_steps 10000 \
  --trainer.lr_scheduler_type constant_with_warmup \
  --trainer.learning_rate.base 5.0e-5 \
  --trainer.learning_rate.action_head 5.0e-5 \
  --trainer.optimizer.weight_decay 1.0e-10 \
  --trainer.gradient_clipping 1.0 \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.save_interval 30000 \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
