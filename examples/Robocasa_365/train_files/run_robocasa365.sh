#!/usr/bin/env bash
# RoboCasa365 (PandaOmron) — walk-through training (100 steps) with Qwen3VL-OFT.
# Run from the repo root inside the `starVLA` conda env.
set -euo pipefail

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
# export WANDB_MODE=disabled
export WANDB_API_KEY=${WANDB_API_KEY:-943ecb8d26fc2b3879cbc2d667414974906aebb9}

# Activate conda env if not already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "starVLA" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate starVLA
fi

# DeepSpeed needs a real nvcc on PATH; the user's ~/.local/bin/nvcc is a stub.
if [[ -x /cm/shared/apps/cuda12.2/toolkit/12.2.2/bin/nvcc ]]; then
  export CUDA_HOME=/cm/shared/apps/cuda12.2/toolkit/12.2.2
  export PATH=${CUDA_HOME}/bin:${PATH}
fi

# How many GPUs to use; falls back to "all visible".
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

run_root_dir=./playground/Checkpoints
run_id=robocasa365_qwenoft_OpenDrawer
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robocasa_365/train_files/starvla_qwenoft_robocasa365.yaml \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_robocasa365 \
  --wandb_entity jinhuiye \
