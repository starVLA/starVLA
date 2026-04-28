#!/usr/bin/env bash
# RoboCasa365 (PandaOmron) — full target/human training (Qwen3VL-OFT).
# Trains on ALL 50 target/human LeRobot tasks (18 atomic + 32 composite) under
# the `robocasa365_target_human_all` named mixture.
#
# Run from the repo root inside the `starVLA` conda env, AFTER you have run
#   bash examples/Robocasa_365/train_files/download_target_human.sh
# Override knobs via env vars, e.g.
#   MIXTURE=robocasa365_atomic_target_human_all NUM_GPUS=4 \
#     bash examples/Robocasa_365/train_files/run_robocasa365_all.sh
set -euo pipefail

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
# export WANDB_MODE=disabled

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

# ---- knobs ----
MIXTURE=${MIXTURE:-robocasa365_target_human_all}        # also: ..._atomic_..., ..._composite_...
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}
BATCH=${BATCH:-8}
MAX_STEPS=${MAX_STEPS:-200000}
SAVE_EVERY=${SAVE_EVERY:-10000}
EVAL_EVERY=${EVAL_EVERY:-2000}
LOG_EVERY=${LOG_EVERY:-100}

run_root_dir=./playground/Checkpoints
run_id=${RUN_ID:-robocasa365_qwenoft_${MIXTURE}}
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robocasa_365/train_files/starvla_qwenoft_robocasa365.yaml \
  --datasets.vla_data.data_mix "${MIXTURE}" \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --trainer.eval_interval "${EVAL_EVERY}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_robocasa365
