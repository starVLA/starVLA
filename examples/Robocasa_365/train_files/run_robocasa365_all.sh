#!/usr/bin/env bash
# RoboCasa365 (PandaOmron) — full target/human training (QwenOFT).
# Trains on ALL 50 target/human LeRobot tasks (18 atomic + 32 composite) under
# the `robocasa365_target_human_all` named mixture.
#
# Run from the repo root inside the `starVLA` conda env, AFTER you have run
#   bash examples/Robocasa_365/train_files/download_target_human.sh
set -euo pipefail

# ==============================================================
#  Platform-specific settings — adjust these for YOUR cluster
# ==============================================================
# WandB — set your own key, or run `wandb login` before launching,
# or uncomment the line below to disable WandB entirely:
#   export WANDB_MODE=disabled
export WANDB_API_KEY=<your_wandb_api_key>

# NCCL networking — uncomment and edit to match your InfiniBand / RoCE setup:
#   export NCCL_SOCKET_IFNAME=<your_network_interface>   # e.g. eth0, bond0
#   export NCCL_IB_HCA=<your_ib_devices>                # e.g. mlx5_0,mlx5_1
#   export NCCL_BLOCKING_WAIT=1
#   export NCCL_ASYNC_ERROR_HANDLING=1
#   export NCCL_TIMEOUT=1000

# CUDA / nvcc — DeepSpeed requires a real nvcc; point to your toolkit if needed:
#   export CUDA_HOME=/path/to/cuda                       # e.g. /usr/local/cuda-12.2
#   export PATH=${CUDA_HOME}/bin:${PATH}
# ==============================================================

# Activate conda env if not already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "starVLA" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate starVLA
fi

# How many GPUs to use; falls back to "all visible".
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

# ---- training knobs (edit here) ----
MIXTURE=robocasa365_target_human_all   # also: robocasa365_atomic_target_human_all / robocasa365_composite_target_human_all
BATCH=8
MAX_STEPS=200000
SAVE_EVERY=10000
EVAL_EVERY=2000
LOG_EVERY=100

run_root_dir=./playground/Checkpoints
run_id=robocasa365_qwenoft_${MIXTURE}
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
