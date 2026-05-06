#!/usr/bin/env bash
# RoboChallenge Table30v2 — single-task walk-through training (UR5 / shred_paper)
# with Qwen3VL-OFT. Run from the repo root inside the `starVLA` conda env.
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "starVLA_dev" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate starVLA_dev
fi

# DeepSpeed needs a real nvcc. The login-node stub at $HOME/.local/bin/nvcc is
# not enough — point CUDA_HOME at the cluster CUDA 12.2 toolkit and prepend its
# bin to PATH.
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda12.2/toolkit/12.2.2}
export PATH=${CUDA_HOME}/bin:${PATH}

# How many GPUs to use; defaults to all visible.
NUM_GPUS=${NUM_GPUS:-$(python -c "import torch;print(torch.cuda.device_count())")}

# ---- training knobs (edit here) ----
TASK=shred_paper
BATCH=${BATCH:-2}
MAX_STEPS=${MAX_STEPS:-100}
SAVE_EVERY=${SAVE_EVERY:-100}
EVAL_EVERY=${EVAL_EVERY:-1000}
LOG_EVERY=${LOG_EVERY:-5}

run_root_dir=./playground/Checkpoints
run_id=robochallenge_table30v2_qwenoft_${TASK}_${MAX_STEPS}step
output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

# Disable WandB for the walk-through; remove this line and `wandb login` for real runs.
export WANDB_MODE=${WANDB_MODE:-disabled}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/RoboChallenge_table30v2/train_files/starvla_qwenoft_robochallenge_table30v2.yaml \
  --datasets.vla_data.per_device_batch_size "${BATCH}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_EVERY}" \
  --trainer.logging_frequency "${LOG_EVERY}" \
  --trainer.eval_interval "${EVAL_EVERY}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_robochallenge_table30v2
