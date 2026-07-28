#!/usr/bin/env bash
# StarVLA Diffusion Policy smoke training for Realman RM-75.
#
# Run from the repository root after activating the StarVLA environment:
#   DATA_ROOT_DIR=/path/to/lerobot_datasets \
#     bash examples/realRobots/Realman/train_files/train_realman_dp.sh
#
# Override smoke defaults with NUM_PROCESSES, BATCH_SIZE, MAX_STEPS, RUN_ID,
# LOG_DIR, DATA_MIX, WANDB_ENTITY, or LOG_FILE.

set -euo pipefail

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

################################################################################
# Training defaults
Framework_name=DiffusionPolicy
freeze_module_list=""
config_yaml=./examples/realRobots/Realman/train_files/train_realman_dp.yaml
data_root_dir=${DATA_ROOT_DIR:-/path/to/lerobot_datasets}
data_mix=${DATA_MIX:-realman_example}
run_root_dir=${RUN_ROOT_DIR:-./results/Checkpoints}
run_id=${RUN_ID:-$(date +%m%d)_realman_dp_smoke}
################################################################################

# Set WANDB_MODE=disabled to run without online logging.
num_processes=${NUM_PROCESSES:-4}
batch_size=${BATCH_SIZE:-8}
max_steps=${MAX_STEPS:-5000}
wandb_entity=${WANDB_ENTITY:-your_wandb_entity}
log_dir=${LOG_DIR:-./results/logs}
mkdir -p "${log_dir}"
log_file=${LOG_FILE:-${log_dir}/dp_smoke_$(date +%Y%m%d_%H%M%S).log}

output_dir=${run_root_dir}/${run_id}
if [[ -d "${output_dir}" ]]; then
  has_ckpt=$(find "${output_dir}" \( -path "*/checkpoints/*" -o -path "*/final_model/*" -o -name "*_pytorch_model.pt" -o -name "*.safetensors" \) -print -quit)
  if [[ -n "${has_ckpt}" ]]; then
    echo "Refusing to overwrite existing checkpoint output: ${output_dir}" | tee -a "${log_file}"
    exit 1
  fi

  first_entry=$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)
  if [[ -z "${first_entry}" ]]; then
    rm -rf "${output_dir}"
  else
    echo "Refusing to reuse non-empty output directory without checkpoints: ${output_dir}" | tee -a "${log_file}"
    exit 1
  fi
fi

mkdir -p "${output_dir}"
# Archive this launch script next to the checkpoint for reproducibility.
cp "$0" "${output_dir}/"

echo "Logging to ${log_file}"
echo "Output dir ${output_dir}"

set +e
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${batch_size}" \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_steps}" \
  --trainer.save_interval 2500 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 999999 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_realman \
  --wandb_entity "${wandb_entity}" \
  2>&1 | tee -a "${log_file}"
train_rc=${PIPESTATUS[0]}
set -e

echo "Training command exit code: ${train_rc}" | tee -a "${log_file}"
exit "${train_rc}"

