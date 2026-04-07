#!/bin/bash
#SBATCH --account=vonneumann1
#SBATCH --partition=vonneumann
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --job-name=oxe_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#
# Usage (single-node):
#   sbatch examples/SimplerEnv/train_files/sbatch_oxe_train.sh
#
# Usage (multi-node, e.g. 2 nodes × 4 GPUs = 8 GPUs):
#   sbatch --nodes=2 examples/SimplerEnv/train_files/sbatch_oxe_train.sh
#
# Override GPU count per node:
#   sbatch --gpus-per-node=8 examples/SimplerEnv/train_files/sbatch_oxe_train.sh
#
set -e

# === Conda setup ===
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate starVLA

# === CUDA setup ===
for cuda_path in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.4; do
  if [ -x "${cuda_path}/bin/nvcc" ]; then
    export CUDA_HOME="${cuda_path}"
    export PATH="${cuda_path}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_path}/lib64:${LD_LIBRARY_PATH:-}"
    break
  fi
done

# nvcc wrapper fallback
if ! nvcc --version 2>&1 | grep -q "release"; then
  _WRAPPER_DIR="${CONDA_PREFIX}/cuda_compat/bin"
  mkdir -p "${_WRAPPER_DIR}" 2>/dev/null || true
  _TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.4")
  _MAJOR=$(echo "${_TORCH_CUDA_VER}" | cut -d. -f1)
  _MINOR=$(echo "${_TORCH_CUDA_VER}" | cut -d. -f2)
  cat > "${_WRAPPER_DIR}/nvcc" << NVCC_EOF
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${_MAJOR}.${_MINOR}, V${_TORCH_CUDA_VER}"
NVCC_EOF
  chmod +x "${_WRAPPER_DIR}/nvcc"
  export PATH="${_WRAPPER_DIR}:${PATH}"
  export CUDA_HOME="${CONDA_PREFIX}/cuda_compat"
  echo "[INFO] Created nvcc wrapper: CUDA ${_TORCH_CUDA_VER}"
fi

echo "[INFO] CUDA_HOME=$CUDA_HOME"
nvcc --version 2>/dev/null || echo "[WARN] nvcc not found"

# === NCCL ===
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === Training config ===
cd /home/jye624/Projcets/starVLA

Framework_name=CosmoPredict2OFT
freeze_module_list=''
base_vlm=/home/jye624/Models/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
oxe_data_root=/home/jye624/Datasets
data_mix=bridge_rt_1
run_root_dir=./results/Checkpoints
run_id=0408_oxe_${data_mix}_${Framework_name}
per_device_batch_size=8
###########################################################################################

export WANDB_API_KEY=${WANDB_API_KEY:-943ecb8d26fc2b3879cbc2d667414974906aebb9}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir} logs/
cp $0 ${output_dir}/

# Auto-detect single-node vs multi-node from SLURM allocation
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
attn_implementation=sdpa
accelerate_config_file=starVLA/config/deepseeds/deepspeed_zero2.yaml

# Multi-node: need a fixed port for cross-node communication
# Single-node: use 0 to auto-select and avoid conflicts
if [ "$NNODES" -gt 1 ]; then
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  MASTER_PORT=${MAIN_PROCESS_PORT:-$((29500 + SLURM_JOB_ID % 1000))}
else
  MASTER_ADDR="127.0.0.1"
  MASTER_PORT=${MAIN_PROCESS_PORT:-0}
fi

echo "=============================="
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Nodes:        ${SLURM_NODELIST} (${NNODES} nodes)"
echo "GPUs/node:    ${GPUS_PER_NODE}"
echo "Total GPUs:   ${TOTAL_GPUS}"
echo "Batch/GPU:    ${per_device_batch_size}"
echo "Framework:    ${Framework_name}"
echo "Run ID:       ${run_id}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "=============================="

sg vonneumann1 -c "
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh && \
conda activate starVLA && \
accelerate launch \
  --config_file ${accelerate_config_file} \
  --num_processes ${TOTAL_GPUS} \
  --num_machines ${NNODES} \
  --machine_rank \${SLURM_PROCID:-0} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.vla_data.video_backend torchvision_av \
  --framework.qwenvl.attn_implementation ${attn_implementation} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_simplerEnv \
  --wandb_entity jinhuiye
"
