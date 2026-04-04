#!/bin/bash
# Smoke tests for VLA-only and VLA+VLM cotrain training after DataLoaderManager changes
# Usage: run on a compute node with 2+ GPUs
#   srun --jobid=<JOB_ID> --overlap --pty bash /home/jye624/Projcets/starVLA/tmp/run_train_test.sh
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


# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
cd /home/jye624/Projcets/starVLA

Framework_name=CosmoPredict2GR00T
freeze_module_list=''
base_vlm=/home/jye624/Models/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=/home/jye624/Datasets/LIBERO
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=1229_libero4in1_qwen3oft
# === End of environment variable configuration ===
###########################################################################################


export WANDB_MODE=disabled


output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-2}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
accelerate_config_file=${ACCELERATE_CONFIG_FILE:-starVLA/config/deepseeds/deepspeed_zero2.yaml}


# Fix: ensure vonneumann1 group is active for NFS file access on compute nodes
# Worker processes spawned by accelerate/deepspeed may lose supplementary group context
if id -nG 2>/dev/null | grep -qw vonneumann1; then
  export _STARVLA_GROUP_FIX=vonneumann1
  echo "[INFO] Group vonneumann1 detected, using newgrp for NFS access"
fi

# Resolve conda activation command for sub-shells (sg spawns a new shell)
CONDA_BASE=$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")
CONDA_INIT="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_DEFAULT_ENV:-starVLA}"

sg vonneumann1 -c "
${CONDA_INIT} && \
accelerate launch \
  --config_file ${accelerate_config_file} \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.future_action_window_size 7 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --trainer.vla_data.video_backend torchvision_av \
  --framework.qwenvl.attn_implementation ${attn_implementation} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero \
  --wandb_entity jinhuiye
"



##### Multi-Server Multi-GPU training script #####
  # accelerate launch \
  #   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  #   --main_process_ip $MASTER_ADDR \
  #   --main_process_port $MASTER_PORT \
  #   --machine_rank $SLURM_PROCID \
  #   --num_machines $SLURM_NNODES \
  #   --num_processes=${TOTAL_GPUS} \
  #   starVLA/training/train_starvla.py \
  #   --config_yaml ${config_yaml} \
  #   --framework.name ${Framework_name} \
  #   --framework.qwenvl.base_vlm ${base_vlm} \
  #   --run_root_dir ${run_root_dir} \
  #   --run_id ${run_id} \
  #   --wandb_project your_project \
  #   --wandb_entity your_name
##### Multi-Server Multi-GPU training script #####
