set -e

# === CUDA auto-detection for compute nodes ===
# Try common CUDA paths and set CUDA_HOME + PATH so DeepSpeed can find nvcc
if [ -z "$CUDA_HOME" ]; then
  for cuda_path in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.4; do
    if [ -x "${cuda_path}/bin/nvcc" ]; then
      export CUDA_HOME="${cuda_path}"
      export PATH="${cuda_path}/bin:${PATH}"
      export LD_LIBRARY_PATH="${cuda_path}/lib64:${LD_LIBRARY_PATH:-}"
      echo "[INFO] Auto-detected CUDA_HOME=${CUDA_HOME}"
      break
    fi
  done
fi

# Fallback: use conda env's nvcc wrapper (for clusters without system CUDA toolkit)
CONDA_NVCC_COMPAT="${CONDA_PREFIX:-$HOME/.conda/envs/starVLA}/cuda_compat/bin"
if ! nvcc --version 2>&1 | grep -q "release"; then
  if [ -x "${CONDA_NVCC_COMPAT}/nvcc" ]; then
    export PATH="${CONDA_NVCC_COMPAT}:${PATH}"
    echo "[INFO] Using nvcc wrapper from ${CONDA_NVCC_COMPAT}"
  fi
fi

# Final verify
if ! nvcc --version 2>&1 | grep -q "release"; then
  echo "[WARN] nvcc not found or not working. DeepSpeed may fail to import."
  echo "[WARN] Make sure you are on a compute node with CUDA."
fi

# Set Triton cache to local to avoid NFS slowdowns
export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" 2>/dev/null || true

if ip link show bond0 >/dev/null 2>&1; then
  export NCCL_SOCKET_IFNAME=bond0
fi

if [ -d /sys/class/infiniband ]; then
  ib_hca_list=$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//')
  if [ -n "${ib_hca_list}" ]; then
    export NCCL_IB_HCA=${ib_hca_list}
  fi
fi

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
cd /home/jye624/Projcets/starVLA

Framework_name=QwenOFT
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
per_device_batch_size=${PER_DEVICE_BATCH_SIZE:-8}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
accelerate_config_file=${ACCELERATE_CONFIG_FILE:-starVLA/config/deepseeds/deepspeed_zero2.yaml}


# Fix: ensure vonneumann1 group is active for NFS file access on compute nodes
# Worker processes spawned by accelerate/deepspeed may lose supplementary group context
if id -nG 2>/dev/null | grep -qw vonneumann1; then
  export _STARVLA_GROUP_FIX=vonneumann1
  echo "[INFO] Group vonneumann1 detected, using newgrp for NFS access"
fi

sg vonneumann1 -c "
accelerate launch \
  --config_file ${accelerate_config_file} \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
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
