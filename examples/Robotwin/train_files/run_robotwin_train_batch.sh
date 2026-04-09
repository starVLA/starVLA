#!/bin/bash
#SBATCH --job-name=ebench_baseline
#SBATCH -p ebench_t
#SBATCH -N 6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/petrelfs/gaoning/trash/%x-%j.out
#SBATCH --error=/mnt/petrelfs/gaoning/trash/%x-%j.err
#SBATCH --exclude=HOST-10-140-66-29

set -e

# -------------------- NCCL / Networking --------------------
export NCCL_SOCKET_IFNAME=bond0
# For multi-node, list all available mlx5 interfaces for stability (adjust for your cluster)
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600

# -------------------- Required distributed environment --------------------
export GPUS_PER_NODE=8
export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + RANDOM % 10000))

echo "SLURM_NNODES=$SLURM_NNODES  GPUS_PER_NODE=$GPUS_PER_NODE  TOTAL_GPUS=$TOTAL_GPUS"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

# -------------------- Your original config --------------------
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin_all_50
run_id=0214_${data_mix}_abs_qwen3OFT_all

export HF_ENDPOINT=https://hf-mirror.com

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"

source /mnt/petrelfs/gaoning/miniconda3/bin/activate
conda activate starvla

# -------------------- Key: launch accelerate once per node --------------------
srun --jobid "$SLURM_JOBID" bash -c '
  set -e
  echo "Host=$(hostname)  SLURM_PROCID=$SLURM_PROCID"

  accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --main_process_ip '"$MASTER_ADDR"' \
    --main_process_port '"$MASTER_PORT"' \
    --machine_rank $SLURM_PROCID \
    --num_machines '"$SLURM_NNODES"' \
    --num_processes '"$TOTAL_GPUS"' \
    starVLA/training/train_starvla.py \
    --config_yaml '"$config_yaml"' \
    --framework.name '"$Framework_name"' \
    --framework.qwenvl.base_vlm '"$base_vlm"' \
    --datasets.vla_data.per_device_batch_size 4 \
    --datasets.vla_data.action_type abs_qpos \
    --datasets.vla_data.action_mode abs \
    --datasets.vla_data.data_mix '"$data_mix"' \
    --trainer.freeze_modules '"$freeze_module_list"' \
    --trainer.max_train_steps 150000 \
    --trainer.save_interval 10000 \
    --trainer.logging_frequency 50 \
    --trainer.eval_interval 1000 \
    --run_root_dir '"$run_root_dir"' \
    --run_id '"$run_id"' \
    --wandb_project starVLA_Robotwin \
    --wandb_entity axi-the-cat
'
