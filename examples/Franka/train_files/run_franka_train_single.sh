#!/bin/bash
#SBATCH --job-name=tracevla_baseline           # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus-per-node=8        # number of GPUs per node(only valid under large/normal partition)
#SBATCH --cpus-per-task=224      # number of CPUs (28, 56, 112, 224 for 1, 2, 4, 8 GPUs)
#SBATCH --partition=vonneumann   # partition(preempt/large/normal/cpu) where you submit
#SBATCH --account=vonneumann1    # only require for multiple projects


module purge  # clear environment modules inherited from submission
module load slurm cuda12.2/toolkit/12.2.2 
source activate /home/zwanggk/.conda/envs/starVLA

cd '/project/vonneumann1/wzx/SMore/llavavla0'
echo $(pwd)

echo "========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "========================================="

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/Franka/train_files/starvla_cotrain_franka_single.yaml
run_root_dir=./results/Checkpoints
run_id=1221_${data_mix}_qwen3OFT
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/



srun accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_franka \
  --wandb_entity zwanggk \
  # --is_debug True



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

# module load slurm
# module load cuda12.2/toolkit/12.2.2