#!/bin/bash
#SBATCH --job-name=test_RoCaPI         # name
#SBATCH -p si
#SBATCH -N 4                         # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=/mnt/petrelfs/yejinhui/Projects/starVLA/results/logs/%x-%j.out           # output file name
#SBATCH --error=/mnt/petrelfs/yejinhui/Projects/starVLA/results/logs/%x-%j.err
#SBATCH --exclude=SH-IDCA1404-10-140-54-13

#  6955707                 si     RoCaPI       reserved yejinhui       normal  R        9:14      4     gpu:32 SH-IDCA1404-10-140-54-[13,88-89,107]   

# source ~/.bashrc     # 确保 conda 命令可用
# source ~/.zshrc
# source ~/envs4jinhui.sh
# proxy_on

# conda activate llavavla310  # 替换为你的环境名

# export task_id=31_32_33

export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000


export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export NCCL_IB_HCA=mlx5_2,mlx5_3

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((RANDOM % 101 + 20000))


cd /mnt/petrelfs/yejinhui/Projects/starVLA
export PYTHONPATH="$PWD/starVLA/model/openvla:$PYTHONPATH"

# conda activate llavavla310
proxy_on

# <model_id/local_path_to_model,e.g,"CogACT/CogACT-Base">

# === Please modify the following paths according to your environment ===
###########################################################################################

export Framework_name=QwenPI_v2
export base_vlm=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct # must be a local path, due to simpler will run in other where
export base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
export freeze_module_list=""
export action_input_dim=2560
export DIT_TYPE="DiT-B"
export config_yaml=./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml
export data_mix=fourier_gr1_unified_1000
export include_state=True
export run_root_dir=./results/Checkpoints
export run_id=debug_1224_${data_mix}_${Framework_name}_nostate_qwen3
# === End of environment variable configuration ===
###########################################################################################


unset DEBUG

export TOTAL_GPUS=$((GPUS_PER_NODE * SLURM_NNODES))
export global_batch_size=$((TOTAL_GPUS * vla_per_device_batch_size)) # 512 is the default global batch size, you can change it if needed
echo "Total GPUs: $TOTAL_GPUS"


# export WANDB_MODE=disabled

export output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

  # --datasets.vla_data.include_state ${include_state} \

srun --jobid $SLURM_JOBID bash -c 'accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $SLURM_PROCID \
  --num_machines $SLURM_NNODES \
  --num_processes=${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.max_train_steps 200000 \
  --trainer.save_interval 10000 \
  --trainer.eval_interval 100 \
  --trainer.num_warmup_steps 5000 \
  --trainer.logging_frequency 50 \
  --trainer.learning_rate.base 3e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project StarVLA_Robocasa \
  --wandb_entity jinhuiye \
  --trainer.is_resume True '

