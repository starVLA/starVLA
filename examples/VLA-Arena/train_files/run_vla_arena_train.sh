#!/bin/bash
# run_vla_arena_train.sh
#
# Fine-tunes a starVLA model on VLA-Arena training data (LeRobot format).
#
# Before running:
#   1. Download datasets:  bash examples/VLA-Arena/data_preparation.sh /path/to/storage
#      (downloads VLA_Arena_L0_{S,M,L}_lerobot_openpi from HuggingFace)
#   2. Set the variables in the "User configuration" section below.
#
# Launch (single node, 8 GPUs):
#   bash examples/VLA-Arena/train_files/run_vla_arena_train.sh

# ---------------------------------------------------------------------------
# NCCL settings (adjust for your cluster network interface)
# ---------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# ---------------------------------------------------------------------------
# === User configuration – modify for your environment ===
# ---------------------------------------------------------------------------
Framework_name=QwenOFT          # QwenGR00T | QwenOFT | QwenPI | QwenFast
freeze_module_list=''            # e.g. 'vlm' to freeze the VLM backbone

base_vlm=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct
config_yaml=./examples/VLA-Arena/train_files/starvla_cotrain_vla_arena.yaml

# Root of your VLA-Arena LeRobot dataset (contains suite sub-directories)
vla_arena_data_root=playground/Datasets/VLA_ARENA_LEROBOT_DATA

# Which data mix to use (see starVLA/dataloader/gr00t_lerobot/mixtures.py)
#   vla_arena_L0_S        – small split
#   vla_arena_L0_M        – medium split
#   vla_arena_L0_L        – large split
data_mix=vla_arena_L0_L

run_root_dir=./results/Checkpoints
run_id=vla_arena_qwenoft_all
# ---------------------------------------------------------------------------

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# Archive this script for reproducibility
cp $0 ${output_dir}/

# ---------------------------------------------------------------------------
# Single-node launch (8 GPUs via accelerate + DeepSpeed ZeRO-2)
# ---------------------------------------------------------------------------
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${vla_arena_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_VLA_Arena \
  --wandb_entity your_wandb_entity
  # --is_debug True


# ---------------------------------------------------------------------------
# Multi-node example (uncomment and adapt for your cluster scheduler):
# ---------------------------------------------------------------------------
# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes ${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ${config_yaml} \
#   --framework.name ${Framework_name} \
#   --framework.qwenvl.base_vlm ${base_vlm} \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project starVLA_VLA_Arena \
#   --wandb_entity your_wandb_entity
