#!/usr/bin/env bash
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used to abort on hung collectives
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # 1 hour

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/DOMINO/train_files/starvla_train_domino.yaml
run_root_dir=./results/Checkpoints
# Available mixtures (see examples/DOMINO/train_files/data_registry/data_config.py):
#   domino                 - 35 tasks x (Clean_Dynamic + Random_Dynamic)
#   domino_clean           - 35 tasks x Clean_Dynamic
#   domino_random          - 35 tasks x Random_Dynamic
#   domino_cotrain         - DOMINO dynamic + RoboTwin static co-training
data_mix=domino_clean
run_id=starvla_${data_mix}_qwen3OFT
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# keep a copy of the launch script next to the checkpoints
cp "$0" "${output_dir}/"


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 150000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_DOMINO \
  --wandb_entity your_wandb_entity
  # --is_debug True
