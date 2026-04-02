#!/bin/bash
set -euo pipefail

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

###########################################################################################
# Please modify the following paths according to your environment
Framework_name=QwenOFT
freeze_module_list=''
base_vlm=/root/model/Qwen2.5-VL-3B-Instruct
config_yaml=./examples/Dual_XARM/train_files/starvla_dual_xarm_abs_cart.yaml
run_root_dir=./results/Checkpoints
data_mix=dual_xarm_pick_box_action_cart_20260311
run_id=$(date +%m%d)_${data_mix}_qwenoft_abs_cart
num_processes=1
# End of environment variable configuration
###########################################################################################

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
