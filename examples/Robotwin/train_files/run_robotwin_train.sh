#!/usr/bin/env bash

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name="${Framework_name:-QwenOFT}"
freeze_module_list="${FREEZE_MODULES:-}"
base_vlm="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
config_yaml="${CONFIG_YAML:-./examples/Robotwin/train_files/starvla_cotrain_robotwin_abs.yaml}"
run_root_dir="${RUN_ROOT_DIR:-./results/Checkpoints}"
data_mix="${DATA_MIX:-robotwin_all_50}"
run_id="${RUN_ID:-0129_${data_mix}_qwen3OFT_all}"
num_processes="${NUM_PROCESSES:-8}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-4}"
max_train_steps="${MAX_TRAIN_STEPS:-150000}"
save_interval="${SAVE_INTERVAL:-10000}"
logging_frequency="${LOGGING_FREQUENCY:-100}"
eval_interval="${EVAL_INTERVAL:-1000}"
# === End of environment variable configuration ===
###########################################################################################


# export WANDB_MODE=disabled

output_dir="${run_root_dir}/${run_id}"
mkdir -p "${output_dir}"
# mv this script to the output dir
cp "$0" "${output_dir}/"

EXTRA_ARGS=()

source examples/common/vlm_lora_args.sh
append_vlm_lora_args


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${max_train_steps}" \
  --trainer.save_interval "${save_interval}" \
  --trainer.logging_frequency "${logging_frequency}" \
  --trainer.eval_interval "${eval_interval}" \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project starVLA_Robotwin \
  --wandb_entity axi-the-cat \
  "${EXTRA_ARGS[@]}"
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
