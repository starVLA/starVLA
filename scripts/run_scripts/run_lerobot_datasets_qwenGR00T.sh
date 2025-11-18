export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)


# === Please modify the following paths according to your environment ===
###########################################################################################

Framework_name=QwenGR00T
base_vlm=StarVLA/Qwen3-VL-4B-Instruct-Action
action_input_dim=2560
DIT_TYPE="DiT-B"
oxe_data_root=playground/Datasets/OXE_LEROBOT
data_mix=bridge_rt_1
run_root_dir=./playground/Checkpoints
run_id=1004_starvla_qwengroot_oxe
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/
export WANDB_MODE=disabled

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_hidden_dim ${action_input_dim} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${oxe_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA \
  --wandb_entity jinhuiye \
  # --is_debug True




# multi-node launch example

# accelerate launch \
#   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
#   --main_process_ip $MASTER_ADDR \
#   --main_process_port $MASTER_PORT \
#   --machine_rank $SLURM_PROCID \
#   --num_machines $SLURM_NNODES \
#   --num_processes=${TOTAL_GPUS} \
#   starVLA/training/train_starvla.py \
#   --config_yaml ./starVLA/config/training/starvla_cotrain_oxe.yaml \
#   --framework.framework_py QwenGR00T \
#   --framework.qwenvl.base_vlm microsoft/Florence-2-large \
#   --run_root_dir ${run_root_dir} \
#   --run_id ${run_id} \
#   --wandb_project your_project \
#   --wandb_entity your_name

