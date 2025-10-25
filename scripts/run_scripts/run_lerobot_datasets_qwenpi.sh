export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)


Framework_name=QwenPI
base_vlm=./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct # must be a local path, due to simpler will run in other where
freeze_module_list='' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""
DIT_TYPE="DiT-B"
# freeze_module_list="qwen_vl_interface.model.model.visual,dino_encoder" # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""

llavadata="asv2_conversation_en,asv2_detailed_description_en"
oxe_data_root=playground/Datasets/OXE_LEROBOT
data_mix=bridge_rt_1

run_root_dir=./playground/Checkpoints
run_id=1011_starvla_qwenpi

export action_input_dim=2048
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_contrain_oxe.yaml \
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


