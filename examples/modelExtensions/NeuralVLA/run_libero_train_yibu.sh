
# export HF_HOME=/mnt/petrelfs/share/yejinhui/Models/huggingface_cache

# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

# # For checkpoint save communication
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_TIMEOUT=1000  # Timeout set to 1 hour (unit: seconds)
MODEL_PATH=/workspace/model/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3 # must be a local path, due to simpler will run in other
data_root_dir=/workspace/dataset/libero_goal_no_noops_1.0.0_lerobot/datasets--IPEC-COMMUNITY--libero_goal_no_noops_1.0.0_lerobot/snapshots
run_root_dir=./playground/Checkpoints
run_id=1104_neurovla_gru_xiaonao_goal_dualimage_spike_multistep_ac8_768*2_yibu
export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

  # --pretrained_checkpoint ${MODEL_PATH} \

#export NCCL_SOCKET_IFNAME=bond0
#export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1,2,3
framework_name=NeuroVLA
dataset_py=lerobot_datasets
data_mix=libero_goal
action_chunk=4
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --main_process_port 29500 \
  starVLA/training/train_starvla.py\
  --config_yaml examples/modelExtensions/NeuralVLA/NeuroVLA_cotrain_custom.yaml \
  --framework.qwenvl.base_vlm ${MODEL_PATH} \
  --framework.name ${framework_name} \
  --framework.layer_qformer.num_query_tokens ${action_chunk} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.dataset_py ${dataset_py} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project spikeVLA-MLP \
  --wandb_entity weiyuguo \
  # --is_debug True


