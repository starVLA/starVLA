#!/bin/bash

echo `which python`

export sim_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
#### set environment variables #####

#### get parameters #####
if [ -n "$1" ]; then
  MODEL_PATH="$1" # model path indict the output tree
else
  MODEL_PATH=./results/Checkpoints/1208_bridge_rt_1_Qwen3PI/final_model/pytorch_model.pt
fi

port=${2:-6678} # connect to your policy server port


#### build output directory #####
ckpt_path=${MODEL_PATH}
ckpt_dir=$(dirname "${ckpt_path}")
ckpt_base=$(basename "${ckpt_path}")
ckpt_name="${ckpt_base%.*}"

# Create output directories
output_server_dir="${ckpt_dir}/output_server"
output_eval_dir="${ckpt_dir}/output_eval"
mkdir -p "${output_server_dir}"
mkdir -p "${output_eval_dir}"
#### build output directory #####

TSET_NUM=1
# export DEBUG=1

IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#CUDA_DEVICES[@]} 

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  # StackGreenCubeOnYellowCubeBakedTexInScene-v0
  # PutCarrotOnPlateInScene-v0
  # PutSpoonOnTableClothInScene-v0
)

for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "▶️ Launching task [${env}] run#${run_idx}, log → ${task_log}"

    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &

    sleep 6

  done
done

declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
  # Path for log file
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "▶️ Launching V2 task [${env}] run#${run_idx}, log → ${task_log}"

    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py\
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &

    sleep 6
  done
done

# echo "✅ Finished"