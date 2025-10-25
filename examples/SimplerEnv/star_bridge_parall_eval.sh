#!/bin/bash

echo `which python`
# Define environment
cd .
export star_vla_python=~/miniconda3/envs/starvla/bin/python
export sim_python=~/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=~/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=10097


MODEL_PATH=$1
TSET_NUM=4 # repeat each task 4 times
run_count=0

if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH not provided as the first argument, using default value"
  export MODEL_PATH="./StarVLA/Qwen-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt"
fi

ckpt_path=${MODEL_PATH}

# Define a function to start the service
policyserver_pids=()
eval_pids=()



start_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local port=$3
  local server_log_dir="$(dirname "${ckpt_path}")/server_logs"
  local svc_log="${server_log_dir}/$(basename "${ckpt_path%.*}")_policy_server_${port}.log"
  mkdir -p "${server_log_dir}"

  echo "▶️ Starting service on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &
  
  local pid=$!          # Capture the PID immediately
  policyserver_pids+=($pid)
  sleep 10
}

# Define a function to stop all services
stop_all_services() {
  # Wait for all evaluation tasks to finish
  if [ "${#eval_pids[@]}" -gt 0 ]; then
    echo "⏳ Waiting for evaluation tasks to finish..."
    for pid in "${eval_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        wait "$pid"
        status=$?
        if [ $status -ne 0 ]; then
            echo "Warning: evaluation task $pid exited abnormally (status: $status)"
        fi
      fi
    done
  fi

  # Stop all services
  if [ "${#policyserver_pids[@]}" -gt 0 ]; then
    echo "⏳ Stopping service processes..."
    for pid in "${policyserver_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
      else
        echo "⚠️ Service process $pid no longer exists (may have exited early)"
      fi
    done
  fi

  # Clear PID arrays
  eval_pids=()
  policyserver_pids=()
  echo "✅ All services and tasks have stopped"
}

# Get the CUDA_VISIBLE_DEVICES list
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # Convert the comma-separated GPU list into an array
NUM_GPUS=${#CUDA_DEVICES[@]}  # Number of available GPUs



# Debug info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

# Task list, each item is an env-name
declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)


for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}
    
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"  # strip .pt or .bin suffix

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"
    
    # Start the service and record its PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/start_simpler_env.py \
      --port $port \
      --ckpt-path ${ckpt_path} \
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
    
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# V2: PutEggplantInBasketScene-v0 also runs TSET_NUM times
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
    gpu_id=${CUDA_DEVICES[$(((run_count) % NUM_GPUS))]}  # Map to the GPU ID in CUDA_VISIBLE_DEVICES
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching V2 task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"

    # Start the service and record its PID
    echo "server start run#${run_idx}"
    port=$((base_port + run_count))
    server_pid=$(start_service ${gpu_id} ${ckpt_path} ${port})

    echo "sim start run#${run_idx}"
    ${sim_python} examples/SimplerEnv/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port $port \
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
      2>&1 | tee "${task_log}" &
    
    eval_pids+=($!)
    echo "sim end run#${run_idx}"
    
    run_count=$((run_count + 1))
  done
done



stop_all_services
wait
echo "✅ All tests complete"


