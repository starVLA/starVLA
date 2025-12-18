#!/bin/bash

echo `which python`
# Environment setup
cd /mnt/petrelfs/yejinhui/Projects/starVLA
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python
export sim_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=6350 

# export DEBUG=1


MODEL_PATH=$1
# MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/1120_bridge_rt_1_QwenDual_florence/checkpoints/steps_11000_pytorch_model.pt
TSET_NUM=4 # repeat each task 4 times
run_count=0

if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH not provided as the first argument; using default"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/1007_qwenLargefm/checkpoints/steps_20000_pytorch_model.pt"
fi

ckpt_path=${MODEL_PATH}

# Helper to launch policy servers
policyserver_pids=()
eval_pids=()



start_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local port=$3
  local server_log_dir="$(dirname "${ckpt_path}")/server_logs"
  local svc_log="${server_log_dir}/$(basename "${ckpt_path%.*}")_policy_server_${port}.log"
  mkdir -p "${server_log_dir}"

  # Pre-check the port and free it if already occupied
  if lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ Port ${port} is occupied; attempting to free it..."
    lsof -iTCP:"${port}" -sTCP:LISTEN -t | xargs kill -9
    sleep 2
    if lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null ; then
      echo "❌ Unable to free port ${port}; please investigate manually"
      exit 1
    else
      echo "✅ Port ${port} successfully freed"
    fi
  fi
  echo "▶️ Starting service on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &
  
  local pid=$!          # capture PID immediately
  policyserver_pids+=($pid)
  sleep 10
}

# Helper to stop every service
stop_all_services() {
  # Wait for every evaluation job to complete
  if [ "${#eval_pids[@]}" -gt 0 ]; then
    echo "⏳ Waiting for evaluation jobs to finish..."
    for pid in "${eval_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        wait "$pid"
        status=$?
    if [ $status -ne 0 ]; then
      echo "Warning: evaluation job $pid exited abnormally (status: $status)"
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
        echo "⚠️ Service process $pid no longer exists (might have exited early)"
      fi
    done
  fi

  # Clear PID arrays
  eval_pids=()
  policyserver_pids=()
  echo "✅ All services and tasks stopped"
}

# Retrieve CUDA_VISIBLE_DEVICES list on this host
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # convert comma-separated GPU list to an array
NUM_GPUS=${#CUDA_DEVICES[@]}  # count available GPUs



# Debug info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

# Task list: each entry defines an env-name
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
    
    # Launch service and capture the process ID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
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

# V2 variant: run PutEggplantInBasketScene-v0 five times as well
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
    gpu_id=${CUDA_DEVICES[$(((run_count) % NUM_GPUS))]}  # map to GPU ID in CUDA_VISIBLE_DEVICES
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching V2 task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"

  # Launch service and capture the process ID
    echo "server start run#${run_idx}"
    port=$((base_port + run_count))
    server_pid=$(start_service ${gpu_id} ${ckpt_path} ${port})

    echo "sim start run#${run_idx}"
    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
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
echo "✅ All evaluations finished"


