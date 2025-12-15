# Define environment
cd /mnt/petrelfs/yejinhui/Projects/llavavla
export starvla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starvlaSAM/bin/python
export sim_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=5500

MODEL_PATH=$1

# Optional: check whether MODEL_PATH argument is provided
if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH not provided as the first argument, using default value"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1003_qwenfast/checkpoints/steps_10000_pytorch_model.pt"
fi

export ckpt_path=${MODEL_PATH}

# Define a function to start the service
policyserver_pids=()
eval_pids=()


task_name=near_va


start_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local port=$3
  local server_log_dir="$(dirname "${ckpt_path}")/server_logs"
  local svc_log="${server_log_dir}/$(basename "${ckpt_path%.*}")_${task_name}_${port}.log"
  mkdir -p "${server_log_dir}"
  
  echo "▶️ Starting service on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${starvla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &
  
  local pid=$!          # Capture PID immediately
  policyserver_pids+=($pid)
  sleep 20
}

# Define a function to stop all services
stop_all_services() {
  # Wait for all evaluation tasks to complete
  echo "⏳ Waiting for evaluation tasks to complete..."
  for pid in "${eval_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      wait "$pid"
      status=$?
      if [ $status -ne 0 ]; then
          echo "Warning: Evaluation task $pid exited abnormally (status: $status)"
      fi
    fi
  done

  # Stop all services
  echo "⏳ Stopping service processes..."
  for pid in "${policyserver_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    else
      echo "⚠️ Service process $pid no longer exists (may have exited early)"
    fi
  done


  # Clear PID arrays
  eval_pids=()
  policyserver_pids=()
  echo "✅ All services and tasks stopped"
}

# Retrieve the CUDA_VISIBLE_DEVICES list from the current system
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # Convert comma-separated GPU list into array
NUM_GPUS=${#CUDA_DEVICES[@]}  # Get number of available GPUs



policy_model=Qwenpi

declare -a arr=(
  ${MODEL_PATH}
)

# Variables for round-robin allocation
total_gpus=8
run_count=0

# CogACT/CogACT-Large CogACT/CogACT-Small
for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

# base setup
env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${arr[@]}"; do
  gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  # 启动服务并获取服务进程的 PID
  port=$((base_port + run_count))
  start_service ${gpu_id} ${ckpt_path} ${port}

  CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
  eval_pids+=($!)
  run_count=$((run_count + 1))
done

# distractor
for ckpt_path in "${arr[@]}"; do
  gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  # 启动服务并获取服务进程的 PID
  port=$((base_port + run_count))
  start_service ${gpu_id} ${ckpt_path} ${port}
  CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs no_distractor=True &
  eval_pids+=($!)
  run_count=$((run_count + 1))
done

# backgrounds
env_name=MoveNearGoogleInScene-v0
declare -a scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                      "google_pick_coke_can_1_v4_alt_background_2")
for scene_name in "${scene_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}
      
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
    
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# lighting
env_name=MoveNearGoogleInScene-v0
scene_name=google_pick_coke_can_1_v4

for ckpt_path in "${arr[@]}"; do
  # Slightly darker
  gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  # 启动服务并获取服务进程的 PID
  port=$((base_port + run_count))
  start_service ${gpu_id} ${ckpt_path} ${port}
  CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs slightly_darker_lighting=True &
  
  eval_pids+=($!)
  run_count=$((run_count + 1))

  # Slightly brighter
  gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
  # 启动服务并获取服务进程的 PID
  port=$((base_port + run_count))
  start_service ${gpu_id} ${ckpt_path} ${port}

  CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
    --additional-env-build-kwargs slightly_brighter_lighting=True &
  
  eval_pids+=($!)
  run_count=$((run_count + 1))
done

# table textures
env_name=MoveNearGoogleInScene-v0
declare -a table_scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                            "Baked_sc1_staging_objaverse_cabinet2_h870")

for scene_name in "${table_scene_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
    
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# camera orientations
declare -a env_arr=("MoveNearAltGoogleCameraInScene-v0" \
                    "MoveNearAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for env_name in "${env_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 &
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# Wait for all background tasks to finish
stop_all_services
# wait
echo "✅ All tests completed"

