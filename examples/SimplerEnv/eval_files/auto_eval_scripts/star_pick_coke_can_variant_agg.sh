# Environment setup
cd /mnt/petrelfs/yejinhui/Projects/llavavla
export starvla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starvlaSAM/bin/python
export sim_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=5500

MODEL_PATH=$1

# Optional: allow overriding via argument
if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH not provided as the first argument; using default"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1003_qwenfast/checkpoints/steps_10000_pytorch_model.pt"
fi

export ckpt_path=${MODEL_PATH}

# Helper to launch policy servers
policyserver_pids=()
eval_pids=()

task_name=pick_coke_va


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

# Helper to stop all services
stop_all_services() {
  # Wait for evaluation jobs to finish
  echo "⏳ Waiting for evaluation jobs to finish..."
  for pid in "${eval_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      wait "$pid"
      status=$?
      if [ $status -ne 0 ]; then
          echo "⚠️ Warning: evaluation job $pid exited abnormally (status: $status)"
      fi
    fi
  done

  # Stop policy servers
  echo "⏳ Stopping service processes..."
  for pid in "${policyserver_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    else
      echo "⚠️ Service process $pid no longer exists (may have exited early)"
    fi
  done
  # Reset PID arrays
  eval_pids=()
  policyserver_pids=()
  echo "✅ All services and jobs have stopped"
}


# Derive CUDA device list from CUDA_VISIBLE_DEVICES
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # Convert comma-separated GPU list into array
NUM_GPUS=${#CUDA_DEVICES[@]}  # Count available GPUs


declare -a arr=(${MODEL_PATH})

# Round-robin bookkeeping
total_gpus=8
run_count=0

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

# base setup
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
    # Launch service and capture its PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} &

    eval_pids+=($!)  # Track evaluation job PID
    run_count=$((run_count + 1))
  done
done

# table textures
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                      "Baked_sc1_staging_objaverse_cabinet2_h870")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
      # Launch service and capture its PID
      port=$((base_port + run_count))
      start_service ${gpu_id} ${ckpt_path} ${port}

      CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      
      eval_pids+=($!)  # Track evaluation job PID
      run_count=$((run_count + 1))
    done
  done
done

# distractors
env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
    # Launch service and capture its PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} &
    
    eval_pids+=($!)  # Track evaluation job PID
    run_count=$((run_count + 1))

    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
    # Launch service and capture its PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} distractor_config=more &
    
    eval_pids+=($!)  # Track evaluation job PID
    run_count=$((run_count + 1))
  done
done

# backgrounds
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a bg_scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                         "google_pick_coke_can_1_v4_alt_background_2")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${bg_scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
      # Launch service and capture its PID
      port=$((base_port + run_count))
      start_service ${gpu_id} ${ckpt_path} ${port}

      CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      
      eval_pids+=($!)  # Track evaluation job PID
      run_count=$((run_count + 1))
    done
  done
done

# lightings
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
    # Launch service and capture its PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True &
    
  eval_pids+=($!)  # Track evaluation job PID
    run_count=$((run_count + 1))

  gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
  # Launch service and capture its PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_brighter_lighting=True &
    
    eval_pids+=($!)  # Track evaluation job PID
    run_count=$((run_count + 1))
  done
done

# camera orientations
declare -a env_arr=("GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0" \
                    "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for env_name in "${env_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # Map to CUDA_VISIBLE_DEVICES GPU ID
      # Launch service and capture its PID
      port=$((base_port + run_count))
      start_service ${gpu_id} ${ckpt_path} ${port}
      CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      
      eval_pids+=($!)  # Track evaluation job PID
      run_count=$((run_count + 1))
    done
  done
done

# Wait for all background jobs to finish
stop_all_services
echo "✅ All evaluations completed"

