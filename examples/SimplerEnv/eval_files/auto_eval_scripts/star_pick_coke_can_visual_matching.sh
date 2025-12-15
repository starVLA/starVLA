# 4 (URDF variants) × 3 (coke_can options) × 1 (model path) = 12 runs


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
  echo "❌ MODEL_PATH not provided as the first argument; falling back to default"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1003_qwenfast/checkpoints/steps_10000_pytorch_model.pt"
fi

export ckpt_path=${MODEL_PATH}

# Helper to launch policy servers
policyserver_pids=()
eval_pids=()


task_name=pick_coke_vm


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
  
  local pid=$!          # capture PID immediately
  policyserver_pids+=($pid)
  sleep 20
}

# Helper to stop every service
stop_all_services() {
  # Wait for every evaluation job to complete
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

  # Stop all services
  echo "⏳ Stopping service processes..."
  for pid in "${policyserver_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    else
      echo "⚠️ Service process $pid no longer exists (might have exited early)"
    fi
  done


  # Clear PID arrays
  eval_pids=()
  policyserver_pids=()
  echo "✅ All services and tasks stopped"
}


# Retrieve CUDA_VISIBLE_DEVICES list on this host
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # convert comma-separated GPU list to an array
NUM_GPUS=${#CUDA_DEVICES[@]}  # count available GPUs



policy_model=Qwenpi

declare -a arr=(${MODEL_PATH})

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png

# Variables for round-robin assignment
total_gpus=8
run_count=0

for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

for urdf_version in "${urdf_version_arr[@]}"; do
  for coke_can_option in "${coke_can_options_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # map to GPU ID in CUDA_VISIBLE_DEVICES
    # Launch service and capture the process ID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

      CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port $port \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} urdf_version=${urdf_version} &
      
      eval_pids+=($!)
      run_count=$((run_count + 1))
    done
  done
done

# Wait for all background jobs
stop_all_services
echo "✅ All evaluations finished"

