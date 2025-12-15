# Use all 8 GPUs and run tasks in the background
# Total runs: 1 + 2 + 2 + 2 = 7 executions of main_inference.py.


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
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1017_QwenOFT/checkpoints/steps_65000_pytorch_model.pt"
fi

export ckpt_path=${MODEL_PATH}

# Helper to launch policy servers
policyserver_pids=()
eval_pids=()



task_name=putin_va


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

declare -a ckpt_paths=(
  ${MODEL_PATH}
)

declare -a env_names=(
  PlaceIntoClosedTopDrawerCustomInScene-v0
)

# Variables for round-robin assignment
total_gpus=8
run_count=0

EXTRA_ARGS="--enable-raytracing  --additional-env-build-kwargs model_ids=apple"

# base setup
scene_name=frl_apartment_stage_simple

EvalSim() {
  echo ${ckpt_path} ${env_name}
  # Launch service and capture the process ID
  port=$((base_port + run_count))
  start_service ${gpu_id} ${ckpt_path} ${port}

  CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port ${port} \
    --control-freq 3 --sim-freq 513 --max-episode-steps 200 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.65 0.65 1 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range -0.08 -0.02 3 --obj-init-y-range -0.02 0.08 3 \
    ${EXTRA_ARGS}
  
}

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}
    EvalSim &
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# backgrounds
declare -a scene_names=(
  "modern_bedroom_no_roof"
  "modern_office_no_roof"
)

for scene_name in "${scene_names[@]}"; do
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt model_ids=apple"
  gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # map to GPU ID in CUDA_VISIBLE_DEVICES
      EvalSim &
      eval_pids+=($!)
      run_count=$((run_count + 1))
    done
  done
done

# lightings
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter model_ids=apple"
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    EvalSim &
    eval_pids+=($!)
    run_count=$((run_count + 1))

    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker model_ids=apple"
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    EvalSim &
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# new cabinets
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2 model_ids=apple"
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    EvalSim &
    eval_pids+=($!)
    run_count=$((run_count + 1))

    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3 model_ids=apple"
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    EvalSim &
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# Wait for all background jobs
stop_all_services
echo "✅ All evaluations finished"


