#!/bin/bash

### MANUALLY SET THESE ###
# set the environment variables
export star_vla_python=/root/miniconda3/envs/starVLA/bin/python
export sim_python=/root/miniconda3/envs/behavior/bin/python
export TASKS_JSONL_PATH=/workspace/llavavla0/examples/Behavior/tasks.jsonl
export BEHAVIOR_ASSET_PATH=/workspace/llavavla0/BEHAVIOR-1K/datasets
export PYTHONPATH=$(pwd):${PYTHONPATH}

# set the eval parameters
MODEL_PATH=/workspace/llavavla0/playground/Checkpoints/BEHAVIOR-QwenDual-Pretrained-224/checkpoints/steps_300000_pytorch_model.pt
base_port=10197
WRAPPERS="DefaultWrapper" # DefaultWrapper, RGBLowResWrapper or RichObservationWrapper
USE_STATE=True # whether to use state as part of the observation
TEST_NUM=1 # only evaluate one time as specified by the rule, note only one video will be saved


# Configure which instances to evaluate
# Space-separated list, e.g., "0" or "0 1 2", up to 9. The rule requires using the first 10 instances for evaluation results
TEST_EVAL_INSTANCE_IDS="0 1 2 3 4 5 6 7"  

# Config which task to evaluate
declare -a INSTANCE_NAMES=(
    "turning_on_radio"
)
### END OF MANUALLY SETUP ###


# Start Evaluating on train instances
run_count=0

# Track used ports to avoid conflicts
declare -a used_ports=()

# Source the port utility function
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/port_utils.sh"

if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH not provided as the first argument, using default value"
  export MODEL_PATH="./results/Checkpoints/1007_qwenLargefm/checkpoints/steps_20000_pytorch_model.pt"
fi

ckpt_path=${MODEL_PATH}


# Define a function to start the service
start_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local base_port=$3
  local port=$(find_available_port $base_port)
  local server_log_dir="$(dirname "${ckpt_path}")/server_logs"
  local svc_log="${server_log_dir}/$(basename "${ckpt_path%.*}")_policy_server_${port}.log"
  mkdir -p "${server_log_dir}"

  echo "▶️ Starting service on GPU ${gpu_id}, port ${port} (requested: ${base_port})" >&2
  CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &
  
  local pid=$!          # Capture the PID immediately
  
  # Wait for server to be ready
  if wait_for_server $port; then
    echo "$pid:$port"  # Return PID and port separated by colon
  else
    echo "❌ Failed to start server on port ${port}" >&2
    return 1
  fi
}


# Get the CUDA_VISIBLE_DEVICES list
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # Convert the comma-separated GPU list into an array
NUM_GPUS=${#CUDA_DEVICES[@]}  # Number of available GPUs

# Handle case where CUDA_VISIBLE_DEVICES is not set
if [ $NUM_GPUS -eq 0 ]; then
  echo "⚠️ CUDA_VISIBLE_DEVICES not set, using default GPU 0"
  CUDA_DEVICES=(0)
  NUM_GPUS=1
fi



# Debug info
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"


# Start Evaluating on test instances
for task_name in "${INSTANCE_NAMES[@]}"; do   
  task_name=${task_name}
  for ((run_idx=1; run_idx<=TEST_NUM; run_idx++)); do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}

      ckpt_dir=$(dirname "${ckpt_path}")
      ckpt_base=$(basename "${ckpt_path}")
      ckpt_name="${ckpt_base%.*}"  # strip .pt or .bin suffix

      tag="run${run_idx}"
      task_log="${ckpt_dir}/${ckpt_name}_infer_test_${task_name}.log.${tag}"

      echo "▶️ Launching task [test_instances] run#${run_idx} on GPU $gpu_id, log → ${task_log}"

      # Start the service and get the actual port used
      requested_port=$((base_port + run_count))
      actual_port=$(start_service ${gpu_id} ${ckpt_path} ${requested_port})

      if [ $? -eq 0 ]; then
        echo "✅ Server started successfully on port ${actual_port}"
        
        # Build command with optional eval-instance-ids
        cmd="CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/Behavior/start_behavior_env.py \
          --ckpt-path ${ckpt_path} \
          --eval-on-train-instances False \
          --port ${actual_port} \
          --task-name ${task_name} \
          --behavior-tasks-jsonl-path ${TASKS_JSONL_PATH} \
          --behavior-asset-path ${BEHAVIOR_ASSET_PATH} \
          --wrappers ${WRAPPERS} \
          --use-state ${USE_STATE} \
          --eval-instance-ids \"${TEST_EVAL_INSTANCE_IDS}\""
        
        eval "$cmd" > "${task_log}" 2>&1 &

      else
        echo "❌ Failed to start server for test instances run#${run_idx}, skipping..."
      fi
      
      run_count=$((run_count + 1))
  done
done

wait
echo "✅ All tests complete"