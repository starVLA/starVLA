#!/bin/bash

# too many thing want to do here, put them to diff .sh

echo `which python`
# Define environment
cd .
export star_vla_python=/data/wzx/conda_env/starVLA/bin/python
export sim_python=/data/wzx/behavior/bin/python
export BEHAVIOR_PATH=/data/wzx/behavior_evaluation/behavior/Datasets/BEHAVIOR_challenge
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=10197
WRAPPERS="RGBLowResWrapper"
MODEL_PATH=$1
TSET_NUM=1 # only evaluate one time as specified by the rule, note only one video will be saved
run_count=0

USE_STATE=False # whether to use state as part of the observation

# Configure which instances to evaluate
TRAIN_EVAL_INSTANCE_IDS="0"  # Space-separated list, e.g., "0" or "0 1 2" or "" for all instances #TODO
TEST_EVAL_INSTANCE_IDS="0 1 2 3 4 5 6 7 8 9"     # Space-separated list, e.g., "0" or "0 1 2" or "" for all instances

# Track used ports to avoid conflicts
declare -a used_ports=()

if [ -z "$MODEL_PATH" ]; then
  echo "❌ MODEL_PATH not provided as the first argument, using default value"
  export MODEL_PATH="./results/Checkpoints/1007_qwenLargefm/checkpoints/steps_20000_pytorch_model.pt"
fi

ckpt_path=${MODEL_PATH}

# Define a function to find an available port
find_available_port() {
  local base_port=$1
  local port=$base_port
  
  # First check our internal tracking
  while [[ " ${used_ports[@]} " =~ " ${port} " ]]; do
    port=$((port + 1))
  done
  
  # Then check system ports if tools are available
  if command -v netstat >/dev/null 2>&1; then
    while netstat -tuln 2>/dev/null | grep -q ":$port "; do
      port=$((port + 1))
    done
  elif command -v ss >/dev/null 2>&1; then
    while ss -tuln 2>/dev/null | grep -q ":$port "; do
      port=$((port + 1))
    done
  elif command -v lsof >/dev/null 2>&1; then
    while lsof -i :$port >/dev/null 2>&1; do
      port=$((port + 1))
    done
  else
    # Fallback: just increment port and hope for the best
    echo "⚠️ Warning: No port checking tools available, using port ${port}" >&2
  fi
  
  # Add to our tracking
  used_ports+=($port)
  echo $port
}

# Define a function to check if server is ready
wait_for_server() {
  local port=$1
  local max_attempts=30
  local attempt=0
  
  echo "⏳ Waiting for server on port ${port} to be ready..." >&2
  while [ $attempt -lt $max_attempts ]; do
    # Try different methods to check if port is in use
    local port_in_use=false
    
    # Method 1: Try to connect to the port (most reliable)
    if timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
      port_in_use=true
    # Method 2: Use system tools if available
    elif command -v netstat >/dev/null 2>&1; then
      if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        port_in_use=true
      fi
    elif command -v ss >/dev/null 2>&1; then
      if ss -tuln 2>/dev/null | grep -q ":$port "; then
        port_in_use=true
      fi
    elif command -v lsof >/dev/null 2>&1; then
      if lsof -i :$port >/dev/null 2>&1; then
        port_in_use=true
      fi
    fi
    
    if [ "$port_in_use" = true ]; then
      echo "✅ Server on port ${port} is ready" >&2
      return 0
    fi
    
    sleep 2
    attempt=$((attempt + 1))
  done
  
  echo "❌ Server on port ${port} failed to start after $((max_attempts * 2)) seconds" >&2
  return 1
}

# Define a function to start the service
policyserver_pids=()
eval_pids=()

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
  policyserver_pids+=($pid)
  
  # Wait for server to be ready
  if wait_for_server $port; then
    echo $port  # Return only the port number
  else
    echo "❌ Failed to start server on port ${port}" >&2
    return 1
  fi
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

# Start Evaluating on train instances
declare -a TRAIN_INSTANCE_NAMES=(
  "turning_on_radio"
  # "picking_up_trash"
)

# declare -a TRAIN_INSTANCE_NAMES=(
#     # B10
#     "turning_on_radio"
#     "picking_up_trash"
#     "putting_away_Halloween_decorations"
#     "cleaning_up_plates_and_food"
#     "can_meat"
#     "setting_mousetraps"
#     "hiding_Easter_eggs"
#     "picking_up_toys"
#     "rearranging_kitchen_furniture"
#     "putting_up_Christmas_decorations_inside"
#     # B20
#     "set_up_a_coffee_station_in_your_kitchen"
#     "putting_dishes_away_after_cleaning"
#     "preparing_lunch_box"
#     "loading_the_car"
#     "carrying_in_groceries"
#     "bringing_in_wood"
#     "moving_boxes_to_storage"
#     "bringing_water"
#     "tidying_bedroom"
#     "outfit_a_basic_toolbox"
#     # B30
#     "sorting_vegetables"
#     "collecting_childrens_toys"
#     "putting_shoes_on_rack"
#     "boxing_books_up_for_storage"
#     "storing_food"
#     "clearing_food_from_table_into_fridge"
#     "assembling_gift_baskets"
#     "sorting_household_items"
#     "getting_organized_for_work"
#     "clean_up_your_desk"
#     # B40
#     "setting_the_fire"
#     "clean_boxing_gloves"
#     "wash_a_baseball_cap"
#     "wash_dog_toys"
#     "hanging_pictures"
#     "attach_a_camera_to_a_tripod"
#     "clean_a_patio"
#     "clean_a_trumpet"
#     "spraying_for_bugs"
#     "spraying_fruit_trees"
#     # B50
#     "make_microwave_popcorn"
#     "cook_cabbage"
#     "chop_an_onion"
#     "slicing_vegetables"
#     "chopping_wood"
#     "cook_hot_dogs"
#     "cook_bacon"
#     "freeze_pies"
#     "canning_food"
#     "make_pizza"
# )

for i in "${!TRAIN_INSTANCE_NAMES[@]}"; do
    task_name="${TRAIN_INSTANCE_NAMES[i]}"
    for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
        gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}
        
        ckpt_dir=$(dirname "${ckpt_path}")
        ckpt_base=$(basename "${ckpt_path}")
        ckpt_name="${ckpt_base%.*}"  # strip .pt or .bin suffix

        tag="run${run_idx}"
        task_log="${ckpt_dir}/${ckpt_name}_infer_${task_name}.log.${tag}"

        echo "▶️ Launching task [${task_name}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"
        
        # Start the service and get the actual port used
        requested_port=$((base_port + run_count))
        actual_port=$(start_service ${gpu_id} ${ckpt_path} ${requested_port})
        
        if [ $? -eq 0 ]; then
          echo "✅ Server started successfully on port ${actual_port}"
          
          # Build command with optional eval-instance-ids
          cmd="CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/Behavior/start_behavior_env.py \
            --ckpt-path ${ckpt_path} \
            --eval-on-train-instances True \
            --port ${actual_port} \
            --task-name ${task_name} \
            --behaviro-data-path ${BEHAVIOR_PATH} \
            --wrappers ${WRAPPERS} \
            --use-state ${USE_STATE}"
          
          # Add eval-instance-ids if specified
          if [ -n "$TRAIN_EVAL_INSTANCE_IDS" ]; then
            cmd="$cmd --eval-instance-ids $TRAIN_EVAL_INSTANCE_IDS"
          fi
          
          eval "$cmd" > "${task_log}" 2>&1 &
          
          eval_pids+=($!)
        else
          echo "❌ Failed to start server for task ${task_name} run#${run_idx}, skipping..."
        fi
        
        run_count=$((run_count + 1))
    done
done

# Start Evaluating on test instances
for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
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
        --behaviro-data-path ${BEHAVIOR_PATH} \
        --wrappers ${WRAPPERS} \
        --use-state ${USE_STATE}"
      
      # Add eval-instance-ids if specified
      if [ -n "$TEST_EVAL_INSTANCE_IDS" ]; then
        cmd="$cmd --eval-instance-ids $TEST_EVAL_INSTANCE_IDS"
      fi
      
      eval "$cmd" > "${task_log}" 2>&1 &

      eval_pids+=($!)
    else
      echo "❌ Failed to start server for test instances run#${run_idx}, skipping..."
    fi
    
    run_count=$((run_count + 1))
done


stop_all_services
wait
echo "✅ All tests complete"