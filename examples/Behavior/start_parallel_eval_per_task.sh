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
    # B10
    "turning_on_radio"
    "picking_up_trash"
    "putting_away_Halloween_decorations"
    "cleaning_up_plates_and_food"
    "can_meat"
    "setting_mousetraps"
    "hiding_Easter_eggs"
    "picking_up_toys"
    "rearranging_kitchen_furniture"
    "putting_up_Christmas_decorations_inside"
    # B20
    "set_up_a_coffee_station_in_your_kitchen"
    "putting_dishes_away_after_cleaning"
    "preparing_lunch_box"
    "loading_the_car"
    "carrying_in_groceries"
    "bringing_in_wood"
    "moving_boxes_to_storage"
    "bringing_water"
    "tidying_bedroom"
    "outfit_a_basic_toolbox"
    # B30
    "sorting_vegetables"
    "collecting_childrens_toys"
    "putting_shoes_on_rack"
    "boxing_books_up_for_storage"
    "storing_food"
    "clearing_food_from_table_into_fridge"
    "assembling_gift_baskets"
    "sorting_household_items"
    "getting_organized_for_work"
    "clean_up_your_desk"
    # B40
    "setting_the_fire"
    "clean_boxing_gloves"
    "wash_a_baseball_cap"
    "wash_dog_toys"
    "hanging_pictures"
    "attach_a_camera_to_a_tripod"
    "clean_a_patio"
    "clean_a_trumpet"
    "spraying_for_bugs"
    "spraying_fruit_trees"
    # B50
    "make_microwave_popcorn"
    "cook_cabbage"
    "chop_an_onion"
    "slicing_vegetables"
    "chopping_wood"
    "cook_hot_dogs"
    "cook_bacon"
    "freeze_pies"
    "canning_food"
    "make_pizza"
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

# Define a function to stop the service
stop_service() {
  local pid=$1
  if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
    echo "⏹️ Stopping service (PID: ${pid})..."
    kill $pid
    wait $pid 2>/dev/null
    echo "✅ Service stopped"
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


# Parse TEST_EVAL_INSTANCE_IDS into an array
read -r -a INSTANCE_ID_ARRAY <<< "$TEST_EVAL_INSTANCE_IDS"
NUM_INSTANCES=${#INSTANCE_ID_ARRAY[@]}

echo "📊 Configuration:"
echo "   - Tasks to evaluate: ${#INSTANCE_NAMES[@]}"
echo "   - Instances per task: ${NUM_INSTANCES} (${TEST_EVAL_INSTANCE_IDS})"
echo "   - Available GPUs: ${NUM_GPUS} (${CUDA_DEVICES[@]})"
echo "   - Runs per task: ${TEST_NUM}"
echo ""

# Start Evaluating on test instances
# For each task, run all instances in parallel across GPUs, then wait before next task
for i in "${!INSTANCE_NAMES[@]}"; do
    task_name="${INSTANCE_NAMES[i]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📋 Processing task $((i+1))/${#INSTANCE_NAMES[@]}: ${task_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Arrays to track services and evaluations for this task
    declare -a task_service_pids=()
    declare -a task_service_ports=()
    declare -a task_eval_pids=()
    declare -a task_eval_logs=()
    
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"  # strip .pt or .bin suffix
    
    # Launch all instances for this task in parallel
    expected_launches=$((TEST_NUM * NUM_INSTANCES))
    successful_launches=0
    
    for ((run_idx=1; run_idx<=TEST_NUM; run_idx++)); do
      for instance_idx in "${!INSTANCE_ID_ARRAY[@]}"; do
        instance_id="${INSTANCE_ID_ARRAY[instance_idx]}"
        
        # Distribute across GPUs
        gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}
        
        tag="run${run_idx}_instance${instance_id}"
        task_log="${ckpt_dir}/${ckpt_name}_infer_test_${task_name}.log.${tag}"
        
        echo "▶️ Launching instance ${instance_id} for task ${task_name} (run#${run_idx}) on GPU $gpu_id"
        
        # Start the service and get the PID and port
        requested_port=$((base_port + run_count))
        service_info=$(start_service ${gpu_id} ${ckpt_path} ${requested_port})
        service_exit_code=$?
        
        if [ $service_exit_code -eq 0 ] && [ -n "$service_info" ]; then
          # Extract PID and port from the returned string (format: "PID:PORT")
          service_pid=$(echo $service_info | cut -d':' -f1)
          actual_port=$(echo $service_info | cut -d':' -f2)
          
          # Validate that we got both PID and port
          if [ -z "$service_pid" ] || [ -z "$actual_port" ]; then
            echo "❌ Invalid service info returned for instance ${instance_id}, skipping..."
            run_count=$((run_count + 1))
            continue
          fi
          
          task_service_pids+=($service_pid)
          task_service_ports+=($actual_port)
          
          echo "✅ Server started on port ${actual_port} (PID: ${service_pid}) for instance ${instance_id}"
          
          # Build command with single instance ID
          cmd="CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/Behavior/start_behavior_env.py \
            --ckpt-path ${ckpt_path} \
            --eval-on-train-instances False \
            --port ${actual_port} \
            --task-name ${task_name} \
            --behavior-tasks-jsonl-path ${TASKS_JSONL_PATH} \
            --behavior-asset-path ${BEHAVIOR_ASSET_PATH} \
            --wrappers ${WRAPPERS} \
            --use-state ${USE_STATE} \
            --eval-instance-ids \"${instance_id}\""
          
          # Run evaluation in background
          echo "🔄 Starting evaluation for ${task_name} instance ${instance_id} (run#${run_idx})..."
          eval "$cmd" > "${task_log}" 2>&1 &
          eval_pid=$!
          
          # Verify the background process started successfully
          if ! kill -0 $eval_pid 2>/dev/null; then
            echo "❌ Failed to start evaluation process for instance ${instance_id}"
            stop_service ${service_pid}
            run_count=$((run_count + 1))
            continue
          fi
          
          task_eval_pids+=($eval_pid)
          task_eval_logs+=("${task_log}")
          successful_launches=$((successful_launches + 1))
        else
          echo "❌ Failed to start server for instance ${instance_id} (exit code: ${service_exit_code}), skipping..."
        fi
        
        run_count=$((run_count + 1))
      done
    done
    
    echo ""
    echo "📊 Launch summary: ${successful_launches}/${expected_launches} evaluations launched successfully"
    
    # Wait for all evaluations for this task to complete
    echo ""
    echo "⏳ Waiting for all ${#task_eval_pids[@]} evaluations to complete for task: ${task_name}..."
    
    failed_count=0
    for eval_idx in "${!task_eval_pids[@]}"; do
      eval_pid=${task_eval_pids[$eval_idx]}
      eval_log=${task_eval_logs[$eval_idx]}
      
      if wait $eval_pid; then
        echo "✅ Evaluation completed successfully: ${eval_log}"
      else
        eval_exit_code=$?
        echo "❌ Evaluation failed (exit code ${eval_exit_code}): ${eval_log}"
        failed_count=$((failed_count + 1))
      fi
    done
    
    # Stop all services for this task
    echo ""
    echo "⏹️ Stopping all services for task: ${task_name}..."
    for service_pid in "${task_service_pids[@]}"; do
      stop_service ${service_pid}
    done
    
    echo ""
    if [ $failed_count -eq 0 ]; then
      echo "✅ All evaluations completed successfully for task: ${task_name}"
    else
      echo "⚠️ Completed task: ${task_name} with ${failed_count} failed evaluation(s)"
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All tests complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"