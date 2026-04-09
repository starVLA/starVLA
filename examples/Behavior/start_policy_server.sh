#!/bin/bash

# Debug: Print the current Python environment
echo "Using Python: $(which python)"

# Set necessary environment variables
export star_vla_python=/data/wzx/conda_env/starVLA/bin/python
export sim_python=/data/wzx/behavior/bin/python
export BEHAVIOR_PATH=/data/wzx/behavior_evaluation/behavior/Datasets/BEHAVIOR_challenge
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Configure model path and port
MODEL_PATH="/data/wzx/behavior_evaluation/behavior/playground/Pretrained_models/Qwen3-VL-GR00T-Behavior-nostate/checkpoints/steps_20000_pytorch_model.pt"
PORT=10197
WRAPPERS="DefaultWrapper"
USE_STATE=False  # Whether to use state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  # Choose a simple task
LOG_FILE="/data/wzx/behavior_evaluation/behavior/playground/Pretrained_models/Qwen3-VL-GR00T-Behavior-nostate/checkpoints/client_logs/log_${TASK_NAME}.txt"
SERVER_LOG_FILE="/data/wzx/behavior_evaluation/behavior/playground/Pretrained_models/Qwen3-VL-GR00T-Behavior-nostate/checkpoints/server_logs/log_${TASK_NAME}.txt"

# Start server
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --is_debug \
    --use_bf16
    
    #  > ${SERVER_LOG_FILE} 2>&1 &


# SERVER_PID=$!
# sleep 15  # Wait for server to start

# Check if server started successfully
if ps -p ${SERVER_PID} > /dev/null; then
    echo "✅ Server started successfully (PID: ${SERVER_PID})"
else
    echo "❌ Failed to start server"
    exit 1
fi
