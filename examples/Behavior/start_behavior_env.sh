#!/bin/bash

# Debug: Print the current Python environment
echo "Using Python: $(which python)"

# Set necessary environment variables
export star_vla_python=/data/wzx/conda_env/starVLA/bin/python
export sim_python=/data/wzx/behavior/bin/python
export BEHAVIOR_PATH=/data/wzx/behavior_evaluation/behavior/Datasets/BEHAVIOR_challenge
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Configure model path and port
MODEL_PATH="./results/Checkpoints/1007_qwenLargefm/checkpoints/steps_20000_pytorch_model.pt"
PORT=10197
WRAPPERS="RGBLowResWrapper"
USE_STATE=False  # Whether to use state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  # Choose a simple task
LOG_FILE="./results/Checkpoints/1007_qwenLargefm/log_${TASK_NAME}.txt"

# Start server
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --use_bf16 > server_log.txt 2>&1 &

SERVER_PID=$!
sleep 15  # Wait for server to start

# Check if server started successfully
if ps -p ${SERVER_PID} > /dev/null; then
    echo "✅ Server started successfully (PID: ${SERVER_PID})"
else
    echo "❌ Failed to start server"
    exit 1
fi

# Run a single task
echo "▶️ Running task '${TASK_NAME}'..."
CUDA_VISIBLE_DEVICES=0 ${sim_python} examples/Behavior/start_behavior_env.py \
    --ckpt-path ${MODEL_PATH} \
    --eval-on-train-instances True \
    --port ${PORT} \
    --task-name ${TASK_NAME} \
    --behaviro-data-path ${BEHAVIOR_PATH} \
    --wrappers ${WRAPPERS} \
    --use-state ${USE_STATE} > ${LOG_FILE} 2>&1

# Check if task completed
if [ $? -eq 0 ]; then
    echo "✅ Task '${TASK_NAME}' completed successfully. Log: ${LOG_FILE}"
else
    echo "❌ Task '${TASK_NAME}' failed. Check log: ${LOG_FILE}"
fi

# Stop server
echo "⏹️ Stopping server (PID: ${SERVER_PID})..."
kill ${SERVER_PID}
wait ${SERVER_PID} 2>/dev/null
echo "✅ Server stopped"