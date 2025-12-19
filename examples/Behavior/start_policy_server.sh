#!/bin/bash

# Debug: 输出当前使用的 Python 环境
echo "Using Python: $(which python)"

# 设置必要的环境变量
export star_vla_python=/data/wzx/conda_env/starVLA/bin/python
export sim_python=/data/wzx/behavior/bin/python
export BEHAVIOR_PATH=/data/wzx/behavior_evaluation/behavior/Datasets/BEHAVIOR_challenge
export PYTHONPATH=$(pwd):${PYTHONPATH}

# 配置模型路径和端口
MODEL_PATH="/data/wzx/behavior_evaluation/behavior/playground/Pretrained_models/Qwen3-VL-GR00T-Behavior-nostate/checkpoints/steps_20000_pytorch_model.pt"
PORT=10197
WRAPPERS="DefaultWrapper"
USE_STATE=False  # 是否使用状态作为观察的一部分

# 配置任务名称
TASK_NAME="turning_on_radio"  # 选择一个简单的任务
LOG_FILE="/data/wzx/behavior_evaluation/behavior/playground/Pretrained_models/Qwen3-VL-GR00T-Behavior-nostate/checkpoints/client_logs/log_${TASK_NAME}.txt"
SERVER_LOG_FILE="/data/wzx/behavior_evaluation/behavior/playground/Pretrained_models/Qwen3-VL-GR00T-Behavior-nostate/checkpoints/server_logs/log_${TASK_NAME}.txt"

# 启动服务
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --is_debug \
    --use_bf16
    
    #  > ${SERVER_LOG_FILE} 2>&1 &


# SERVER_PID=$!
# sleep 15  # 等待服务器启动

# 检查服务器是否启动成功
if ps -p ${SERVER_PID} > /dev/null; then
    echo "✅ Server started successfully (PID: ${SERVER_PID})"
else
    echo "❌ Failed to start server"
    exit 1
fi
