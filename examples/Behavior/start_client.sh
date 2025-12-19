#!/bin/bash

# Debug: 输出当前使用的 Python 环境
echo "Using Python: $(which python)"

# 设置必要的环境变量
export star_vla_python=/gpfs/wangzixuan/conda_envs/starVLA/bin/python
export sim_python=/gpfs/wangzixuan/conda_envs/behavior/bin/python
export BEHAVIOR_PATH=/gpfs/wangzixuan/Jinhui/llavavla0/playground/Datasets/BEHAVIOR_challenge
export PYTHONPATH=$(pwd):${PYTHONPATH}

# Force Vulkan to use only the NVIDIA ICD to avoid duplicate ICDs seen by the loader
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Prefer NVIDIA GLX vendor when any GL deps are touched
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# 配置模型路径和端口
MODEL_PATH="/gpfs/wangzixuan/Jinhui/llavavla0/playground/Checkpoints/1031_BEHAVIOR_challenge_qwengroot/checkpoints/steps_40000_pytorch_model.pt"
PORT=10197
WRAPPERS="DefaultWrapper"
USE_STATE=False  # 是否使用状态作为观察的一部分

# 配置任务名称
TASK_NAME="turning_on_radio"  # 选择一个简单的任务
CLIENT_LOG_FILE="/gpfs/wangzixuan/Jinhui/llavavla0/playground/Checkpoints/1031_BEHAVIOR_challenge_qwengroot/client_log/log_${TASK_NAME}.txt"
SERVER_LOG_FILE="/gpfs/wangzixuan/Jinhui/llavavla0/playground/Checkpoints/1031_BEHAVIOR_challenge_qwengroot/server_log/log_${TASK_NAME}.txt"

# 启动服务
# echo "▶️ Starting server on port ${PORT}..."
# CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
#     --ckpt_path ${MODEL_PATH} \
#     --port ${PORT} \
#     --is_debug \
#     --use_bf16
    
#     #  > ${SERVER_LOG_FILE} 2>&1 &


# SERVER_PID=$!
# sleep 15  # 等待服务器启动

# 检查服务器是否启动成功
# if ps -p ${SERVER_PID} > /dev/null; then
#     echo "✅ Server started successfully (PID: ${SERVER_PID})"
# else
#     echo "❌ Failed to start server"
#     exit 1
# fi

# 运行单个任务
export DEBUG=true
echo "▶️ Running task '${TASK_NAME}'..."
CUDA_VISIBLE_DEVICES=6 ${sim_python} examples/Behavior/start_behavior_env.py \
    --ckpt-path ${MODEL_PATH} \
    --eval-instance-ids "0 1 2 3 4 5 6 7 8 9 10"  \
    --eval-on-train-instances True \
    --port ${PORT} \
    --task-name ${TASK_NAME} \
    --behaviro-data-path ${BEHAVIOR_PATH} \
    --wrappers ${WRAPPERS} \
    --use-state ${USE_STATE}
    
    #  > ${LOG_FILE} 2>&1

# 检查任务是否完成
if [ $? -eq 0 ]; then
    echo "✅ Task '${TASK_NAME}' completed successfully. Log: ${LOG_FILE}"
else
    echo "❌ Task '${TASK_NAME}' failed. Check log: ${LOG_FILE}"
fi

# 停止服务器
echo "⏹️ Stopping server (PID: ${SERVER_PID})..."
kill ${SERVER_PID}
wait ${SERVER_PID} 2>/dev/null
echo "✅ Server stopped"