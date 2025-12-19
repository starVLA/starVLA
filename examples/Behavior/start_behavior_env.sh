#!/bin/bash

# Debug: 输出当前使用的 Python 环境
echo "Using Python: $(which python)"

# 设置必要的环境变量
export star_vla_python=/data/wzx/conda_env/starVLA/bin/python
export sim_python=/data/wzx/behavior/bin/python
export BEHAVIOR_PATH=/data/wzx/behavior_evaluation/behavior/Datasets/BEHAVIOR_challenge
export PYTHONPATH=$(pwd):${PYTHONPATH}

# 配置模型路径和端口
MODEL_PATH="./results/Checkpoints/1007_qwenLargefm/checkpoints/steps_20000_pytorch_model.pt"
PORT=10197
WRAPPERS="RGBLowResWrapper"
USE_STATE=False  # 是否使用状态作为观察的一部分

# 配置任务名称
TASK_NAME="turning_on_radio"  # 选择一个简单的任务
LOG_FILE="./results/Checkpoints/1007_qwenLargefm/log_${TASK_NAME}.txt"

# 启动服务
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --use_bf16 > server_log.txt 2>&1 &

SERVER_PID=$!
sleep 15  # 等待服务器启动

# 检查服务器是否启动成功
if ps -p ${SERVER_PID} > /dev/null; then
    echo "✅ Server started successfully (PID: ${SERVER_PID})"
else
    echo "❌ Failed to start server"
    exit 1
fi

# 运行单个任务
echo "▶️ Running task '${TASK_NAME}'..."
CUDA_VISIBLE_DEVICES=0 ${sim_python} examples/Behavior/start_behavior_env.py \
    --ckpt-path ${MODEL_PATH} \
    --eval-on-train-instances True \
    --port ${PORT} \
    --task-name ${TASK_NAME} \
    --behaviro-data-path ${BEHAVIOR_PATH} \
    --wrappers ${WRAPPERS} \
    --use-state ${USE_STATE} > ${LOG_FILE} 2>&1

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