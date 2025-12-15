
# 定义环境
cd /mnt/petrelfs/yejinhui/Projects/llavavla
export starvla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starvlaSAM/bin/python
export sim_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=5500

MODEL_PATH=$1

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1003_qwenfast/checkpoints/steps_10000_pytorch_model.pt"
fi

export ckpt_path=${MODEL_PATH}

# 定义一个函数来启动服务
policyserver_pids=()
eval_pids=()

task_name=near_vm


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
  
  local pid=$!          # 立即捕获正确 PID
  policyserver_pids+=($pid)
  sleep 20
}

# 定义一个函数来停止所有服务
stop_all_services() {
  # 等待所有评估任务完成
  echo "⏳ 等待评估任务完成..."
  for pid in "${eval_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      wait "$pid"
      status=$?
      if [ $status -ne 0 ]; then
          echo "警告: 评估任务 $pid 异常退出 (状态: $status)"
      fi
    fi
  done

  # 停止所有服务
  echo "⏳ 停止服务进程..."
  for pid in "${policyserver_pids[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
    else
      echo "⚠️ 服务进程 $pid 已不存在 (可能已提前退出)"
    fi
  done


  # 清空 PID 数组
  eval_pids=()
  policyserver_pids=()
  echo "✅ 所有服务和任务已停止"
}

# 获取当前系统的 CUDA_VISIBLE_DEVICES 列表
IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # 将逗号分隔的 GPU 列表转换为数组
NUM_GPUS=${#CUDA_DEVICES[@]}  # 获取可用 GPU 的数量


policy_model=Qwenpi

declare -a arr=(
  ${MODEL_PATH}
)
env_name=MoveNearGoogleBakedTexInScene-v0
# env_name=MoveNearGoogleBakedTexInScene-v1
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

# 轮转分配用的变量
total_gpus=8
run_count=0

for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

for urdf_version in "${urdf_version_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 60 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
      --additional-env-build-kwargs urdf_version=${urdf_version} \
      --additional-env-save-tags baked_except_bpb_orange &
    
    eval_pids+=($!)
    run_count=$((run_run_count + 1))
  done
done

stop_all_services
# wait
echo "✅ 所有测试完成"

