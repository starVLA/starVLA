# 使用所有 8 个 GPU 并将任务挂到后台运行



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
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1_need/QwenGR00T/checkpoints/steps_30000_pytorch_model.pt"
fi

export ckpt_path=${MODEL_PATH}

# 定义一个函数来启动服务
policyserver_pids=()
eval_pids=()

task_name=drawer_va


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

declare -a ckpt_paths=(
  ${MODEL_PATH}
)
# CogACT/CogACT-Large CogACT/CogACT-Small
declare -a env_names=(
  CloseTopDrawerCustomInScene-v0
  CloseMiddleDrawerCustomInScene-v0
  CloseBottomDrawerCustomInScene-v0
  OpenTopDrawerCustomInScene-v0
  OpenMiddleDrawerCustomInScene-v0
  OpenBottomDrawerCustomInScene-v0
)

EXTRA_ARGS="--enable-raytracing"

# 使用 8 GPUs 轮转分配
total_gpus=8
run_count=0

# base setup
scene_name=frl_apartment_stage_simple

EvalSim() {
  echo "使用 GPU ${gpu_id} 运行: ${ckpt_path} ${env_name}"
  # 启动服务并获取服务进程的 PID
  port=$((base_port + run_count))
  start_service ${gpu_id} ${ckpt_path} ${port}

  CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
    --robot google_robot_static \
    --port $port \
    --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
    --env-name ${env_name} --scene-name ${scene_name} \
    --robot-init-x 0.65 0.85 3 --robot-init-y -0.2 0.2 3 \
    --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0.0 0.0 1 \
    --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
    ${EXTRA_ARGS}
  
  eval_pids+=($!)
}

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do

    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    if (( (run_count + 1) == 32 )); then
      EvalSim
    else
      EvalSim &
    fi
    run_count=$((run_count + 1))
    sleep 2
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
      EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt"
      
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
      if (( (run_count + 1) == 32 )); then
        EvalSim
      else
        EvalSim &
      fi
      run_count=$((run_count + 1))
      sleep 2
    done
  done
done

# lightings
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=brighter"
    
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    if (( (run_count + 1) == 32 )); then
      EvalSim
    else
      EvalSim &
    fi
    run_count=$((run_count + 1))

    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt light_mode=darker"
    
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    if (( (run_count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim &
    fi
    run_count=$((run_count + 1))
    sleep 2
  done
done

# new cabinets
scene_name=frl_apartment_stage_simple

for ckpt_path in "${ckpt_paths[@]}"; do
  for env_name in "${env_names[@]}"; do
    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station2"
    
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    if (( (run_count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim &
    fi
    run_count=$((run_count + 1))
    sleep 2

    EXTRA_ARGS="--additional-env-build-kwargs shader_dir=rt station_name=mk_station3"
    
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    if (( (run_count + 1) % 32 == 0 )); then
      EvalSim
    else
      EvalSim &
    fi
    run_count=$((run_count + 1))
    sleep 2
  done
done


stop_all_services
# wait
echo "✅ 所有测试完成"

