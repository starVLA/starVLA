

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

task_name=pick_coke_va


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


declare -a arr=(${MODEL_PATH})

# 轮转分配用的变量
total_gpus=8
run_count=0

# lr_switch=laying horizontally but flipped left-right to match real eval; upright=standing; laid_vertically=laying vertically
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")

for ckpt_path in "${arr[@]}"; do
  echo "$ckpt_path"
done

# base setup
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} &
    
    eval_pids+=($!)  # 将评估任务的 PID 添加到数组中
    run_count=$((run_count + 1))
  done
done

# table textures
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a scene_arr=("Baked_sc1_staging_objaverse_cabinet1_h870" \
                      "Baked_sc1_staging_objaverse_cabinet2_h870")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

      CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      
      eval_pids+=($!)  # 将评估任务的 PID 添加到数组中
      run_count=$((run_count + 1))
    done
  done
done

# distractors
env_name=GraspSingleOpenedCokeCanDistractorInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} &
    
    eval_pids+=($!)  # 将评估任务的 PID 添加到数组中
    run_count=$((run_count + 1))

    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} distractor_config=more &
    
    eval_pids+=($!)  # 将评估任务的 PID 添加到数组中
    run_count=$((run_count + 1))
  done
done

# backgrounds
env_name=GraspSingleOpenedCokeCanInScene-v0
declare -a bg_scene_arr=("google_pick_coke_can_1_v4_alt_background" \
                         "google_pick_coke_can_1_v4_alt_background_2")

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for scene_name in "${bg_scene_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
      # 启动服务并获取服务进程的 PID
      port=$((base_port + run_count))
      start_service ${gpu_id} ${ckpt_path} ${port}

      CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      
      eval_pids
      run_count=$((run_count + 1))
    done
  done
done

# lightings
env_name=GraspSingleOpenedCokeCanInScene-v0
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_darker_lighting=True &
    
    eval_pids+=($!)  # 将评估任务的 PID 添加到数组中
    run_count=$((run_count + 1))

    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port ${port} \
      --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
      --env-name ${env_name} --scene-name ${scene_name} \
      --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --additional-env-build-kwargs ${coke_can_option} slightly_brighter_lighting=True &
    
    eval_pids+=($!)  # 将评估任务的 PID 添加到数组中
    run_count=$((run_count + 1))
  done
done

# camera orientations
declare -a env_arr=("GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0" \
                    "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0")
scene_name=google_pick_coke_can_1_v4

for coke_can_option in "${coke_can_options_arr[@]}"; do
  for env_name in "${env_arr[@]}"; do
    for ckpt_path in "${arr[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
      # 启动服务并获取服务进程的 PID
      port=$((base_port + run_count))
      start_service ${gpu_id} ${ckpt_path} ${port}
      CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --port ${port} \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
        --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
        --additional-env-build-kwargs ${coke_can_option} &
      
      eval_pids+=($!)  # 将评估任务的 PID 添加到数组中
      run_count=$((run_count + 1))
    done
  done
done

# 等待所有后台任务完成
stop_all_services
echo "✅ 所有测试完成"

