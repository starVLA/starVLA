# 使用所有 8 个 GPU 并将任务挂到后台运行


# 定义环境
cd /mnt/petrelfs/yejinhui/Projects/llavavla
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starvlaSAM/bin/python
export sim_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=5500


MODEL_PATH=$1

# 可选：判断是否传入了参数
if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1017_QwenOFT/checkpoints/steps_65000_pytorch_model.pt"
fi


# 定义一个函数来启动服务
policyserver_pids=()
eval_pids=()

task_name=drawer_vm

start_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local port=$3
  local server_log_dir="$(dirname "${ckpt_path}")/server_logs"
  local svc_log="${server_log_dir}/$(basename "${ckpt_path%.*}")_${task_name}_${port}.log"
  mkdir -p "${server_log_dir}"
  
  echo "▶️ Starting service on GPU ${gpu_id}, port ${port}"
  CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${ckpt_path} \
    --port ${port} \
    --use_bf16 \
    > "${svc_log}" 2>&1 &
  
  local pid=$!          # 立即捕获正确 PID
  policyserver_pids+=($pid)
  sleep 10
}

# 定义一个函数来停止所有服务
stop_all_services() {
  # 等待所有评估任务完成
  if [ "${#eval_pids[@]}" -gt 0 ]; then
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
  fi

  # 停止所有服务
  if [ "${#policyserver_pids[@]}" -gt 0 ]; then
    echo "⏳ 停止服务进程..."
    for pid in "${policyserver_pids[@]}"; do
      if ps -p "$pid" > /dev/null 2>&1; then
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
      else
        echo "⚠️ 服务进程 $pid 已不存在 (可能已提前退出)"
      fi
    done
  fi

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

declare -a env_names=(
  OpenTopDrawerCustomInScene-v0
  OpenMiddleDrawerCustomInScene-v0
  OpenBottomDrawerCustomInScene-v0
  CloseTopDrawerCustomInScene-v0
  CloseMiddleDrawerCustomInScene-v0
  CloseBottomDrawerCustomInScene-v0
)

# URDF variations
declare -a urdf_version_arr=("recolor_cabinet_visual_matching_1" "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" None)

# 使用 8 GPUs 轮转分配
total_gpus=8
run_count=0

for urdf_version in "${urdf_version_arr[@]}"; do

  EXTRA_ARGS="--enable-raytracing --additional-env-build-kwargs station_name=mk_station_recolor light_mode=simple disable_bad_material=True urdf_version=${urdf_version}"

  EvalSim() {
    # A0

    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}


    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.644 0.644 1 --robot-init-y -0.179 -0.179 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.03 -0.03 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a0.png \
      ${EXTRA_ARGS}

    # A1
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.765 0.765 1 --robot-init-y -0.182 -0.182 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.02 -0.02 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a1.png \
      ${EXTRA_ARGS}

    # A2
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.889 0.889 1 --robot-init-y -0.203 -0.203 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.06 -0.06 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_a2.png \
      ${EXTRA_ARGS}

    # B0
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.652 0.652 1 --robot-init-y 0.009 0.009 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b0.png \
      ${EXTRA_ARGS}

    # B1
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.752 0.752 1 --robot-init-y 0.009 0.009 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b1.png \
      ${EXTRA_ARGS}

    # B2
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.851 0.851 1 --robot-init-y 0.035 0.035 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_b2.png \
      ${EXTRA_ARGS}

    # C0
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.665 0.665 1 --robot-init-y 0.224 0.224 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c0.png \
      ${EXTRA_ARGS}

    # C1
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.765 0.765 1 --robot-init-y 0.222 0.222 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c1.png \
      ${EXTRA_ARGS}

    # C2
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py --ckpt-path ${ckpt_path} \
      --robot google_robot_static \
      --port $port \
      --control-freq 3 --sim-freq 513 --max-episode-steps 113 \
      --env-name ${env_name} --scene-name dummy_drawer \
      --robot-init-x 0.865 0.865 1 --robot-init-y 0.222 0.222 1 \
      --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.025 -0.025 1 \
      --obj-init-x-range 0 0 1 --obj-init-y-range 0 0 1 \
      --rgb-overlay-path ${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/open_drawer_c2.png \
      ${EXTRA_ARGS}
  }
  # 32 = 4*8 gpus， 一个GPU最多允许 4 个
  for ckpt_path in "${ckpt_paths[@]}"; do
    for env_name in "${env_names[@]}"; do
      gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
      if (( (run_count + 1) % 32 == 0 )); then
        EvalSim
      else
        EvalSim &
        eval_pids+=($!)
      fi

      sleep 15
      run_count=$((run_count + 1))
    done
  done

done

# 等待所有后台任务完成
stop_all_services
# wait
echo "✅ 所有测试完成"