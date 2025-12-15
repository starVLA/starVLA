#!/bin/bash

echo `which python`
# 定义环境
cd /mnt/petrelfs/yejinhui/Projects/starVLA
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python
export sim_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/dinoact/bin/python
export SimplerEnv_PATH=/mnt/petrelfs/share/yejinhui/Projects/SimplerEnv
export PYTHONPATH=$(pwd):${PYTHONPATH}
base_port=6350 

# export DEBUG=1


MODEL_PATH=$1
# MODEL_PATH=/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/1120_bridge_rt_1_QwenDual_florence/checkpoints/steps_11000_pytorch_model.pt
TSET_NUM=4 # repeat each task 4 times
run_count=0

if [ -z "$MODEL_PATH" ]; then
  echo "❌ 没传入 MODEL_PATH 作为第一个参数, 使用默认参数"
  export MODEL_PATH="/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/1007_qwenLargefm/checkpoints/steps_20000_pytorch_model.pt"
fi

ckpt_path=${MODEL_PATH}

# 定义一个函数来启动服务
policyserver_pids=()
eval_pids=()



start_service() {
  local gpu_id=$1
  local ckpt_path=$2
  local port=$3
  local server_log_dir="$(dirname "${ckpt_path}")/server_logs"
  local svc_log="${server_log_dir}/$(basename "${ckpt_path%.*}")_policy_server_${port}.log"
  mkdir -p "${server_log_dir}"

  # 提前检查端口，如果已经用了 kill 掉它
  if lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ 端口 ${port} 已被占用，尝试释放..."
    lsof -iTCP:"${port}" -sTCP:LISTEN -t | xargs kill -9
    sleep 2
    if lsof -iTCP:"${port}" -sTCP:LISTEN -t >/dev/null ; then
      echo "❌ 无法释放端口 ${port}，请手动检查"
      exit 1
    else
      echo "✅ 端口 ${port} 已成功释放"
    fi
  fi
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



# 打印调试信息
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_DEVICES: ${CUDA_DEVICES[@]}"
echo "NUM_GPUS: $NUM_GPUS"

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

# 任务列表，每行指定一个 env-name
declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)


for i in "${!ENV_NAMES[@]}"; do
  env="${ENV_NAMES[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$((run_count % NUM_GPUS))]}
    
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"  # 去掉 .pt 或 .bin 后缀

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"
    
    # 启动服务并获取服务进程的 PID
    port=$((base_port + run_count))
    start_service ${gpu_id} ${ckpt_path} ${port}

    
    CUDA_VISIBLE_DEVICES=${gpu_id} ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
      --port $port \
      --ckpt-path ${ckpt_path} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &
    
    eval_pids+=($!)
    run_count=$((run_count + 1))
  done
done

# V2 同理：PutEggplantInBasketScene-v0 也执行 5 次
declare -a ENV_NAMES_V2=(
  PutEggplantInBasketScene-v0
)

scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup

rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for i in "${!ENV_NAMES_V2[@]}"; do
  env="${ENV_NAMES_V2[i]}"
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    gpu_id=${CUDA_DEVICES[$(((run_count) % NUM_GPUS))]}  # 映射到 CUDA_VISIBLE_DEVICES 中的 GPU ID
    ckpt_dir=$(dirname "${ckpt_path}")
    ckpt_base=$(basename "${ckpt_path}")
    ckpt_name="${ckpt_base%.*}"

    tag="run${run_idx}"
    task_log="${ckpt_dir}/${ckpt_name}_infer_${env}.log.${tag}"

    echo "▶️ Launching V2 task [${env}] run#${run_idx} on GPU $gpu_id, log → ${task_log}"

    # 启动服务并获取服务进程的 PID
    echo "server start run#${run_idx}"
    port=$((base_port + run_count))
    server_pid=$(start_service ${gpu_id} ${ckpt_path} ${port})

    echo "sim start run#${run_idx}"
    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port $port \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      2>&1 | tee "${task_log}" &
    
    eval_pids+=($!)
    echo "sim end run#${run_idx}"
    
    run_count=$((run_count + 1))
  done
done



stop_all_services
wait
echo "✅ 所有测试完成"


