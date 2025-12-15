###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/petrelfs/share/yejinhui/Projects/LIBERO  # Root directory of the LIBERO project
export LIBERO_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/lerobot/bin/python  # Path to the Python environment
export starVLA_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python  # Path to the Python environment

# === End of environment variable configuration ===
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero  # Path to LIBERO configuration files
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from starVLA repo
###########################################################################################



##### === variables for which evaluation to setup ===
your_ckpt=$1 # results/Checkpoints/1025_libero_all_qwengroot/checkpoints/steps_20000_pytorch_model.pt
task_suite_name=$2 # align with your model | libero_goal
run_index=$3
# your_ckpt=results/Checkpoints/1025_libero_10_qwengroot/checkpoints/steps_10000_pytorch_model.pt
# task_suite_name=libero_10
# run_index=8
##### === variables for which evaluation to setup ===

num_gpus=8
gpu_id=$((run_index % num_gpus))

num_trials_per_task=50
host="127.0.0.1"
base_port=$((6450 + run_index))
unnorm_key="franka"

CUDA_VISIBLE_DEVICES=$gpu_id ${starVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16 &


# 获取服务的 PID
server_pid=$!


# Extract model_root from your_ckpt
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}"
log_path="${model_root}/logs/${task_suite_name}"
mkdir -p "$video_out_path"
mkdir -p "$log_path"



${LIBERO_python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"  \
    2>&1 | tee ${log_path}/${folder_name}.log

echo "Evaluation completed. Videos saved to ${video_out_path}, logs saved to ${log_path}/${folder_name}.log"



# 结束的时候要 kill 掉 base_port 的程序
if [ -n "$server_pid" ]; then
    echo "Killing server process with PID: $server_pid"
    kill $server_pid
else
    echo "No server process found to kill."
fi