#!/bin/bash


###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/petrelfs/share/yejinhui/Projects/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/lerobot/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

host="127.0.0.1"
base_port=10093
unnorm_key="franka"
your_ckpt=results/Checkpoints/1_need/Qwen2.5-FAST-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

# export DEBUG=true

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"

${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"


# ##########  eval libero_spatial ##########

# # set it in background to run multiple evals in parallel with &

# task_suite_name=libero_spatial
# num_trials_per_task=50
# video_out_path="results/${task_suite_name}/${folder_name}"

# ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
#     --args.pretrained-path ${your_ckpt} \
#     --args.host "$host" \
#     --args.port $base_port \
#     --args.task-suite-name "$task_suite_name" \
#     --args.num-trials-per-task "$num_trials_per_task" \
#     --args.video-out-path "$video_out_path" &


# ##########  eval libero_object ##########

# task_suite_name=libero_object
# num_trials_per_task=50
# video_out_path="results/${task_suite_name}/${folder_name}"

# ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
#     --args.pretrained-path ${your_ckpt} \
#     --args.host "$host" \
#     --args.port $base_port \
#     --args.task-suite-name "$task_suite_name" \
#     --args.num-trials-per-task "$num_trials_per_task" \
#     --args.video-out-path "$video_out_path" &



# ##########  eval libero_long ##########

# task_suite_name=libero_10
# num_trials_per_task=50
# video_out_path="results/${task_suite_name}/${folder_name}"

# ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
#     --args.pretrained-path ${your_ckpt} \
#     --args.host "$host" \
#     --args.port $base_port \
#     --args.task-suite-name "$task_suite_name" \
#     --args.num-trials-per-task "$num_trials_per_task" \
#     --args.video-out-path "$video_out_path" &