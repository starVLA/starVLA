#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LIBERO_HOME="${LIBERO_HOME:-}"
LIBERO_Python="${LIBERO_Python:-python}"
MUJOCO_GL="${MUJOCO_GL:-osmesa}"
PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
host="${host:-127.0.0.1}"
your_ckpt="${your_ckpt:-/path/to/checkpoint.pt}"
output_dir="${output_dir:-${STARVLA_DIR}/results/libero_plus_eval}"
gpu_id="${gpu_id:-0}"

if [[ -z "${LIBERO_HOME}" ]]; then
  echo "LIBERO_HOME is required."
  exit 1
fi

cd "${STARVLA_DIR}"
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
LOG_DIR="${output_dir}/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}"

base_port=9883
task_suite_name=libero_goal
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_Python}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.log-path "$LOG_DIR" \
    2>&1 | tee "${log_file}" &


##########  eval libero_spatial ##########

# set it in background to run multiple evals in parallel with &
base_port=9883
task_suite_name=libero_spatial
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_Python}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.log-path "$LOG_DIR" \
    2>&1 | tee "${log_file}" &


##########  eval libero_object ##########
base_port=9883
task_suite_name=libero_object
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_Python}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.log-path "$LOG_DIR" \
    2>&1 | tee "${log_file}" &



##########  eval libero_long ##########
base_port=9883
task_suite_name=libero_10
num_trials_per_task=1
video_out_path="${output_dir}/${task_suite_name}/${folder_name}"
log_file="${LOG_DIR}/${task_suite_name}.log"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${LIBERO_Python}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.log-path "$LOG_DIR" \
    2>&1 | tee "${log_file}" &

# =============== Wait for all background tasks to finish ===============
echo "Waiting for all evaluation tasks to finish..."
wait  

# # =============== Aggregate results ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py
