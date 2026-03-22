#!/bin/bash
# eval_vla_arena_parall.sh
#
# Launches ONE parallel evaluation job: starts a policy server on a free GPU port,
# runs evaluation for the given suite + level, then kills the server.
#
# Usage:
#   bash eval_vla_arena_parall.sh <ckpt_path> <suite_name> <task_level> <run_index>
#
# Example:
#   bash eval_vla_arena_parall.sh results/.../steps_50000.pt safety_static_obstacles 0 0

###########################################################################################
# === Please modify the following paths according to your environment ===
export VLA_ARENA_HOME=/path/to/VLA-Arena
export VLA_ARENA_python=python       # Python env that has vla_arena installed
export starVLA_python=python         # Python env that has starVLA installed

export starVLA_HOME=$(pwd)
export PYTHONPATH=${VLA_ARENA_HOME}/vla_arena:${PYTHONPATH}
export PYTHONPATH=${starVLA_HOME}:${PYTHONPATH}
###########################################################################################

your_ckpt=$1          # path to .pt checkpoint
suite_name=$2         # e.g. safety_static_obstacles
task_level=$3         # 0 | 1 | 2
run_index=$4          # unique integer to avoid GPU/port collisions

num_gpus=8
gpu_id=$((run_index % num_gpus))
base_port=$((6600 + run_index))
num_trials_per_task=10

# Derive output paths from checkpoint path
model_root=$(echo "${your_ckpt}" | awk -F'/checkpoints/' '{print $1}')
folder_name=$(echo "${your_ckpt}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="${model_root}/videos/${suite_name}_L${task_level}/${folder_name}"
log_path="${model_root}/logs/${suite_name}_L${task_level}"
mkdir -p "${video_out_path}" "${log_path}"

# ---------------------------------------------------------------------------
# Start policy server in the background
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${gpu_id} ${starVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${base_port}" \
    --use_bf16 &

server_pid=$!

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------
${VLA_ARENA_python} ./examples/VLA-Arena/eval_files/eval_vla_arena.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "127.0.0.1" \
    --args.port "${base_port}" \
    --args.task-suite-name "${suite_name}" \
    --args.task-level "${task_level}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.video-out-path "${video_out_path}" \
    --args.save-video-mode "first_success_failure" \
    2>&1 | tee "${log_path}/${folder_name}.log"

echo "Evaluation done. Videos: ${video_out_path}  Log: ${log_path}/${folder_name}.log"

# ---------------------------------------------------------------------------
# Kill policy server
# ---------------------------------------------------------------------------
if [ -n "${server_pid}" ]; then
    echo "Killing server PID: ${server_pid}"
    kill "${server_pid}"
fi
