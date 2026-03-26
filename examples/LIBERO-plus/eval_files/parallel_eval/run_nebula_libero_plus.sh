#!/bin/bash

ENVS="WANDB_MODE=offline,WANDB_MODE=disabled,HF_ENDPOINT=https://hf-mirror.com"

sizes=(2519 2591 2518 2402)
tasks=("libero_10" "libero_goal" "libero_object" "libero_spatial")

############ GPUs number for each task
total_slices=(12 4 4 4)
############

for i in "${!sizes[@]}"; do
    size=${sizes[$i]}
    task=${tasks[$i]}
    num_slice=${total_slices[$i]}
    base_size=$((size / num_slice))
    remainder=$((size % num_slice))

    start_idx=0

    for slice in $(seq 0 $((num_slice - 1))); do

        if [ "$slice" -lt "$remainder" ]; then
            end_idx=$((start_idx + base_size + 1))
        else
            end_idx=$((start_idx + base_size))
        fi


        if [ "$slice" -eq $((num_slice - 1)) ]; then
            end_idx=$size
        fi

        echo "task=$task, slice=$slice, start_idx=$start_idx, end_idx=$end_idx"

        nebulactl run mdl --queue=...... \
                  --entry="bash examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_in_one.sh $task $start_idx $end_idx" \
                  --user_params="" \
                  --worker_count=1 \
                  --algoame=pytorch240 \
                  --file.cluster_file=./cluster.json \
                  --job_name="libero_plus" \
                  --nas_file_system_id=...... \
                  --nas_file_system_mount_path=...... \
                  --custom_docker_image=...... \
                  --env="${ENVS}" \
                #   --public_pool_job_type=queuing

        start_idx=$end_idx
    done
done
