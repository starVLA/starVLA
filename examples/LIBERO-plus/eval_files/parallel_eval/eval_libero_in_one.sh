#!/bin/bash
eval "$(conda shell.bash hook)"
# source activate
conda activate python3.10
# pip install -r requirements.txt
# pip list
# ls /usr/lib64/libOSMesa.so*
###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=path_to_LIBERO-plus_code
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export MUJOCO_GL=osmesa
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

unnorm_key="franka"
tasks_per_gpu=3
your_ckpt=path_to_checkpoint
output_dir=path_to_output_dir
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=$1
start_idx=$2
end_idx=$3
num_trials_per_task=1
# torchrun --nproc_per_node=1 ./examples/LIBERO-plus/eval_files/eval_nebula/eval_libero_model.py \

total=$((end_idx - start_idx))
chunk_size=$((total / tasks_per_gpu))
remainder=$((total % tasks_per_gpu))
current_start=$start_idx

for ((i=0; i<tasks_per_gpu; i++)); do

    if [ $i -lt $remainder ]; then
        current_end=$((current_start + chunk_size + 1))
    else
        current_end=$((current_start + chunk_size))
    fi


    if [ $current_end -gt $end_idx ]; then
        current_end=$end_idx
    fi

    echo "Part $((i)): start=$current_start, end=$current_end ([$current_start, $current_end))"
    # torchrun --nproc_per_node=$gpu_per_pod --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$((MASTER_ADDR+i)) --master_port=$MASTER_PORT 
    python ./examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_model.py \
    --pretrained_path $your_ckpt \
    --task_suite_name $task_suite_name \
    --num_trials_per_task $num_trials_per_task \
    --output_dir $output_dir \
    --start_idx $current_start \
    --end_idx $current_end &

    current_start=$current_end

    if [ $current_start -ge $end_idx ]; then
        break
    fi
done


wait

# # =============== Aggregate results ===============
# echo "All tasks completed. Aggregating results..."
# export LOG_DIR="${LOG_DIR}"
# python ./examples/LIBERO-plus/eval_files/aggregate_results.py
