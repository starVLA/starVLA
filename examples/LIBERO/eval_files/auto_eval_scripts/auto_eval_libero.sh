#!/bin/bash

cd /mnt/petrelfs/yejinhui/Projects/starVLA
SCRIPT_PATH="./examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"


your_ckpt=./results/Checkpoints/1208_libero_all_QwenPI_qwen2/checkpoints/steps_60000_pytorch_model.pt
run_index_base=292

#####################################################
task_suite_name=libero_10 # align with your model
run_index=$((run_index_base + 0))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

sleep 15
#####################################################
task_suite_name=libero_goal # align with your model
run_index=$((run_index_base + 1))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
#####################################################
task_suite_name=libero_object # align with your model
run_index=$((run_index_base + 2))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
####################################################
task_suite_name=libero_spatial # align with your model
run_index=$((run_index_base + 3))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

