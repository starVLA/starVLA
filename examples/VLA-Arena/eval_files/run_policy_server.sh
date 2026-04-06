#!/bin/bash
# run_policy_server.sh
#
# Launches the starVLA WebSocket policy server for VLA-Arena evaluation.
# Run this script first, then launch eval_vla_arena.sh in a separate terminal.

export PYTHONPATH=$(pwd):${PYTHONPATH}

###########################################################################################
# === Please modify the following paths according to your environment ===
export starVLA_python=python   # or: /path/to/conda/envs/starVLA/bin/python

your_ckpt=/mnt/file2/jiachen/pr/starVLA/playground/test/qwen2.5-libero/checkpoints/steps_30000_pytorch_model.pt
gpu_id=7
port=1009${gpu_id}
###########################################################################################

CUDA_VISIBLE_DEVICES=${gpu_id} ${starVLA_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
