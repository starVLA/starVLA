#!/bin/bash

your_ckpt=results/Checkpoints/1025_libero_goal_qwengroot/checkpoints/steps_20000_pytorch_model.pt
base_port=10093
# export DEBUG=true

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16