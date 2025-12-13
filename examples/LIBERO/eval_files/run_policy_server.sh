#!/bin/bash

your_ckpt=./results/Checkpoints/1207_libero4in1_qwen3fast/checkpoints/steps_10000_pytorch_model.pt

base_port=10093
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

# export DEBUG=1

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16