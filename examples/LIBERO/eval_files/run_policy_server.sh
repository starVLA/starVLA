#!/bin/bash

your_ckpt=results/Checkpoints/1201_libero4in1_Qwen2.5fast/checkpoints/steps_20000_pytorch_model.pt
base_port=10093
export star_vla_python={path_to}/Envs/miniconda3/envs/starVLA/bin/python

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16