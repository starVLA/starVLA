#!/bin/bash

your_ckpt=playground/Pretrained_models/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt
base_port=10093
DEBUG=true

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16