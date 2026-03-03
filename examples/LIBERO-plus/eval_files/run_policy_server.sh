#!/bin/bash

your_ckpt=path_to_ABot_checkpoint
base_port=9883
export ABot_python=path_to_ABot_env_python

export DEBUG=1

CUDA_VISIBLE_DEVICES=3 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16