#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let Franka find the websocket tools from main repo
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python
your_ckpt=results/Checkpoints/Franka_QwenPI_qwen3/checkpoints/steps_50000_pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################