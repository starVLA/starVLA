
# sim_python=~/Envs/miniconda3/envs/starVLA/bin/python
port=5678
# export DEBUG=true
export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

your_ckpt=./results/Checkpoints/1208_bridge_rt_1_QwenPI_qwen2.5/checkpoints/steps_80000_pytorch_model.pt

CUDA_VISIBLE_DEVICES=0 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16