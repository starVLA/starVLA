your_ckpt=./results/Checkpoints/1003_qwenoft/final_model/pytorch_model.pt
sim_python=~/Envs/miniconda3/envs/starVLA/bin/python
port=5678
# DEBUG=true


CUDA_VISIBLE_DEVICES=0 ${sim_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16