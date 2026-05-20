#!/bin/bash
export star_vla_python=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA/.venv/bin/python
export PYTHONPATH=${STARVLA_ROOT}:${PYTHONPATH}     # ★ 关键这行
your_ckpt=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA/results/Checkpoints/qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview_d_style_ft_20260520_043105/checkpoints/steps_2000_pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################

