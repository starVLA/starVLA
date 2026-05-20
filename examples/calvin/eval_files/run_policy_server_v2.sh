#!/bin/bash
# ===========================================================================
# Second policy server for parallel eval — DO NOT collide with v1
#   * GPU 1         (v1 uses GPU 0)
#   * port 5695     (v1 uses 5694)
#   * starvla_backup env (transformers 5.8.0) — required for Qwen3.5 ckpts
#   * Qwen35 2B GR00T multiview 40k-step ckpt — known-good baseline
# ===========================================================================

export STARVLA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA
export PYTHONPATH=${STARVLA_ROOT}:${PYTHONPATH}

# Use the backup conda env on /four (transformers 5.8.0 supports Qwen3_5*).
# DO NOT switch to .venv (transformers 4.57.0) — it cannot import the model.
export star_vla_python=/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/starvla_backup/bin/python

your_ckpt=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA/results/Checkpoints/qwen35_2b_gr00t_calvin_abc_multiview_20260519_133550/checkpoints/steps_40000_pytorch_model.pt
gpu_id=1
port=5695

cd "${STARVLA_ROOT}"

CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16
