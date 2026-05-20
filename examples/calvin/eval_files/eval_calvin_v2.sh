#!/bin/bash
# ===========================================================================
# Second eval client for parallel eval — DO NOT collide with v1
#   * connects to port 5695 (v1 connects to 5694)
#   * writes to its own log dir under logs/eval_v2_<timestamp>/
#   * samples 100 sequences with seed=0 (same 100 as v1 if v1 also seed=0)
# ===========================================================================

export STARVLA_ROOT=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA
export calvin_python=/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/calvin_venv/bin/python
export PYTHONPATH=${STARVLA_ROOT}:${PYTHONPATH}

# Pin the calvin sim's EGL to GPU 1 so it shares with v2's policy server
# rather than crowding v1 on GPU 0. Comment out to let EGL auto-pick.
export EGL_DEVICE_ID=1

host="127.0.0.1"
base_port=5695
unnorm_key="franka"
your_ckpt=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA/results/Checkpoints/qwen35_2b_gr00t_calvin_abc_multiview_20260519_133550/checkpoints/steps_40000_pytorch_model.pt

LOG_DIR="${STARVLA_ROOT}/logs/eval_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}

${calvin_python} ${STARVLA_ROOT}/examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path ${your_ckpt} \
    --args.unnorm-key ${unnorm_key} \
    --args.host "$host" --args.port $base_port \
    --args.dataset_path /inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d \
    --args.calvin_config_path /inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/calvin_models/conf \
    --args.eval_sequences_path ${STARVLA_ROOT}/examples/calvin/eval_files/eval_sequences.json \
    --args.num_sequences 100 \
    --args.seed 0 \
    --args.eval-log-dir ${LOG_DIR}
