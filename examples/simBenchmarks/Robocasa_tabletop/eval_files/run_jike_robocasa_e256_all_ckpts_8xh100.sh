#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=/root/feihong/starVLA
RUN_DIR=/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_closebalanced_e256_bestworst_e47_100k_lr1e4_warmup5000_gbs512_fullcache
QUEUE_SCRIPT=${RUN_DIR}/robocasa_eval_queue_logs/run_e256_all_ckpts_gr1_24_50eps.sh
MASTER_LOG=${MASTER_LOG:-${RUN_DIR}/robocasa_eval_queue_logs/e256_8gpu_sweep_master.log}

cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export EVAL_GPUS=${EVAL_GPUS:-"0 1 2 3 4 5 6 7"}
export BASE_PORT=${BASE_PORT:-22000}
export MAX_PASSES=${MAX_PASSES:-20}
export CHUNK_MAX_RETRIES=${CHUNK_MAX_RETRIES:-3}
export WORKER_STAGGER_SECONDS=${WORKER_STAGGER_SECONDS:-8}
export STARVLA_PYTHON=${STARVLA_PYTHON:-${REPO_DIR}/.venv/bin/python}
export ROBOCASA_PYTHON=${ROBOCASA_PYTHON:-${REPO_DIR}/.venv-robocasa-eval/bin/python}
export PYTHONPATH=${REPO_DIR}:${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export HF_HOME=${HF_HOME:-/root/feihong/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/root/feihong/.cache/torch}
export ROBOCASA_MJCF_TMPDIR=${ROBOCASA_MJCF_TMPDIR:-/tmp/robocasa_mjcf_tmp_e256_8gpu}

mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${ROBOCASA_MJCF_TMPDIR}" "$(dirname "${MASTER_LOG}")"

echo "[$(date "+%F %T %Z")] master_log=${MASTER_LOG}" | tee -a "${MASTER_LOG}"
bash "${QUEUE_SCRIPT}" 2>&1 | tee -a "${MASTER_LOG}"
