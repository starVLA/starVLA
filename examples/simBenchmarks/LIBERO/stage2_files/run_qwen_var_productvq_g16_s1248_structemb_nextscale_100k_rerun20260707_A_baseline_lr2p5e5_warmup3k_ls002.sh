#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

RUN_ID="qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_A_baseline_lr2p5e5_warmup3k_ls002"
CONFIG_YAML="examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_A_baseline_lr2p5e5_warmup3k_ls002.yaml"
STAGE1_ARTIFACT="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb_rerun_local_20260707/best_recon.ckpt"
STAGE1_CONFIG="examples/simBenchmarks/LIBERO/train_files/train_var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb_rerun_local_20260707.yaml"
TOKEN_CACHE="playground/Checkpoints/var_stage1_pi05_libero_q99_e32_aeinit_productvq_g16_s1_2_4_8_structemb_rerun_local_20260707/stage2_token_cache_full.pt"
RUN_DIR="/root/feihong/starVLA/Checkpoints/${RUN_ID}"

for required_path in "${CONFIG_YAML}" "${STAGE1_ARTIFACT}" "${STAGE1_CONFIG}" "${TOKEN_CACHE}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

export PATH="$(pwd)/.venv/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="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"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${RUN_DIR}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/train.log}"

cat <<EOF
[libero_stage2_rerun]
  run_id=${RUN_ID}
  config=${CONFIG_YAML}
  run_dir=${RUN_DIR}
  stage1_artifact=${STAGE1_ARTIFACT}
  stage1_config=${STAGE1_CONFIG}
  token_cache=${TOKEN_CACHE}
  cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
  num_processes=${NUM_PROCESSES:-8}
  mixed_precision=${MIXED_PRECISION:-bf16}
  main_process_port=${MAIN_PROCESS_PORT:-29542}
EOF

"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_PROCESSES:-8}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --main_process_port "${MAIN_PROCESS_PORT:-29542}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  2>&1 | tee -a "${LOG_FILE}"
