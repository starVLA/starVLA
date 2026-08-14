#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME
fi
if [[ -n "${NCCL_IB_HCA:-}" ]]; then
  export NCCL_IB_HCA
fi
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
export TOKENIZERS_PARALLELISM=false

CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/LIBERO/stage2_files/train_qwen_var_stage3_flow_g16_s1248_structemb_nextscale_100k.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
STAGE2_CHECKPOINT="${STAGE2_CHECKPOINT:-/home/zhangfeihong/starVLA/playground/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints/steps_100000_pytorch_model.pt}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/home/zhangfeihong/miniconda3/envs/starVLA/bin/accelerate}"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --trainer.pretrained_checkpoint "${STAGE2_CHECKPOINT}" \
  --trainer.is_resume false \
  "$@"
