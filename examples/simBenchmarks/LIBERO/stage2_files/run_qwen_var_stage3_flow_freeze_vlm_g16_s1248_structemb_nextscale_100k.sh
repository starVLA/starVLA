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

RUN_ID="${RUN_ID:-qwen_var_stage3_pi_flow_freeze_vlm_g16_s1248_structemb_nextscale_100k_fullcache}"

"${ACCELERATE_BIN}" launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --run_id "${RUN_ID}" \
  --trainer.pretrained_checkpoint "${STAGE2_CHECKPOINT}" \
  --trainer.is_resume false \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.max_train_steps 100000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 1000 \
  --trainer.save_interval 1000 \
  --framework.stage3.var_ce_weight 0.0 \
  --framework.stage3.flow_loss_weight 1.0 \
  --trainer.freeze_modules qwen_vl_interface,action_token_queries,action_query_cross_attn,action_token_norm,action_token_classifier,action_factor_classifiers,code_condition_projectors,code_condition_norm \
  --trainer.learning_rate.qwen_vl_interface 0.0 \
  --trainer.learning_rate.base 1.0e-04 \
  --trainer.learning_rate.action_model 1.0e-04 \
  "$@"
