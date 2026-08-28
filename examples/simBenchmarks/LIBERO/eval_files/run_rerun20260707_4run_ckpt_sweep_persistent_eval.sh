#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-/root/feihong/starVLA}"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/run_stage2_ckpt_sweep_single_gpu_parallel_eval.sh}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/root/feihong/starVLA/Checkpoints}"
EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-eval_sweep_26k_to_40k_40task_50ep_robust_seed7_20260709}"
LOG_ROOT="${LOG_ROOT:-/root/feihong/starVLA/eval_logs/${EVAL_OUTPUT_PREFIX}}"
MASTER_LOG="${LOG_ROOT}/master.log"

# One local H100 was detected in this workspace, so this supervisor runs one
# checkpoint at a time on one GPU. All chunks are resumable and incomplete
# chunks are retried by the underlying persistent eval script.
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-19250}"
USE_BF16="${USE_BF16:-1}"
EVAL_SEED="${EVAL_SEED:-7}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-1}"
MAX_RETRIES="${MAX_RETRIES:-100000}"
CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS:-1800}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-900}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-224}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}"
MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN:-2.0}"
MIN_IMAGE_STD="${MIN_IMAGE_STD:-1.0}"
CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-0}"
SWEEP_STEPS="${SWEEP_STEPS:-26000 27000 28000 29000 30000 31000 32000 33000 34000 35000 36000 37000 38000 39000 40000}"

RUN_IDS=(
  "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_A_baseline_lr2p5e5_warmup3k_ls002"
  "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_B_stable_lr2e5_warmup5k_ls002"
  "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_C_aggressive_lr3e5_warmup3k_ls002"
  "qwen_var_productvq_g16_s1248_structemb_nextscale_100k_rerun20260707_D_ls001_lr2p5e5_warmup3k"
)

mkdir -p "${LOG_ROOT}"
cd "${STARVLA_DIR}"

{
  echo "[$(date)] rerun20260707 4-run LIBERO ckpt sweep started"
  echo "[$(date)] checkpoint_base=${CHECKPOINT_BASE}"
  echo "[$(date)] eval_output_prefix=${EVAL_OUTPUT_PREFIX}"
  echo "[$(date)] gpu=${GPU_ID} port=${PORT} bf16=${USE_BF16} seed=${EVAL_SEED} trials=${TRIALS_PER_TASK} chunk=${CHUNK_TRIALS} save_videos=${SAVE_VIDEOS}"
  echo "[$(date)] sweep_steps=${SWEEP_STEPS}"
} | tee -a "${MASTER_LOG}"

for run_id in "${RUN_IDS[@]}"; do
  run_root="${CHECKPOINT_BASE}/${run_id}"
  checkpoint_dir="${run_root}/checkpoints"
  run_log="${LOG_ROOT}/${run_id}.log"

  if [[ ! -d "${checkpoint_dir}" ]]; then
    echo "[$(date)] missing checkpoint dir: ${checkpoint_dir}" | tee -a "${MASTER_LOG}" >&2
    exit 1
  fi

  echo "[$(date)] ===== start run ${run_id} =====" | tee -a "${MASTER_LOG}"
  env \
    STARVLA_DIR="${STARVLA_DIR}" \
    RUN_ROOT="${run_root}" \
    CHECKPOINT_DIR="${checkpoint_dir}" \
    SWEEP_STEPS="${SWEEP_STEPS}" \
    EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX}" \
    GPU_ID="${GPU_ID}" \
    PORT="${PORT}" \
    USE_BF16="${USE_BF16}" \
    EVAL_SEED="${EVAL_SEED}" \
    TRIALS_PER_TASK="${TRIALS_PER_TASK}" \
    CHUNK_TRIALS="${CHUNK_TRIALS}" \
    MAX_RETRIES="${MAX_RETRIES}" \
    CHUNK_TIMEOUT_SECONDS="${CHUNK_TIMEOUT_SECONDS}" \
    SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    IMAGE_VIEWS="${IMAGE_VIEWS}" \
    POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE}" \
    VALIDATE_INPUTS="${VALIDATE_INPUTS}" \
    STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT}" \
    MIN_IMAGE_MEAN="${MIN_IMAGE_MEAN}" \
    MIN_IMAGE_STD="${MIN_IMAGE_STD}" \
    CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS}" \
    CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS}" \
    bash "${SWEEP_SCRIPT}" 2>&1 | tee -a "${run_log}"
  echo "[$(date)] ===== completed run ${run_id}; log=${run_log} =====" | tee -a "${MASTER_LOG}"
done

summary_out="${LOG_ROOT}/all_ckpt_summaries.tsv"
python examples/simBenchmarks/LIBERO/eval_files/summarize_rerun20260707_ckpt_sweep.py \
  --checkpoint-base "${CHECKPOINT_BASE}" \
  --eval-output-prefix "${EVAL_OUTPUT_PREFIX}" \
  --output "${summary_out}" | tee -a "${MASTER_LOG}"

echo "[$(date)] completed all runs; summary=${summary_out}" | tee -a "${MASTER_LOG}"
