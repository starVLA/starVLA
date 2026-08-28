#!/usr/bin/env bash
set -euo pipefail

CKPT="${1:-/home/zhangfeihong/starVLA/playground/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints/steps_100000_pytorch_model.pt}"
STARVLA_DIR="${STARVLA_DIR:-/home/zhangfeihong/starVLA}"
SEEDS_STR="${EVAL_SEEDS:-7 21 42}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-eval_stage2_100k_40task_4suite_seed}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"

MODEL_ROOT="$(echo "${CKPT}" | awk -F'/checkpoints/' '{print $1}')"
MULTISEED_LOG="${MODEL_ROOT}/${BASE_OUTPUT_ROOT}_multiseed_supervisor.log"
mkdir -p "${MODEL_ROOT}"

cd "${STARVLA_DIR}"
echo "[$(date)] multiseed eval supervisor started" | tee -a "${MULTISEED_LOG}"
echo "[$(date)] ckpt=${CKPT}" | tee -a "${MULTISEED_LOG}"
echo "[$(date)] seeds=${SEEDS_STR}" | tee -a "${MULTISEED_LOG}"

for seed in ${SEEDS_STR}; do
  output_root="${BASE_OUTPUT_ROOT}${seed}"
  echo "[$(date)] starting seed=${seed} output=${output_root}" | tee -a "${MULTISEED_LOG}"

  EVAL_SEED="${seed}" EVAL_OUTPUT_ROOT="${output_root}" CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS}" \
    bash examples/simBenchmarks/LIBERO/eval_files/run_stage2_100k_4suite_parallel_eval_supervisor.sh "${CKPT}"

  summary="${MODEL_ROOT}/${output_root}/logs/libero_40task_summary.txt"
  echo "[$(date)] finished seed=${seed} summary=${summary}" | tee -a "${MULTISEED_LOG}"
  if [[ -f "${summary}" ]]; then
    cat "${summary}" | tee -a "${MULTISEED_LOG}"
  fi
done

echo "[$(date)] all seeds completed" | tee -a "${MULTISEED_LOG}"
