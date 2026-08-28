#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR=${STARVLA_DIR:-/root/feihong/starVLA}
cd "${STARVLA_DIR}"
SCRIPT_PATH="./examples/simBenchmarks/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"

CKPT_DIR=${CKPT_DIR:-/root/nas/feihong/starVLA/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints}
TASK_SUITES_STR=${TASK_SUITES_STR:-"libero_spatial libero_object libero_goal libero_10"}
GPU_LIST_STR=${GPU_LIST_STR:-"0 1"}
BASE_PORT=${BASE_PORT:-6450}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-20}

read -r -a TASK_SUITES <<< "${TASK_SUITES_STR}"
read -r -a GPU_LIST <<< "${GPU_LIST_STR}"

if [[ -n "${CKPT_LIST_STR:-}" ]]; then
  read -r -a CKPT_LIST <<< "${CKPT_LIST_STR}"
else
  mapfile -t CKPT_LIST < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'steps_*_pytorch_model.pt' | sort -V)
fi

if [[ ${#CKPT_LIST[@]} -eq 0 ]]; then
  echo "[ERROR] No checkpoints found in ${CKPT_DIR}" >&2
  exit 1
fi

num_gpus=${#GPU_LIST[@]}
job_index=0
pids=()

echo "=========================================="
echo " Auto Eval LIBERO (feihong)"
echo "=========================================="
echo " Checkpoints : ${CKPT_LIST[*]}"
echo " Task suites : ${TASK_SUITES[*]}"
echo " GPU list    : ${GPU_LIST[*]}"
echo "=========================================="

for ckpt in "${CKPT_LIST[@]}"; do
  for task in "${TASK_SUITES[@]}"; do
    gpu_idx=$((job_index % num_gpus))
    gpu_id=${GPU_LIST[$gpu_idx]}
    port=$((BASE_PORT + job_index))
    echo "[Job ${job_index}] GPU=${gpu_id} port=${port} ckpt=$(basename "${ckpt}") task=${task}"
    bash "${SCRIPT_PATH}" "${ckpt}" "${task}" "${gpu_id}" "${port}" &
    pids+=("$!")
    job_index=$((job_index + 1))
    sleep "${SLEEP_BETWEEN}"
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All LIBERO eval jobs completed."
