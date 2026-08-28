#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-/root/feihong/starVLA}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/feihong/LIBERO/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/root/nas/feihong/starVLA/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_60k_fullcache_saveall}"
EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-eval_sweep_40k_to_50k_40task_50ep_8gpu_seed7_20260628}"
MASTER_LOG="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_master.log"

mkdir -p "${RUN_ROOT}"
echo "[$(date)] 8gpu LIBERO eval launcher invoked" | tee -a "${MASTER_LOG}"

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
echo "[$(date)] visible_gpu_count=${gpu_count}" | tee -a "${MASTER_LOG}"
  if [[ "${gpu_count}" -lt 8 ]]; then
    echo "Need 8 visible GPUs for this launcher, found ${gpu_count}. Set CUDA_VISIBLE_DEVICES or run on the 8xH100 node." | tee -a "${MASTER_LOG}" >&2
    exit 2
  fi
fi

JOB_SPECS_GPU0="${JOB_SPECS_GPU0:-50000:libero_10 42000:libero_object 40000:libero_goal}"
JOB_SPECS_GPU1="${JOB_SPECS_GPU1:-48000:libero_10 42000:libero_goal 40000:libero_spatial}"
JOB_SPECS_GPU2="${JOB_SPECS_GPU2:-46000:libero_10 42000:libero_spatial 40000:libero_object}"
JOB_SPECS_GPU3="${JOB_SPECS_GPU3:-44000:libero_10 50000:libero_spatial 48000:libero_goal}"
JOB_SPECS_GPU4="${JOB_SPECS_GPU4:-42000:libero_10 50000:libero_object 46000:libero_goal}"
JOB_SPECS_GPU5="${JOB_SPECS_GPU5:-40000:libero_10 50000:libero_goal 46000:libero_object}"
JOB_SPECS_GPU6="${JOB_SPECS_GPU6:-48000:libero_spatial 44000:libero_object 46000:libero_spatial}"
JOB_SPECS_GPU7="${JOB_SPECS_GPU7:-48000:libero_object 44000:libero_goal 44000:libero_spatial}"

declare -a JOBS_BY_GPU=(
  "${JOB_SPECS_GPU0}"
  "${JOB_SPECS_GPU1}"
  "${JOB_SPECS_GPU2}"
  "${JOB_SPECS_GPU3}"
  "${JOB_SPECS_GPU4}"
  "${JOB_SPECS_GPU5}"
  "${JOB_SPECS_GPU6}"
  "${JOB_SPECS_GPU7}"
)

echo "[$(date)] 8gpu LIBERO eval launcher started" | tee -a "${MASTER_LOG}"
echo "[$(date)] run_root=${RUN_ROOT}" | tee -a "${MASTER_LOG}"
echo "[$(date)] eval_output_prefix=${EVAL_OUTPUT_PREFIX}" | tee -a "${MASTER_LOG}"

declare -a pids=()
for gpu in $(seq 0 7); do
  port=$((19400 + gpu))
  worker_log="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_gpu${gpu}.log"
  echo "[$(date)] launch gpu=${gpu} port=${port} jobs=${JOBS_BY_GPU[$gpu]} log=${worker_log}" | tee -a "${MASTER_LOG}"
  (
    cd "${STARVLA_DIR}"
    RUN_ROOT="${RUN_ROOT}" \
    EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX}" \
    GPU_ID="${gpu}" \
    PORT="${port}" \
    JOB_SPECS="${JOBS_BY_GPU[$gpu]}" \
    USE_BF16="${USE_BF16:-0}" \
    EVAL_SEED="${EVAL_SEED:-7}" \
    TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}" \
    CHUNK_TRIALS="${CHUNK_TRIALS:-1}" \
    SAVE_VIDEOS="${SAVE_VIDEOS:-0}" \
    VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}" \
    STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}" \
    bash examples/simBenchmarks/LIBERO/eval_files/run_stage2_60k_suite_persistent_eval_worker.sh
  ) >> "${worker_log}" 2>&1 &
  pids[$gpu]=$!
done

failed=0
for gpu in $(seq 0 7); do
  if wait "${pids[$gpu]}"; then
    echo "[$(date)] gpu=${gpu} worker completed" | tee -a "${MASTER_LOG}"
  else
    echo "[$(date)] gpu=${gpu} worker FAILED" | tee -a "${MASTER_LOG}" >&2
    failed=1
  fi
done

for step in 50000 48000 46000 44000 42000 40000; do
  log_root="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_steps_${step}/logs"
  summary_path="${log_root}/libero_40task_summary.txt"
  if [[ -d "${log_root}" ]]; then
    "${LIBERO_PYTHON}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py "${log_root}" --chunked --require-ok-marker --episode-csv "${log_root}/libero_40task_episode_results.csv" --episode-jsonl "${log_root}/libero_40task_episode_results.jsonl" | tee "${summary_path}" || failed=1
  fi
done

echo "[$(date)] 8gpu LIBERO eval launcher finished failed=${failed}" | tee -a "${MASTER_LOG}"
exit "${failed}"
