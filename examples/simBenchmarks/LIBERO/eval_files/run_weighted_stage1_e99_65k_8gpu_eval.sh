#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-/root/feihong/starVLA}"
LIBERO_PYTHON="${LIBERO_PYTHON:-/root/feihong/LIBERO/.venv/bin/python}"
WORKER_SCRIPT="${WORKER_SCRIPT:-${STARVLA_DIR}/examples/simBenchmarks/LIBERO/eval_files/run_stage2_60k_suite_persistent_eval_worker.sh}"
RUN_ID="${RUN_ID:-qwen_var_productvq_g16_s1248_weighted_stage1_e99_nextscale_65k_gbs64_jike8h100}"
RUN_ROOT="${RUN_ROOT:-${STARVLA_DIR}/Checkpoints/${RUN_ID}}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints}"
EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-eval_key_48k_to_65k_40task_50ep_8gpu_seed7_20260813}"
MASTER_LOG="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_master.log"

# First-pass peak search. Override to sweep every retained checkpoint:
#   SWEEP_STEPS="$(seq -s ' ' 40000 1000 65000)" bash <this-script>
SWEEP_STEPS="${SWEEP_STEPS:-48000 52000 56000 58000 60000 62000 64000 65000}"
SUITES=(libero_spatial libero_object libero_goal libero_10)

USE_BF16="${USE_BF16:-1}"
EVAL_SEED="${EVAL_SEED:-7}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-50}"
CHUNK_TRIALS="${CHUNK_TRIALS:-1}"
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
VALIDATE_INPUTS="${VALIDATE_INPUTS:-1}"
STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT:-1}"

cd "${STARVLA_DIR}"
mkdir -p "${RUN_ROOT}"

gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${gpu_count}" -lt 8 ]]; then
  echo "Need 8 visible GPUs, found ${gpu_count}. Run this launcher inside the 8xH100 JiKe job." >&2
  exit 2
fi

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO Python is missing or not executable: ${LIBERO_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
  echo "Worker script is missing: ${WORKER_SCRIPT}" >&2
  exit 1
fi

declare -a JOBS_BY_GPU=("" "" "" "" "" "" "" "")
job_index=0
for step in ${SWEEP_STEPS}; do
  ckpt="${CHECKPOINT_DIR}/steps_${step}_pytorch_model.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
  for suite in "${SUITES[@]}"; do
    gpu=$((job_index % 8))
    JOBS_BY_GPU[$gpu]="${JOBS_BY_GPU[$gpu]} ${step}:${suite}"
    job_index=$((job_index + 1))
  done
done

{
  echo "[$(date)] weighted-stage1 LIBERO 8gpu eval started"
  echo "[$(date)] run_root=${RUN_ROOT}"
  echo "[$(date)] checkpoints=${SWEEP_STEPS}"
  echo "[$(date)] seed=${EVAL_SEED} trials_per_task=${TRIALS_PER_TASK} chunk=${CHUNK_TRIALS} bf16=${USE_BF16}"
  for gpu in $(seq 0 7); do
    echo "[$(date)] gpu=${gpu} jobs=${JOBS_BY_GPU[$gpu]}"
  done
} | tee -a "${MASTER_LOG}"

declare -a pids=()
for gpu in $(seq 0 7); do
  port=$((19400 + gpu))
  worker_log="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_gpu${gpu}.log"
  (
    RUN_ROOT="${RUN_ROOT}" \
    CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
    EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX}" \
    GPU_ID="${gpu}" \
    PORT="${port}" \
    JOB_SPECS="${JOBS_BY_GPU[$gpu]}" \
    USE_BF16="${USE_BF16}" \
    EVAL_SEED="${EVAL_SEED}" \
    TRIALS_PER_TASK="${TRIALS_PER_TASK}" \
    CHUNK_TRIALS="${CHUNK_TRIALS}" \
    SAVE_VIDEOS="${SAVE_VIDEOS}" \
    VALIDATE_INPUTS="${VALIDATE_INPUTS}" \
    STRICT_TRIAL_COUNT="${STRICT_TRIAL_COUNT}" \
    bash "${WORKER_SCRIPT}"
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

summary_tsv="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_summary.tsv"
printf 'step\toverall_success_pct\tlibero_spatial_pct\tlibero_object_pct\tlibero_goal_pct\tlibero_10_pct\ttasks\tepisodes\n' > "${summary_tsv}"

for step in ${SWEEP_STEPS}; do
  log_root="${RUN_ROOT}/${EVAL_OUTPUT_PREFIX}_steps_${step}/logs"
  summary_path="${log_root}/libero_40task_summary.txt"
  episode_csv="${log_root}/libero_40task_episode_results.csv"
  episode_jsonl="${log_root}/libero_40task_episode_results.jsonl"
  if ! "${LIBERO_PYTHON}" examples/simBenchmarks/LIBERO/eval_files/summarize_libero_success.py \
      "${log_root}" --chunked --require-ok-marker \
      --episode-csv "${episode_csv}" --episode-jsonl "${episode_jsonl}" \
      | tee "${summary_path}"; then
    failed=1
    continue
  fi

  "${LIBERO_PYTHON}" - "${step}" "${episode_csv}" >> "${summary_tsv}" <<'PY'
import csv, sys
from collections import defaultdict

step, path = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path, newline="")))
by_suite = defaultdict(list)
for row in rows:
    raw = str(row.get("success", "")).strip().lower()
    success = raw in {"1", "1.0", "true", "yes"}
    by_suite[row["suite"]].append(success)

def pct(values):
    return 100.0 * sum(values) / len(values) if values else float("nan")

all_values = [value for values in by_suite.values() for value in values]
suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
task_keys = {(row["suite"], row.get("task_id", row.get("task", ""))) for row in rows}
values = [step, pct(all_values), *(pct(by_suite[s]) for s in suites), len(task_keys), len(rows)]
print("\t".join(str(v) for v in values))
PY
done

echo "[$(date)] 8gpu eval finished failed=${failed} summary=${summary_tsv}" | tee -a "${MASTER_LOG}"
exit "${failed}"
