#!/usr/bin/env bash
set -euo pipefail

# Short ABC-only training sweep for a 4-GPU machine.
# Intended for throughput exploration while a separate machine runs eval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

cd "${STARVLA_ROOT}"

TS="${TS:-$(date +%m%d_%H%M%S)}"
MEMBER="${MEMBER:-WMH}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-300}"
WARMUP_SKIP_STEPS="${WARMUP_SKIP_STEPS:-30}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100000000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
BASE_LOG_DIR="${LOG_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/${MEMBER}/logs/train_speed_sweep_${TS}}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/${MEMBER}/runs_speed_sweep}"
DRY_RUN="${DRY_RUN:-0}"

# Semicolon-separated cases: name,batch_size,num_workers,prefetch_factor,grad_accum
SWEEP_CASES="${SWEEP_CASES:-\
bs12_w8_pf2_ga1,12,8,2,1;\
bs16_w8_pf2_ga1,16,8,2,1;\
bs20_w8_pf2_ga1,20,8,2,1;\
bs16_w12_pf2_ga1,16,12,2,1;\
bs16_w12_pf4_ga1,16,12,4,1}"

mkdir -p "${BASE_LOG_DIR}" "${RUN_ROOT_DIR}"

export STRICT_ASSETS="${STRICT_ASSETS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_HOME="${TORCH_HOME:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/cache/xdg}"

echo "[speed-sweep] gpu_ids=${GPU_IDS}"
echo "[speed-sweep] num_processes=${NUM_PROCESSES}"
echo "[speed-sweep] max_train_steps=${MAX_TRAIN_STEPS}"
echo "[speed-sweep] log_dir=${BASE_LOG_DIR}"
echo "[speed-sweep] run_root_dir=${RUN_ROOT_DIR}"
echo "[speed-sweep] cases=${SWEEP_CASES}"

IFS=';' read -r -a CASE_ARRAY <<< "${SWEEP_CASES}"

for case_spec in "${CASE_ARRAY[@]}"; do
  [[ -n "${case_spec}" ]] || continue
  IFS=',' read -r case_name batch_size num_workers prefetch_factor grad_accum <<< "${case_spec}"
  if [[ -z "${case_name:-}" || -z "${batch_size:-}" || -z "${num_workers:-}" || -z "${prefetch_factor:-}" || -z "${grad_accum:-}" ]]; then
    echo "Bad sweep case: ${case_spec}" >&2
    exit 2
  fi

  run_id="speed4g_${case_name}_${TS}"
  log_path="${BASE_LOG_DIR}/${run_id}.log"
  echo "[speed-sweep] starting ${run_id}"

  GPU_IDS="${GPU_IDS}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  RUN_ID="${run_id}" \
  RUN_ROOT_DIR="${RUN_ROOT_DIR}" \
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS}" \
  BATCH_SIZE="${batch_size}" \
  GRADIENT_ACCUMULATION_STEPS="${grad_accum}" \
  DATALOADER_NUM_WORKERS="${num_workers}" \
  DATALOADER_PREFETCH_FACTOR="${prefetch_factor}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  LOGGING_FREQUENCY="${LOGGING_FREQUENCY}" \
  SKIP_FINAL_SAVE=1 \
  DRY_RUN="${DRY_RUN}" \
    bash examples/calvin_autoresearch/scripts/run_train_abc_pretrain_h200.sh \
    2>&1 | tee "${log_path}"

  echo "[speed-sweep] finished ${run_id}; log=${log_path}"
done

PYTHONDONTWRITEBYTECODE=1 python - "${BASE_LOG_DIR}" "${WARMUP_SKIP_STEPS}" <<'PY'
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
warmup_skip = int(sys.argv[2])
rows = []

for log_path in sorted(root.glob("*.log")):
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    data_times = [float(x) for x in re.findall(r"data_times=([0-9.]+)", text)]
    model_times = [float(x) for x in re.findall(r"model_times=([0-9.]+)", text)]
    if len(data_times) > warmup_skip:
        data_times = data_times[warmup_skip:]
    if len(model_times) > warmup_skip:
        model_times = model_times[warmup_skip:]
    if not data_times or not model_times:
        rows.append((log_path.name, None))
        continue
    mean_data = statistics.mean(data_times)
    mean_model = statistics.mean(model_times)
    mean_step = mean_data + mean_model
    rows.append(
        (
            log_path.name,
            {
                "samples": min(len(data_times), len(model_times)),
                "data": mean_data,
                "model": mean_model,
                "step": mean_step,
                "steps_per_hour": 3600 / mean_step if mean_step > 0 else 0.0,
            },
        )
    )

print("\n[speed-sweep] summary")
for name, row in rows:
    if row is None:
        print(f"{name}: no timing samples")
    else:
        print(
            f"{name}: samples={row['samples']} "
            f"data={row['data']:.4f}s model={row['model']:.4f}s "
            f"step={row['step']:.4f}s steps/hour={row['steps_per_hour']:.1f}"
        )
PY
