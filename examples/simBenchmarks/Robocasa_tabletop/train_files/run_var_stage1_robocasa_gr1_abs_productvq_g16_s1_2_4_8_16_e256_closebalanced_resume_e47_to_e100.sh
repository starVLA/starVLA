#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../../.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1

PYTHON_BIN="$(pwd)/.venv/bin/python"
CONFIG_YAML="examples/simBenchmarks/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_abs_productvq_g16_s1_2_4_8_16_e256_closebalanced_resume_e47_to_e100.yaml"
OUTPUT_DIR="/root/feihong/starVLA/Checkpoints/var_stage1_robocasa_gr1_abs_productvq_g16_s124816_e256_closebalanced_resume_e47_to_e100"
EXPECTED_HASHES="${OUTPUT_DIR}/QUEUED_SOURCE_SHA256SUMS"
GUARDED_TRAINER="starVLA/training/train_var_stage1_guarded.py"

mkdir -p "${OUTPUT_DIR}"
exec 9>"${OUTPUT_DIR}/run.lock"
if ! flock -n 9; then
  echo "Another Stage-1 process already owns ${OUTPUT_DIR}/run.lock." >&2
  exit 1
fi

exec > >(tee -a "${OUTPUT_DIR}/train.log") 2>&1
echo "[$(date --iso-8601=seconds)] Starting guarded RoboCasa Stage-1 launcher."

if [[ -f "${OUTPUT_DIR}/STOPPED_EARLY.json" ]]; then
  echo "Run already stopped by its MAE guardrail: ${OUTPUT_DIR}/STOPPED_EARLY.json" >&2
  exit 2
fi
if [[ -f "${OUTPUT_DIR}/COMPLETED.json" ]]; then
  echo "Run already completed: ${OUTPUT_DIR}/COMPLETED.json" >&2
  exit 0
fi

for required in   "${PYTHON_BIN}"   "${CONFIG_YAML}"   "${GUARDED_TRAINER}"   "${OUTPUT_DIR}/latest.ckpt"   "${OUTPUT_DIR}/epoch_047.ckpt"   "${OUTPUT_DIR}/resume_seed_epoch_047.ckpt"   "${OUTPUT_DIR}/adaptive_task_weights.json"   "${OUTPUT_DIR}/history.json"   "${OUTPUT_DIR}/reconstruction_by_task_epoch_047.json"   "${OUTPUT_DIR}/static_task_balance_weights.json"   "${EXPECTED_HASHES}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required guarded-resume input: ${required}" >&2
    exit 1
  fi
done

sha256sum --check --strict "${EXPECTED_HASHES}"

if [[ "${ALLOW_SHARED_GPU:-0}" != "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; refusing fail-open GPU detection." >&2
    exit 1
  fi
  if ! BUSY_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"; then
    echo "nvidia-smi query failed; refusing fail-open GPU detection." >&2
    exit 1
  fi
  BUSY_PIDS="$(printf '%s\n' "${BUSY_PIDS}" | tr -d ' ' | sed '/^$/d')"
  if [[ -n "${BUSY_PIDS}" ]]; then
    echo "CUDA device is busy with existing compute process(es): ${BUSY_PIDS}." >&2
    exit 1
  fi
else
  echo "ALLOW_SHARED_GPU=1: intentionally sharing CUDA device with existing workload."
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
fi

"${PYTHON_BIN}" - "${CONFIG_YAML}" "${OUTPUT_DIR}" <<'PY'
import json
import os
import sys

import torch
from omegaconf import OmegaConf

config_path, output_dir = sys.argv[1:]
cfg = OmegaConf.load(config_path)
expected_latest = os.path.join(output_dir, "latest.ckpt")
expected_baseline = os.path.join(output_dir, "reconstruction_by_task_epoch_047.json")
checks = {
    "experiment.output_dir": (str(cfg.experiment.output_dir), output_dir),
    "train.resume_checkpoint": (str(cfg.train.resume_checkpoint), expected_latest),
    "train.task_mae_guardrail.baseline_task_metrics_path": (
        str(cfg.train.task_mae_guardrail.baseline_task_metrics_path),
        expected_baseline,
    ),
}
for name, (actual, expected) in checks.items():
    if os.path.realpath(actual) != os.path.realpath(expected):
        raise SystemExit(f"{name} mismatch: actual={actual}, expected={expected}")
if int(cfg.train.epochs) != 100:
    raise SystemExit(f"Expected total epochs=100, got {cfg.train.epochs}")
if float(cfg.train.learning_rate) != 4e-5:
    raise SystemExit(f"Expected truthful learning_rate=4e-5, got {cfg.train.learning_rate}")
if not bool(cfg.train.task_mae_guardrail.enabled):
    raise SystemExit("Task-MAE guardrail is disabled.")

checkpoint = torch.load(expected_latest, map_location="cpu", weights_only=False)
epoch = int(checkpoint["epoch"])
history = list(checkpoint.get("history", []))
lrs = [float(group["lr"]) for group in checkpoint["optimizer_state_dict"]["param_groups"]]
if epoch < 47 or epoch >= 100:
    raise SystemExit(f"Unexpected resume epoch {epoch} in {expected_latest}")
if not history or int(history[-1]["epoch"]) != epoch:
    raise SystemExit(f"Checkpoint history does not end at checkpoint epoch {epoch}.")
if lrs != [4e-5]:
    raise SystemExit(f"Unexpected AdamW LR {lrs} in {expected_latest}")
if isinstance(history[-1].get("task_mae_guardrail"), dict) and history[-1]["task_mae_guardrail"].get("should_stop"):
    raise SystemExit(f"Checkpoint epoch {epoch} already contains a terminal guardrail decision.")

with open(expected_baseline, "r", encoding="utf-8") as handle:
    baseline = json.load(handle)
counts = {name: int(record["count"]) for name, record in baseline.get("tasks", {}).items()}
if len(counts) != 24 or sum(counts.values()) != 5_660_058 or any(value <= 0 for value in counts.values()):
    raise SystemExit("Epoch-47 baseline task counts failed the 24-task/5,660,058-sample contract.")

external_history_path = os.path.join(output_dir, "history.json")
with open(external_history_path, "r", encoding="utf-8") as handle:
    external_history = json.load(handle)
if not external_history or int(external_history[-1]["epoch"]) != epoch:
    raise SystemExit(
        f"External history does not end at checkpoint epoch {epoch}: {external_history_path}"
    )

adaptive_path = os.path.join(output_dir, "adaptive_task_weights.json")
with open(adaptive_path, "r", encoding="utf-8") as handle:
    adaptive = json.load(handle)
embedded_adaptive = history[-1].get("adaptive_task_weights")
if embedded_adaptive is None and int(adaptive.get("epoch", -1)) != epoch:
    raise SystemExit(
        f"External adaptive weights epoch {adaptive.get('epoch')} does not match checkpoint epoch {epoch}."
    )
adaptive_weights = embedded_adaptive if embedded_adaptive is not None else adaptive.get("weights", {})
if set(adaptive_weights) != set(counts):
    raise SystemExit(
        f"Adaptive task keys do not match the 24-task baseline: "
        f"adaptive={len(adaptive_weights)}, baseline={len(counts)}"
    )

static_path = os.path.join(output_dir, "static_task_balance_weights.json")
with open(static_path, "r", encoding="utf-8") as handle:
    static_weights = json.load(handle)
if set(static_weights) != set(counts):
    raise SystemExit(
        f"Static task keys do not match the 24-task baseline: "
        f"static={len(static_weights)}, baseline={len(counts)}"
    )
print(f"Verified guarded resume: epoch={epoch}, optimizer_lr={lrs}, tasks={len(counts)}, samples={sum(counts.values())}.")
PY

if [[ ! -d "${OUTPUT_DIR}/source_snapshot_initial" ]]; then
  SNAPSHOT_TMP="$(mktemp -d "${OUTPUT_DIR}/source_snapshot_initial.tmp.XXXXXX")"
  cp "${CONFIG_YAML}" "${SNAPSHOT_TMP}/"
  cp "${GUARDED_TRAINER}" "${SNAPSHOT_TMP}/"
  cp starVLA/training/stage1_task_mae_guardrail.py "${SNAPSHOT_TMP}/"
  cp starVLA/dataloader/var_stage1_action_dataset.py "${SNAPSHOT_TMP}/"
  cp starVLA/model/modules/action_tokenizer/var_action_tokenizer.py "${SNAPSHOT_TMP}/"
  cp starVLA/utils/action_spec.py "${SNAPSHOT_TMP}/"
  cp examples/simBenchmarks/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml "${SNAPSHOT_TMP}/"
  cp "$0" "${SNAPSHOT_TMP}/"
  cp "${OUTPUT_DIR}/reconstruction_by_task_epoch_047.json" "${SNAPSHOT_TMP}/"
  git rev-parse HEAD > "${SNAPSHOT_TMP}/GIT_COMMIT" 2>/dev/null || true
  git status --short > "${SNAPSHOT_TMP}/GIT_STATUS" 2>/dev/null || true
  (
    cd "${SNAPSHOT_TMP}"
    sha256sum ./*.py ./*.yaml ./*.sh ./reconstruction_by_task_epoch_047.json ./GIT_COMMIT ./GIT_STATUS > SHA256SUMS
  )
  mv "${SNAPSHOT_TMP}" "${OUTPUT_DIR}/source_snapshot_initial"
else
  (
    cd "${OUTPUT_DIR}/source_snapshot_initial"
    sha256sum --check --strict SHA256SUMS
  )
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1: all guarded-resume checks passed; trainer not started."
  exit 0
fi

echo "[$(date --iso-8601=seconds)] Executing in-loop guarded trainer on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
exec "${PYTHON_BIN}" "${GUARDED_TRAINER}" --config_yaml "${CONFIG_YAML}"
