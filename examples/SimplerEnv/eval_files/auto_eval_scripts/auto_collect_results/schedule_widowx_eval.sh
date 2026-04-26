#!/bin/bash
# ==============================================================================
# schedule_widowx_eval.sh
#
# Batch-schedule WidowX (BridgeData v2) evaluations for every checkpoint that
# does not yet have a complete set of result logs.
#
# For each `<ROOT_BASE>/<DIR_GLOB>/checkpoints/steps_<step>_pytorch_model.pt`:
#   - Verify the four expected log files (one per Bridge task) exist.
#   - If any are missing, srun-launch `star_bridge.sh` to evaluate that ckpt.
#
# Usage:
#   ./schedule_widowx_eval.sh                               # uses default DIR_GLOB
#   ./schedule_widowx_eval.sh '0427_oxe_bridge_rt_1_QwenPI_v3'
#   DIR_GLOB='0427_oxe*' ./schedule_widowx_eval.sh
# ==============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Parent directory holding all training experiments.
ROOT_BASE="${ROOT_BASE:-$PROJECT_ROOT/results/Checkpoints}"

# Experiment directory pattern (relative to ROOT_BASE). First positional arg
# overrides the default; can also be set via env var DIR_GLOB.
DIR_GLOB="${1:-${DIR_GLOB:-0427_oxe_bridge_rt_1_QwenPI_v3}}"

# srun partition / gpus
SLURM_PARTITION="${SLURM_PARTITION:-si}"
SLURM_GRES="${SLURM_GRES:-gpu:4}"

# Bridge eval entry-point.
SCRIPT_PATH="${SCRIPT_PATH:-$PROJECT_ROOT/examples/SimplerEnv/eval_files/auto_eval_scripts/star_bridge.sh}"

# Expected result-log filename suffixes (without the steps_${step}_ prefix).
LOG_SUFFIXES=(
  "pytorch_model_infer_PutCarrotOnPlateInScene-v0.log.run1"
  "pytorch_model_infer_PutEggplantInBasketScene-v0.log.run1"
  "pytorch_model_infer_PutSpoonOnTableClothInScene-v0.log.run1"
  "pytorch_model_infer_StackGreenCubeOnYellowCubeBakedTexInScene-v0.log.run1"
)

echo "ROOT_BASE = $ROOT_BASE"
echo "DIR_GLOB  = $DIR_GLOB"
echo "SCRIPT    = $SCRIPT_PATH"
echo

shopt -s nullglob
matched_any=false
for checkpoints_dir in "$ROOT_BASE"/${DIR_GLOB}/checkpoints; do
  matched_any=true
  echo "==> $checkpoints_dir"

  [ -d "$checkpoints_dir" ] || { echo "  (not a directory, skip)"; continue; }
  if [[ "$checkpoints_dir" == *"without"* ]]; then
    echo "  Skipping (contains 'without')"
    continue
  fi

  cd "$checkpoints_dir" || continue

  for pt_file in steps_*_pytorch_model.pt; do
    [ -e "$pt_file" ] || continue

    # Filename convention: steps_<step>_pytorch_model.pt
    step=$(echo "$pt_file" | cut -d'_' -f2)

    all_logs_exist=true
    for suffix in "${LOG_SUFFIXES[@]}"; do
      log_file="steps_${step}_${suffix}"
      if [ ! -f "$log_file" ]; then
        all_logs_exist=false
        break
      fi
    done

    if $all_logs_exist; then
      echo "  ✔ logs complete: $pt_file (skip)"
    else
      echo "  ✘ logs missing: $pt_file -> launching"
      MODEL_PATH="$checkpoints_dir/$pt_file"
      nohup srun -p "$SLURM_PARTITION" --gres="$SLURM_GRES" \
        /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" &
      sleep 10
    fi
  done

  cd - >/dev/null
done

$matched_any || echo "⚠️  No directories matched '$ROOT_BASE/$DIR_GLOB'"
