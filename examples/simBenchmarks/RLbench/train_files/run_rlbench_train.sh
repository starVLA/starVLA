#!/usr/bin/env bash
set -euo pipefail

# Resolve the native RLBench action dimension from a launched scene, then pass
# it to a compatible StarVLA VLA training invocation. Dataset actions and
# normalization statistics must have been generated with this same config.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STARVLA_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
STARVLA_DIR="${STARVLA_DIR:-$DEFAULT_STARVLA_DIR}"
RLBENCH_ENV_CONFIG="${RLBENCH_ENV_CONFIG:-$STARVLA_DIR/examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:?Set TRAIN_CONFIG to a StarVLA training YAML.}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-$STARVLA_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-$STARVLA_DIR/starVLA/training/train_starvla.py}"

cd "$STARVLA_DIR"
# Keep launcher options separate from StarVLA dot-list overrides.
TRAIN_ARGS=()
while (($#)); do
  case "$1" in
    --accelerate-config)
      [[ $# -ge 2 ]] || { echo "--accelerate-config requires a path" >&2; exit 2; }
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    --training-script)
      [[ $# -ge 2 ]] || { echo "--training-script requires a path" >&2; exit 2; }
      TRAINING_SCRIPT="$2"
      shift 2
      ;;
    *)
      TRAIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$RLBENCH_ENV_CONFIG" ]]; then
  echo "RLBench environment config does not exist: $RLBENCH_ENV_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$TRAIN_CONFIG" ]]; then
  echo "StarVLA training config does not exist: $TRAIN_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
  echo "Accelerate config does not exist: $ACCELERATE_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
  echo "Training script does not exist: $TRAINING_SCRIPT" >&2
  exit 1
fi

ACTION_DIM="$({
  python examples/simBenchmarks/RLbench/eval_files/resolve_action_dim.py --config "$RLBENCH_ENV_CONFIG"
} | sed -n 's/^RLBENCH_ACTION_DIM=//p' | tail -n 1)"
if [[ ! "$ACTION_DIM" =~ ^[1-9][0-9]*$ ]]; then
  echo "Failed to resolve a numeric RLBench action dimension from ${RLBENCH_ENV_CONFIG}" >&2
  exit 1
fi
echo "Resolved RLBench action_dim=$ACTION_DIM from $RLBENCH_ENV_CONFIG"
echo "Using StarVLA directory: $STARVLA_DIR"
echo "Using Accelerate config: $ACCELERATE_CONFIG"
echo "Using training script: $TRAINING_SCRIPT"

ACCELERATE_ARGS=()
if [[ -n "${NUM_PROCESSES:-}" ]]; then
  ACCELERATE_ARGS+=(--num_processes "$NUM_PROCESSES")
fi

accelerate launch \
  "${ACCELERATE_ARGS[@]}" \
  --config_file "$ACCELERATE_CONFIG" \
  "$TRAINING_SCRIPT" \
  --config_yaml "$TRAIN_CONFIG" \
  --framework.action_model.action_dim "$ACTION_DIM" \
  "${TRAIN_ARGS[@]}"
