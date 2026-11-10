#!/usr/bin/env bash
# Train one 15-task Evo-DexBench policy. Run from the StarVLA repository root after data_preparation.sh.
#   FRAMEWORK=QwenOFT|EvoWanOFT|EvoCosmosGR00T  EMBODIMENT=single|dual  NUM_PROCESSES=<GPUs>
# Extra arguments are passed to the trainer as config overrides, as `--key value` pairs, e.g.
#   --trainer.max_train_steps 300000 (key=value without the leading -- is ignored by train_starvla.py).
# EvoWanOFT and EvoCosmosGR00T need the model extension from the checkpoint release (see README).
set -euo pipefail

FRAMEWORK="${FRAMEWORK:-QwenOFT}"
EMBODIMENT="${EMBODIMENT:-single}"
RECIPES=examples/simBenchmarks/EvoDexBench/train_files

case "${FRAMEWORK}" in
  QwenOFT)
    recipe="${RECIPES}/${EMBODIMENT}15_qwenoft_h50_q99.yaml"
    entry=(starVLA/training/train_starvla.py)
    num_processes="${NUM_PROCESSES:-2}" ;;
  EvoWanOFT)
    recipe="${RECIPES}/${EMBODIMENT}15_wan_h8_mean_q99.yaml"
    entry=(-m evodex_wm4a.train --framework EvoWanOFT)
    num_processes="${NUM_PROCESSES:-1}" ;;
  EvoCosmosGR00T)
    recipe="${RECIPES}/${EMBODIMENT}15_cosmos_h8_mean_q99.yaml"
    entry=(-m evodex_wm4a.train --framework EvoCosmosGR00T)
    num_processes="${NUM_PROCESSES:-1}" ;;
  *) echo "FRAMEWORK must be QwenOFT, EvoWanOFT or EvoCosmosGR00T" >&2; exit 2 ;;
esac

export STARVLA_ROOT="${STARVLA_ROOT:-$(pwd)}"
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
# The released checkpoints were trained with plain DDP (no DeepSpeed) and bf16 mixed precision.
export STARVLA_DISABLE_DEEPSPEED="${STARVLA_DISABLE_DEEPSPEED:-1}"
accelerate launch --num_processes "${num_processes}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  "${entry[@]}" --config_yaml "${recipe}" "$@"
