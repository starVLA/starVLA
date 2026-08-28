#!/usr/bin/env bash
set -euo pipefail

# FAST-only LIBERO eval entrypoint. Keep this separate from the stage2 eval
# chain so FAST action-token generation / clipping changes do not affect
# QwenVAR stage2 smoke and training eval.

export IMAGE_VIEWS="${IMAGE_VIEWS:-primary+wrist}"
# QwenFast was trained on 224x224 LIBERO frames from the LeRobot dataloader.
export POLICY_IMAGE_SIZE="${POLICY_IMAGE_SIZE:-224}"
# Match QwenFast training/offline-MSE inference by default. Hard action-token
# constraints produced short malformed FAST token streams that often decoded to
# all-zero actions.
export CONSTRAIN_TO_ACTION_TOKENS="${CONSTRAIN_TO_ACTION_TOKENS:-0}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
export CLIP_NORMALIZED_ACTIONS="${CLIP_NORMALIZED_ACTIONS:-1}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-eval_fast_fixed}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_local_eval_once.sh" "$@"
