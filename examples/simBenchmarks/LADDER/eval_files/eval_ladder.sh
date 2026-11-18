#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
LADDER_ROOT="${LADDER_ROOT:-$HOME/ladder}"
SUITE="${SUITE:-simpler}"
TASKS="${TASKS:-all}"
PORT="${PORT:-10093}"
HOST="${HOST:-127.0.0.1}"
PROMPT_MODE="${PROMPT_MODE:-flat}"
RESULTS_FILE="${RESULTS_FILE:-${STARVLA_DIR}/results/ladder_${SUITE}_${PROMPT_MODE}.json}"

export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

python "${STARVLA_DIR}/examples/simBenchmarks/LADDER/eval_files/eval_ladder.py" \
  --suite "${SUITE}" \
  --tasks "${TASKS}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --prompt-mode "${PROMPT_MODE}" \
  --ladder-root "${LADDER_ROOT}" \
  --results-file "${RESULTS_FILE}" \
  "$@"
