#!/usr/bin/env bash
set -euo pipefail

: "${EVAL_DIR:?Set EVAL_DIR to a parallel eval report directory}"
KILL_SERVERS="${KILL_SERVERS:-1}"

if [[ ! -d "${EVAL_DIR}" ]]; then
  echo "EVAL_DIR does not exist: ${EVAL_DIR}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${STARVLA_ROOT}"
PYTHONDONTWRITEBYTECODE=1 python examples/calvin_autoresearch/scripts/aggregate_parallel_eval_dir.py "${EVAL_DIR}"

if [[ "${KILL_SERVERS}" == "1" ]]; then
  for pid_file in "${EVAL_DIR}"/worker_*/server.pid; do
    [[ -f "${pid_file}" ]] || continue
    pid="$(cat "${pid_file}")"
    kill "${pid}" 2>/dev/null || true
  done
fi
