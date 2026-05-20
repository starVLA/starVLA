#!/usr/bin/env bash
set -euo pipefail

# Finalize the newest WMH 8k state8+connector n1000 parallel eval report.
# This avoids manually pasting a very long EVAL_DIR path into the shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${STARVLA_ROOT}/../.." && pwd)}"

if [[ -f "${PROJECT_ROOT}/starvla_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/starvla_env.sh"
fi

MEMBER="${MEMBER:-WMH}"
SHARED_ROOT="${SHARED_ROOT:-/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin}"
REPORT_ROOT="${REPORT_ROOT:-${SHARED_ROOT}/members/${MEMBER}/reports}"
PATTERN="${PATTERN:-eval_8k_state8_connector_steps8000*_d_n1000_*}"
KILL_SERVERS="${KILL_SERVERS:-1}"

if [[ -n "${EVAL_DIR:-}" ]]; then
  target="${EVAL_DIR}"
else
  target="$(
    find "${REPORT_ROOT}" -maxdepth 1 -type d -name "${PATTERN}" -printf '%T@ %p\n' \
      | sort -nr \
      | head -1 \
      | cut -d' ' -f2-
  )"
fi

if [[ -z "${target}" ]]; then
  echo "No eval report found under ${REPORT_ROOT} matching ${PATTERN}" >&2
  exit 2
fi

echo "[finalize-latest-8k-n1000] eval_dir=${target}"
KILL_SERVERS="${KILL_SERVERS}" EVAL_DIR="${target}" bash "${SCRIPT_DIR}/finalize_parallel_eval_dir.sh"
