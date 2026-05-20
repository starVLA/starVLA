#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/examples/calvin_autoresearch/scripts/run_wmh_abc_augmented_moe_adaptive_4gpu.sh" "$@"
