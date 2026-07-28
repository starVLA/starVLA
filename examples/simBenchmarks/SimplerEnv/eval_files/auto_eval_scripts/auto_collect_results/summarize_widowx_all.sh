#!/bin/bash
# ==============================================================================
# summarize_widowx_all.sh
#
# Loop over every experiment directory matching DIR_GLOB and run
# summarize_widowx_one.sh on it.
#
# Usage:
#   ./summarize_widowx_all.sh                       # default DIR_GLOB / RM_LOGS=false
#   ./summarize_widowx_all.sh true                  # also delete unparsable logs
#   DIR_GLOB='0427_oxe_bridge_rt_1_QwenPI_v3' ./summarize_widowx_all.sh
# ==============================================================================
set -uo pipefail


DIR_GLOB=0430*

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

ROOT_BASE="${ROOT_BASE:-$PROJECT_ROOT/results/Checkpoints}"
DIR_GLOB="${DIR_GLOB:-0427_oxe_bridge_rt_1_QwenPI_v3}"
RM_LOGS="${1:-false}"

ONE_SCRIPT="${ONE_SCRIPT:-$SCRIPT_DIR/summarize_widowx_one.sh}"

echo "🔍 ROOT_BASE = $ROOT_BASE"
echo "   DIR_GLOB  = $DIR_GLOB"
echo "   RM_LOGS   = $RM_LOGS"
echo "==========================================="

shopt -s nullglob
matched=false
for dir in "$ROOT_BASE"/${DIR_GLOB}; do
  [ -d "$dir" ] || continue
  matched=true
  echo "📂 Entering: $dir"
  (cd "$dir" && bash "$ONE_SCRIPT" "$dir" "$RM_LOGS")
  echo ""
done

$matched || echo "⚠️  No directories matched '$ROOT_BASE/$DIR_GLOB'"
echo "✅ Done."