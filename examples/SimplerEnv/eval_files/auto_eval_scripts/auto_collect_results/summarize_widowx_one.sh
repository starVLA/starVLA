#!/bin/bash
# ==============================================================================
# summarize_widowx_one.sh
#
# Parse the BridgeData v2 evaluation logs of ONE experiment directory and
# produce raw_success.txt + success_summary.csv + success_plot.png.
#
# Usage:
#   ./summarize_widowx_one.sh <experiment_dir> [rm_logs:true|false]
#
# Expected layout:
#   <experiment_dir>/checkpoints/*.log.*    (produced by star_bridge.sh)
# Produces:
#   <experiment_dir>/success_summary/{raw_success.txt,success_summary.csv,success_plot.png}
# ==============================================================================

LOG_DIR="$1/checkpoints"
OUT_DIR="$1/success_summary"
mkdir -p "$OUT_DIR"
RM_LOGS="${2:-false}"

RAW_TXT="$OUT_DIR/raw_success.txt"
CSV_OUT="$OUT_DIR/success_summary.csv"
PNG_OUT="$OUT_DIR/success_plot.png"

# check input
if [ -z "$LOG_DIR" ] || [ -z "$OUT_DIR" ]; then
  echo "Usage: $0 <log_directory> <output_directory> [true|false]"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "📁 Logs: $LOG_DIR"
echo "📤 Output: $OUT_DIR"

# Step 1: extract success rate logs
echo "🔍 Extracting success scores..."
OUTPUT=""
for file in "$LOG_DIR"/*.log.*; do
  if [ -f "$file" ]; then
    success=$(grep -E "Average success" "$file" | awk '{print $NF}')
    if [ -n "$success" ]; then
      line="$(basename "$file") → Average success: $success"
      echo "$line"
      OUTPUT+="$line"$'\n'
    else
      echo "$(basename "$file") → ❌ Not found"
      if ${RM_LOGS}; then
        rm -f "$file"
        echo "🗑️  Deleted: $file"
      fi
    fi
  fi
done

# ✅ save logs to TXT file (fix key points)
echo "$OUTPUT" > "$RAW_TXT"

# Resolve plotting script relative to this file (portable across machines).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${PYTHON_SCRIPT:-$SCRIPT_DIR/plot_widowx_results.py}"

# Step 3: execute Python analysis script
echo "🐍 Running Python analysis ($PYTHON_SCRIPT)..."
python3 "$PYTHON_SCRIPT" "$RAW_TXT" "$CSV_OUT" "$PNG_OUT"
