#!/bin/bash
# see_sr_auto.sh
#
# Summarises success-rate results from all VLA-Arena evaluation log files
# found under a given checkpoint directory.
#
# Usage:
#   bash see_sr_auto.sh <model_root>
#
# Example:
#   bash see_sr_auto.sh results/Checkpoints/vla_arena_qwenoft_all

model_root=${1:-results/Checkpoints/vla_arena_qwenoft_all}
log_dir="${model_root}/logs"

if [ ! -d "${log_dir}" ]; then
    echo "Log directory not found: ${log_dir}"
    exit 1
fi

echo "======================================================"
echo "VLA-Arena evaluation results under: ${log_dir}"
echo "======================================================"

for suite_dir in "${log_dir}"/*/; do
    suite=$(basename "${suite_dir}")
    echo ""
    echo "--- ${suite} ---"
    for logfile in "${suite_dir}"*.log; do
        [ -f "${logfile}" ] || continue
        sr=$(grep -oP "Final SR: \K[0-9.]+" "${logfile}" | tail -1)
        cost=$(grep -oP "avg_cost=\K[0-9.]+" "${logfile}" | tail -1)
        echo "  $(basename "${logfile%.log}"):  SR=${sr:-N/A}  avg_cost=${cost:-N/A}"
    done
done

echo ""
echo "======================================================"
