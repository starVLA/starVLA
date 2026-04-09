#!/bin/bash

# Specify the root directory for log files
log_dir="${1:-results/Checkpoints/0405_libero4in1_CosmoPredict2GR00T}"

# Iterate over all log files in the specified directory
last_Folder=""
find "$log_dir" -type f -name "*.log" | while read -r log_file; do
    # Extract the last "Total success rate" value from the log file
    success_rate=$(grep "INFO     | >> Total success rate:" "$log_file" | tail -n 1)
    
    # If a match is found, output the log file path and the corresponding success rate
    if [ -n "$success_rate" ]; then
        echo "Folder: $(basename "$(dirname "$log_file")")"
        echo "File: $(basename "$log_file")"
        echo "$success_rate"
        echo
    fi
done