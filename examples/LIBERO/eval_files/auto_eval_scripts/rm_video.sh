#!/bin/bash

# Delete all results/Checkpoints/*/videos/libero_* directories

# Set target directory
TARGET_DIR="results/Checkpoints"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Searching for $TARGET_DIR/*/videos/libero_* directories..."

# Method 1: Use a more direct search approach
MATCHING_DIRS=()
for checkpoints_dir in "$TARGET_DIR"/*/; do
    videos_dir="${checkpoints_dir}videos/"
    if [ -d "$videos_dir" ]; then
        for libero_dir in "$videos_dir"libero_*/; do
            if [ -d "$libero_dir" ]; then
                # Remove trailing slash
                dir="${libero_dir%/}"
                MATCHING_DIRS+=("$dir")
                echo "Found: $dir"
            fi
        done
    fi
done

# If no directories found
if [ ${#MATCHING_DIRS[@]} -eq 0 ]; then
    echo "No libero_* directories found under $TARGET_DIR/*/videos/"
    exit 0
fi

echo ""
echo "Total directories found: ${#MATCHING_DIRS[@]}"
echo ""

# Confirm before deletion
read -p "Confirm deletion of all above directories? (y/n): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting deletion..."
    for dir in "${MATCHING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            rm -rf "$dir"
            echo "Deleted: $dir"
        fi
    done
    echo "Deletion complete!"
else
    echo "Deletion cancelled"
    exit 0
fi