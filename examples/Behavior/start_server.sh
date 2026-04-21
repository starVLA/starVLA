#!/bin/bash

# Debug: Print the current Python environment
echo "Using Python: $(which python)"

### MANUALLY SET THESE ###

# set necessary environment variables
export star_vla_python=/opt/conda/envs/starVLA/bin/python
export sim_python=/opt/conda/envs/behavior/bin/python
export BEHAVIOR_ASSET_PATH=/workspace/llavavla0/BEHAVIOR-1K/datasets
export PYTHONPATH=$(pwd):${PYTHONPATH}



# set model path and port
MODEL_PATH="/workspace/llavavla0/playground/Checkpoints/BEHAVIOR-QwenDual-Pretrained-224/checkpoints/steps_300000_pytorch_model.pt"
PORT=10197
WRAPPERS="DefaultWrapper"
USE_STATE=True  # whether to use state as part of the observation

# Configure task name
TASK_NAME="turning_on_radio"  
### END OF MANUALLY SETUP ###

# Force Vulkan to use only the NVIDIA ICD to avoid duplicate ICDs seen by the loader
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
# Prefer NVIDIA GLX vendor when any GL deps are touched
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Start server
echo "▶️ Starting server on port ${PORT}..."
CUDA_VISIBLE_DEVICES=5 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${MODEL_PATH} \
    --port ${PORT} \
    --is_debug \
    --use_bf16