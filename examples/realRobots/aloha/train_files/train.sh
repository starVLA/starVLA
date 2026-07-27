#!/bin/bash
# RoboTTT training launcher for the ALOHA bimanual setup.
# Uses train_starvla.py (VLA SFT recipe). All heavy settings live in the YAML;
# the CLI flags below are convenience overrides / environment knobs.

set -e
source .venv/bin/activate
# ========== Required parameter ==========
config_yaml=./examples/realRobots/aloha/train_files/aloha.yaml  # Training config file (required)

# ========== Optional overrides (CLI takes priority over YAML values) ==========
Framework_name=RoboTTT
# RoboTTT warm-starts from NVIDIA GR00T N1.7, whose backbone is Cosmos-Reason2-2B
# (gated HF model, hidden_size 2048). Do NOT override to Qwen2.5/Qwen3-VL-4B here —
# the DiT cross-attn K/V weights in the converted checkpoint are [., 2048] and only
# load cleanly with the 2048-dim Cosmos-Reason2 backbone (see aloha.yaml comments).
base_vlm=/inspire/qb-ilm/project/robot-reasoning/public/model/nvidia/Cosmos-Reason2-2B
data_root=data
data_mix=task350
run_root_dir=./results/Checkpoints
run_id=aloha_robottt
# RoboTTT trains on trajectories: per-device batch must be 1 (B=1 for long context).
per_device_batch_size=1

# NCCL / communication settings (mirrors run_libero_train.sh)
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

# Create output directory and stash this script there
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

# Use all visible GPUs by default, override with NUM_PROCESSES env var if needed
num_processes=${NUM_PROCESSES:-$(nvidia-smi -L | wc -l)}

# --config_yaml is the only required argument; all other --xxx flags are optional CLI
# overrides (OmegaConf dotlist, merged on top of the YAML by normalize_dotlist_args).
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id}
