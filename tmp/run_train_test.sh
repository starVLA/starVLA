#!/bin/bash
# Smoke tests for VLA-only and VLA+VLM cotrain training after DataLoaderManager changes
# Usage: run on a compute node with 2+ GPUs
#   srun --jobid=<JOB_ID> --overlap --pty bash /home/jye624/Projcets/starVLA/tmp/run_train_test.sh
set -e

# === Conda setup ===
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate starVLA

# === CUDA setup ===
for cuda_path in /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-12.4; do
  if [ -x "${cuda_path}/bin/nvcc" ]; then
    export CUDA_HOME="${cuda_path}"
    export PATH="${cuda_path}/bin:${PATH}"
    export LD_LIBRARY_PATH="${cuda_path}/lib64:${LD_LIBRARY_PATH:-}"
    break
  fi
done

# nvcc wrapper fallback
if ! nvcc --version 2>&1 | grep -q "release"; then
  _WRAPPER_DIR="${CONDA_PREFIX}/cuda_compat/bin"
  mkdir -p "${_WRAPPER_DIR}" 2>/dev/null || true
  _TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.4")
  _MAJOR=$(echo "${_TORCH_CUDA_VER}" | cut -d. -f1)
  _MINOR=$(echo "${_TORCH_CUDA_VER}" | cut -d. -f2)
  cat > "${_WRAPPER_DIR}/nvcc" << NVCC_EOF
#!/bin/bash
echo "nvcc: NVIDIA (R) Cuda compiler driver"
echo "Cuda compilation tools, release ${_MAJOR}.${_MINOR}, V${_TORCH_CUDA_VER}"
NVCC_EOF
  chmod +x "${_WRAPPER_DIR}/nvcc"
  export PATH="${_WRAPPER_DIR}:${PATH}"
  export CUDA_HOME="${CONDA_PREFIX}/cuda_compat"
  echo "[INFO] Created nvcc wrapper: CUDA ${_TORCH_CUDA_VER}"
fi

echo "[INFO] CUDA_HOME=$CUDA_HOME"
nvcc --version 2>/dev/null || echo "[WARN] nvcc not found"

cd /home/jye624/Projcets/starVLA

export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"
mkdir -p "${TRITON_CACHE_DIR}" 2>/dev/null || true
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000

num_processes=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "=== Detected ${num_processes} GPUs ==="

CONFIG_YAML=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
ACCEL_CFG=./starVLA/config/deepseeds/deepspeed_zero2.yaml
BASE_VLM=./playground/Pretrained_models/Qwen3-VL-4B-Instruct

######################################################################
# TEST 1: VLA-only training (train_starvla.py) — 10 steps
######################################################################
echo ""
echo "========================================"
echo "  TEST 1: VLA-only (train_starvla.py)"
echo "========================================"

sg vonneumann1 -c "
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh && \
conda activate starVLA && \
cd /home/jye624/Projcets/starVLA && \
export WANDB_MODE=disabled && \
export TRITON_CACHE_DIR=/tmp/${USER}_triton_cache && \
export NCCL_BLOCKING_WAIT=1 && \
export NCCL_ASYNC_ERROR_HANDLING=1 && \
export NCCL_TIMEOUT=10000 && \
accelerate launch \
  --config_file ${ACCEL_CFG} \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla.py \
  --config_yaml ${CONFIG_YAML} \
  --framework.qwenvl.base_vlm ${BASE_VLM} \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 2 \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 5 \
  --trainer.logging_frequency 2 \
  --run_root_dir ./tmp/smoke_test_output \
  --run_id test_vla_only \
  --wandb_project test \
  --wandb_entity test
"

echo ""
echo ">>> TEST 1: VLA-only PASSED <<<"

######################################################################
# TEST 2: VLA+VLM cotrain (train_starvla_cotrain.py) — 10 steps
######################################################################
echo ""
echo "========================================"
echo "  TEST 2: VLA+VLM cotrain"
echo "========================================"

sg vonneumann1 -c "
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh && \
conda activate starVLA && \
cd /home/jye624/Projcets/starVLA && \
export WANDB_MODE=disabled && \
export TRITON_CACHE_DIR=/tmp/${USER}_triton_cache && \
export NCCL_BLOCKING_WAIT=1 && \
export NCCL_ASYNC_ERROR_HANDLING=1 && \
export NCCL_TIMEOUT=10000 && \
accelerate launch \
  --config_file ${ACCEL_CFG} \
  --num_processes ${num_processes} \
  starVLA/training/train_starvla_cotrain.py \
  --config_yaml ${CONFIG_YAML} \
  --framework.qwenvl.base_vlm ${BASE_VLM} \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_mix libero_goal \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vlm_data.per_device_batch_size 2 \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 999999 \
  --trainer.eval_interval 5 \
  --trainer.logging_frequency 2 \
  --run_root_dir ./tmp/smoke_test_output \
  --run_id test_cotrain \
  --wandb_project test \
  --wandb_entity test
"

echo ""
echo ">>> TEST 2: VLA+VLM cotrain PASSED <<<"

echo ""
echo "========================================"
echo "  ALL TRAINING TESTS PASSED"
echo "========================================"
