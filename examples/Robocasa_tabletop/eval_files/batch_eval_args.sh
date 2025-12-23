#!/bin/bash

# ============================================================
# Argument Parsing
# ============================================================

# Paths to Python executables in conda environments
cd /mnt/petrelfs/yejinhui/Projects/starVLA

starVLA_PYTHON=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python
ROBOCASA_PYTHON=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/robocasa/bin/python
export PYTHONPATH=$(pwd):${PYTHONPATH}

CKPT_DEFAULT="/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/1221_fourier_gr1_unified_1000_QwenOFT_state_qwen3/checkpoints/steps_90000_pytorch_model.pt"
N_ENVS_DEFAULT=1
MAX_EPISODE_STEPS_DEFAULT=720
N_ACTION_STEPS_DEFAULT=12

# Parse command-line arguments
CKPT_PATH=${1:-$CKPT_DEFAULT}
N_ENVS=${2:-$N_ENVS_DEFAULT}
MAX_EPISODE_STEPS=${3:-$MAX_EPISODE_STEPS_DEFAULT}
N_ACTION_STEPS=${4:-$N_ACTION_STEPS_DEFAULT}


echo "=== Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "Number of Envs       : ${N_ENVS}"
echo "Max Episode Steps    : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length  : ${N_ACTION_STEPS}"
echo "================================"

# ============================================================
# Evaluation Function
# ============================================================

EvalEnv() {
    local GPU_ID=$1
    local PORT=$2
    local ENV_NAME=$3
    local CKPT_PATH=$4
    local LOG_DIR=$5
    local ROBOCASA_PYTHON=$6
    local N_ENVS=$7
    local MAX_EPISODE_STEPS=$8
    local N_ACTION_STEPS=$9
    # save root CKPT_PATH 的 parent 文件夹
    local SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local ckpt_name=$(basename "$CKPT_PATH" .pt)
    local VIDEO_OUT_PATH="${SAVE_ROOT}/videos/${ckpt_name}/n_action_steps_${N_ACTION_STEPS}_max_episode_steps_${MAX_EPISODE_STEPS}_n_envs_${N_ENVS}_${ENV_NAME}"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching evaluation | GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${ROBOCASA_PYTHON} examples/Robocasa_tabletop/eval_files/simulation_env.py \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 200 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        > "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log" 2>&1
}

# ============================================================
# Environment List
# ============================================================

ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

# ============================================================
# Runtime Configuration
# ============================================================

BASE_PORT=5478
NUM_GPUS=8

LOG_DIR="${CKPT_PATH}.log/eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching Multi-GPU Evaluation ==="
echo "GPUs            : ${NUM_GPUS}"
echo "Num Environments: ${#ENV_NAMES[@]}"
echo "Log Directory   : ${LOG_DIR}"

# ============================================================
# Step 1: Launch Policy Servers
# ============================================================

SERVER_PIDS=()

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${starVLA_PYTHON} deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${PORT}" \
        --use_bf16 \
        > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

    SERVER_PIDS[$GPU_ID]=$!

    sleep 10
done

sleep 30

# ============================================================
# Step 2: Dispatch Environments to GPUs
# ============================================================

COUNT=0
for ENV_NAME in "${ENV_NAMES[@]}"; do
    GPU_ID=$((COUNT % NUM_GPUS))
    PORT=$((BASE_PORT + GPU_ID))

    if (( (COUNT + 1) % NUM_GPUS == 0 )); then
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}"
    else
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" &
    fi

    COUNT=$((COUNT + 1))

    sleep 2
done

# ============================================================
# Step 3: Cleanup
# ============================================================

# 判断是否还有 examples/Robocasa_tabletop/eval_files/simulation_env.py
while pgrep -f "examples/Robocasa_tabletop/eval_files/simulation_env.py" > /dev/null; do
    echo "Waiting for all evaluation environments to finish..."
    sleep 30
done

echo ""
echo "Shutting down policy servers..."

for PID in "${SERVER_PIDS[@]}"; do
    kill "${PID}" 2>/dev/null && echo "Killed server PID ${PID}"
done

echo "=== Evaluation Finished ==="
