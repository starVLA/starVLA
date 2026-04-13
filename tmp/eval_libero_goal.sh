#!/bin/bash
# ============================================================
# LIBERO Goal Evaluation Script (adapted for vonneumann cluster)
# This script runs BOTH the model server and eval client on the
# SAME compute node, using srun to get GPU access.
# ============================================================
set -e

# === Paths (adapted for this cluster) ===
STARVLA_DIR=/home/jye624/Projcets/starVLA
LIBERO_HOME=/home/jye624/Projcets/LIBERO
STARVLA_PYTHON=/home/jye624/.conda/envs/starVLA/bin/python
LIBERO_PYTHON=/home/jye624/.conda/envs/libero/bin/python

# === Checkpoint ===
CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt

# === Eval Config ===
HOST="127.0.0.1"
PORT=5694
TASK_SUITE="libero_goal"
NUM_TRIALS=50
GPU_ID=0

# === Derived Paths ===
FOLDER_NAME=$(echo "$CKPT" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
VIDEO_OUT="${STARVLA_DIR}/results/${TASK_SUITE}/${FOLDER_NAME}"
LOG_DIR="${STARVLA_DIR}/logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p "${LOG_DIR}" "${VIDEO_OUT}"

echo "============================================"
echo " LIBERO Evaluation: ${TASK_SUITE}"
echo " Checkpoint: ${CKPT}"
echo " GPU: ${GPU_ID}, Port: ${PORT}"
echo " Video output: ${VIDEO_OUT}"
echo " Log dir: ${LOG_DIR}"
echo "============================================"

# === Step 1: Start Model Server (starVLA env, background) ===
echo "[$(date +%H:%M:%S)] Starting model server on GPU ${GPU_ID}..."
cd ${STARVLA_DIR}
export PYTHONPATH=${STARVLA_DIR}:${PYTHONPATH}

CUDA_VISIBLE_DEVICES=${GPU_ID} ${STARVLA_PYTHON} deployment/model_server/server_policy.py \
    --ckpt_path ${CKPT} \
    --port ${PORT} \
    --use_bf16 \
    --idle_timeout -1 \
    > ${LOG_DIR}/server.log 2>&1 &
SERVER_PID=$!
echo "[$(date +%H:%M:%S)] Server PID: ${SERVER_PID}"

# Wait for server to be ready (check if port is open)
echo "[$(date +%H:%M:%S)] Waiting for server to load model (this takes ~60-120s for 9.8GB checkpoint)..."
SERVER_TIMEOUT=300
for i in $(seq 1 ${SERVER_TIMEOUT}); do
    if ss -tln | grep -q ":${PORT} "; then
        echo "[$(date +%H:%M:%S)] Server ready after ${i}s!"
        break
    fi
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] ERROR: Server process died! Check ${LOG_DIR}/server.log"
        cat ${LOG_DIR}/server.log | tail -30
        exit 1
    fi
    if [ $((i % 30)) -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Still waiting... (${i}s elapsed)"
        tail -1 ${LOG_DIR}/server.log 2>/dev/null
    fi
    sleep 1
done

if ! ss -tln | grep -q ":${PORT} "; then
    echo "[$(date +%H:%M:%S)] ERROR: Server failed to start within ${SERVER_TIMEOUT}s"
    kill ${SERVER_PID} 2>/dev/null
    cat ${LOG_DIR}/server.log | tail -30
    exit 1
fi

# === Step 2: Run LIBERO Evaluation (LIBERO env) ===
echo "[$(date +%H:%M:%S)] Starting LIBERO evaluation..."
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export PYTHONPATH=${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
${LIBERO_PYTHON} ${STARVLA_DIR}/examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${CKPT} \
    --args.host "${HOST}" \
    --args.port ${PORT} \
    --args.task-suite-name "${TASK_SUITE}" \
    --args.num-trials-per-task ${NUM_TRIALS} \
    --args.video-out-path "${VIDEO_OUT}" \
    2>&1 | tee ${LOG_DIR}/eval.log

EVAL_EXIT=$?

# === Step 3: Cleanup ===
echo "[$(date +%H:%M:%S)] Evaluation finished (exit code: ${EVAL_EXIT}). Stopping server..."
kill ${SERVER_PID} 2>/dev/null
wait ${SERVER_PID} 2>/dev/null

echo "[$(date +%H:%M:%S)] Done! Results in: ${VIDEO_OUT}"
echo "[$(date +%H:%M:%S)] Logs in: ${LOG_DIR}"
