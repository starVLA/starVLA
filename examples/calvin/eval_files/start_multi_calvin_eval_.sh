#!/usr/bin/env bash
# ===========================================================================
# Multi-GPU Calvin eval client launcher (7 workers, ports 5695..5701)
#   - Must be run AFTER start_multi_starvla_server_.sh and after each server
#     has finished loading the ckpt (look for "listening on port ..." in the
#     server log).
#   - Splits the 1000-sequence eval set into 7 chunks (or smaller via LIMIT_SEQUENCES)
#   - Each worker uses its own log dir, working dir, and dedicated port.
#   - Adapted from the 26220056 / public-ten template to 26220319 layout.
# ===========================================================================
set -euo pipefail

REPO=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA
export REPO

# === Same defaults as the server launcher ===
CKPT=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA/results/Checkpoints/qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview_d_style_ft_20260520_043105/checkpoints/steps_1000_pytorch_model.pt
DATASET_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
CALVIN_CONFIG_PATH=/inspire/qb-ilm2/project/26summer-camp-10/public/four/calvin/calvin_models/conf
SOURCE_EVAL_SEQUENCES=${REPO}/examples/calvin/eval_files/eval_sequences.json
CALVIN_PYTHON=/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/calvin_venv/bin/python

TIMESTAMP=$(date +%Y%m%d%H%M)
LOG_DIR=${REPO}/logs/multi/${TIMESTAMP}/calvin
RUN_DIR=${REPO}/logs/multi/${TIMESTAMP}/calvin_parallel
SPLIT_DIR=${RUN_DIR}/eval_splits
mkdir -p "$LOG_DIR" "$RUN_DIR" "$SPLIT_DIR"

HOST=127.0.0.1
BASE_PORT=5695
UNNORM_KEY=franka

# Calvin EGL settings. GPU0 is reserved for rendering; all workers share it.
# Set CALVIN_EGL_PER_WORKER=1 to render on each worker's own server GPU.
CALVIN_EGL_GPU=${CALVIN_EGL_GPU:-0}
CALVIN_EGL_PER_WORKER=${CALVIN_EGL_PER_WORKER:-0}
CALVIN_PROGRESS_EVERY=${CALVIN_PROGRESS_EVERY:-25}

GPUS=(1 2 3 4 5 6 7)
NUM_WORKERS=${#GPUS[@]}

# 调试时设小一点 (e.g. 100); 正式跑全量设 1000:
LIMIT_SEQUENCES=${LIMIT_SEQUENCES:-1000}

# Pre-flight
if [ ! -f "$CKPT" ]; then
  echo "[ERROR] CKPT does not exist: $CKPT" >&2; exit 1
fi
if [ ! -d "$DATASET_PATH/validation" ]; then
  echo "[ERROR] DATASET_PATH/validation missing: $DATASET_PATH/validation" >&2; exit 1
fi
if [ ! -d "$CALVIN_CONFIG_PATH" ]; then
  echo "[ERROR] CALVIN_CONFIG_PATH missing: $CALVIN_CONFIG_PATH" >&2; exit 1
fi
if [ ! -f "$SOURCE_EVAL_SEQUENCES" ]; then
  echo "[ERROR] SOURCE_EVAL_SEQUENCES missing: $SOURCE_EVAL_SEQUENCES" >&2; exit 1
fi

echo "[INFO] LIMIT_SEQUENCES=$LIMIT_SEQUENCES  NUM_WORKERS=$NUM_WORKERS"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] SPLIT_DIR=$SPLIT_DIR"
echo

echo "[INFO] Splitting eval sequences ..."
if [ -n "${LIMIT_SEQUENCES}" ]; then
    "$CALVIN_PYTHON" "${REPO}/scripts/split_calvin_sequences.py" \
        --input "$SOURCE_EVAL_SEQUENCES" \
        --out-dir "$SPLIT_DIR" \
        --num-workers "$NUM_WORKERS" \
        --limit "$LIMIT_SEQUENCES"
else
    "$CALVIN_PYTHON" "${REPO}/scripts/split_calvin_sequences.py" \
        --input "$SOURCE_EVAL_SEQUENCES" \
        --out-dir "$SPLIT_DIR" \
        --num-workers "$NUM_WORKERS"
fi

echo
echo "[INFO] Starting Calvin eval workers ..."

for idx in "${!GPUS[@]}"; do
    GPU="${GPUS[$idx]}"
    PORT=$((BASE_PORT + idx))

    EVAL_SEQ="$SPLIT_DIR/eval_sequences_worker_${idx}.json"
    COUNT_FILE="$SPLIT_DIR/eval_sequences_worker_${idx}.count"

    if [ ! -f "$EVAL_SEQ" ]; then
        echo "[WARN] Missing split file: $EVAL_SEQ, skip worker $idx"
        continue
    fi

    NUM_SEQUENCES=$(cat "$COUNT_FILE")
    if [ "$NUM_SEQUENCES" -le 0 ]; then
        echo "[SKIP] worker=$idx has 0 sequences"
        continue
    fi

    WORK_DIR="$RUN_DIR/worker_${idx}"
    EVAL_LOG_DIR="$LOG_DIR/worker_${idx}_results"
    mkdir -p "$WORK_DIR" "$EVAL_LOG_DIR"

    LOG_FILE="$LOG_DIR/calvin_eval_worker${idx}_gpu${GPU}_port${PORT}.log"
    PID_FILE="$LOG_DIR/calvin_eval_worker${idx}_gpu${GPU}_port${PORT}.pid"

    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "[SKIP] worker=$idx already running, PID=$OLD_PID"
            continue
        fi
    fi

    if [ "$CALVIN_EGL_PER_WORKER" = "1" ]; then
        EGL_GPU="$GPU"
    else
        EGL_GPU="$CALVIN_EGL_GPU"
    fi

    echo "[START] worker=$idx GPU=$GPU PORT=$PORT NUM_SEQUENCES=$NUM_SEQUENCES"

    setsid bash -lc "
cd $WORK_DIR

export PYTHONPATH=$REPO:\${PYTHONPATH:-}

# CALVIN / PyBullet EGL rendering. All workers share CALVIN_EGL_GPU by default.
unset DISPLAY
export PYOPENGL_PLATFORM=egl
export EGL_VISIBLE_DEVICES=$EGL_GPU
# This line was needed on the source machine to force the NVIDIA loader.
# On this host /usr/share/glvnd/egl_vendor.d/10_nvidia.json doesn't exist
# but EGL still picks NVIDIA via the default loader path, so we leave it off.
# export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

export CALVIN_PROGRESS_EVERY=$CALVIN_PROGRESS_EVERY

$CALVIN_PYTHON -u $REPO/examples/calvin/eval_files/eval_calvin.py \
    --args.pretrained-path $CKPT \
    --args.unnorm-key $UNNORM_KEY \
    --args.host $HOST \
    --args.port $PORT \
    --args.dataset_path $DATASET_PATH \
    --args.calvin_config_path $CALVIN_CONFIG_PATH \
    --args.eval_sequences_path $EVAL_SEQ \
    --args.num_sequences $NUM_SEQUENCES \
    --args.eval-log-dir $EVAL_LOG_DIR
" > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    echo "[OK] worker=$idx PID=$PID"
    echo "     Log: $LOG_FILE"
done

echo
echo "查看所有 eval：       ps aux | grep eval_calvin.py | grep -v grep"
echo "日志目录:             $LOG_DIR"
echo "看某个日志，例如:     tail -f $LOG_DIR/calvin_eval_worker0_gpu${GPUS[0]}_port${BASE_PORT}.log"
echo "全部杀掉:             for f in $LOG_DIR/*.pid; do kill \$(cat \$f) 2>/dev/null || true; done"
