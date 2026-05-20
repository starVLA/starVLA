#!/usr/bin/env bash
# ===========================================================================
# Multi-GPU StarVLA policy server launcher (7 servers on GPU1..GPU7)
#   - GPU0 reserved for Calvin EGL/PyBullet rendering shared by all workers
#   - Each server listens on its own port: GPU1 -> 5695, GPU2 -> 5696, ...
#   - Uses absolute python paths (no `conda activate` / `source venv`)
#   - Adapted from the 26220056 / public-ten template to 26220319 layout
# ===========================================================================
set -euo pipefail

REPO=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA
export REPO

# Server-side python. starvla_backup has transformers 5.8.0 which is required
# for Qwen3.5 family ckpts (the default ckpt below uses it).
# If you switch to a non-Qwen3.5 ckpt (e.g. CosmoPredict2GR00T), you can use
# the user's training venv: ${REPO}/.venv/bin/python
STAR_VLA_PYTHON=/inspire/qb-ilm2/project/26summer-camp-10/public/four/miniconda3/envs/starvla_backup/bin/python

# Default ckpt: Qwen3.5-2B GR00T multiview, 40k steps (known-good baseline).
CKPT=/inspire/qb-ilm2/project/26summer-camp-10/26220319/starVLA/results/Checkpoints/qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview_d_style_ft_20260520_043105/checkpoints/steps_1000_pytorch_model.pt

TIMESTAMP=$(date +%Y%m%d%H%M)
LOG_DIR=${REPO}/logs/multi/${TIMESTAMP}/starvla
mkdir -p "$LOG_DIR"

# 使用哪些 GPU。GPU0 留给 Calvin EGL。
GPUS=(1 2 3 4 5 6 7)
BASE_PORT=5695

if [ ! -f "$CKPT" ]; then
  echo "[ERROR] CKPT does not exist: $CKPT" >&2
  exit 1
fi
if [ ! -x "$STAR_VLA_PYTHON" ]; then
  echo "[ERROR] STAR_VLA_PYTHON not executable: $STAR_VLA_PYTHON" >&2
  exit 1
fi

echo "[INFO] REPO=$REPO"
echo "[INFO] PYTHON=$STAR_VLA_PYTHON"
echo "[INFO] CKPT=$CKPT"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] GPUS=${GPUS[*]}  base_port=$BASE_PORT"
echo

for idx in "${!GPUS[@]}"; do
    GPU="${GPUS[$idx]}"
    PORT=$((BASE_PORT + idx))

    LOG_FILE="$LOG_DIR/starvla_server_gpu${GPU}_port${PORT}.log"
    PID_FILE="$LOG_DIR/starvla_server_gpu${GPU}_port${PORT}.pid"

    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "[SKIP] GPU=$GPU PORT=$PORT already running, PID=$OLD_PID"
            echo "       Log: $LOG_FILE"
            continue
        fi
    fi

    echo "[START] GPU=$GPU PORT=$PORT"

    nohup bash -lc "
cd $REPO

export PYTHONPATH=$REPO:\${PYTHONPATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export STARVLA_QWEN_ATTN_IMPL=\${STARVLA_QWEN_ATTN_IMPL:-sdpa}

CUDA_VISIBLE_DEVICES=$GPU $STAR_VLA_PYTHON -u deployment/model_server/server_policy.py \
    --ckpt_path $CKPT \
    --port $PORT \
    --use_bf16
" > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    echo "[OK] GPU=$GPU PORT=$PORT PID=$PID"
    echo "     Log: $LOG_FILE"
done

echo
echo "查看所有 server：     ps aux | grep server_policy.py | grep -v grep"
echo "日志目录:             $LOG_DIR"
echo "看某个日志，例如:     tail -f $LOG_DIR/starvla_server_gpu${GPUS[0]}_port${BASE_PORT}.log"
echo "全部杀掉:             for f in $LOG_DIR/*.pid; do kill \$(cat \$f) 2>/dev/null || true; done"
