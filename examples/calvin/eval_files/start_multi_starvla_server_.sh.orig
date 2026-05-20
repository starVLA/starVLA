#!/usr/bin/env bash
set -euo pipefail

BASE=/inspire/qb-ilm2/project/26summer-camp-10/26220056
REPO=$BASE/starVLA
export REPO

STORE=/inspire/qb-ilm2/project/26summer-camp-10/public/ten

TIMESTAMP=$(date +%Y%m%d%H%M)
LOG_BASE=$STORE/log/$TIMESTAMP
LOG_DIR=$LOG_BASE/starVLA

CKPT=/inspire/qb-ilm2/project/26summer-camp-10/26220056/starVLA/ten/qwen35_2b_cosmopredict2_gr00t_calvin_abc_multiview_20260519_112310/checkpoints/steps_10000_pytorch_model.pt
#/inspire/qb-ilm2/project/26summer-camp-10/public/ten/ckpt/v0519/qwen35_2b_gr00t_calvin_abc_multiview_job-b82d046c-d876-441d-bb86-e1b9271fc940_round0_20260519_073006/checkpoints/steps_10000_pytorch_model.pt

# 使用哪些 GPU。GPU0 保留给 Calvin EGL/PyBullet 渲染。
GPUS=(1 2 3 4 5 6 7)

# 起始端口：GPU1 -> 5695, GPU2 -> 5696, ...
BASE_PORT=5695

mkdir -p "$LOG_DIR"

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
source $BASE/.venvs/starVLA/bin/activate
cd \$REPO

export PYTHONPATH="$REPO:${PYTHONPATH:-}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export STARVLA_QWEN_ATTN_IMPL=\${STARVLA_QWEN_ATTN_IMPL:-sdpa}

CUDA_VISIBLE_DEVICES=$GPU python -u deployment/model_server/server_policy.py \
    --ckpt_path $CKPT \
    --port $PORT \
    --use_bf16
" > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    echo "[OK] GPU=$GPU PORT=$PORT PID=$PID"
    echo "     Log: $LOG_FILE"

    # # Wait for this server to be listening before launching the next one.
    # # Model load is memory and CUDA-context heavy; serial startup makes logs
    # # easier to interpret and avoids avoidable driver pressure.
    # if [ "$idx" -lt $(( ${#GPUS[@]} - 1 )) ]; then
    #     echo "     Waiting for server to be ready..."
    #     for i in $(seq 1 300); do
    #         if grep -q "server listening" "$LOG_FILE" 2>/dev/null; then
    #             echo "     Server ready after ${i}s"
    #             break
    #         fi
    #         if ! ps -p "$PID" > /dev/null 2>&1; then
    #             echo "     [ERROR] Server process died!"
    #             break
    #         fi
    #         sleep 1
    #     done
    # fi
done

echo
echo "查看所有 server："
echo "ps aux | grep server_policy.py | grep -v grep"
echo
echo "日志目录: $LOG_DIR"
echo "查看某个日志，例如："
echo "tail -f $LOG_DIR/starvla_server_gpu${GPUS[0]}_port${BASE_PORT}.log"
