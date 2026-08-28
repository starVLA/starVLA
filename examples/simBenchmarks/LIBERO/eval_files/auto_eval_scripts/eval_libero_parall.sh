#!/usr/bin/env bash
set -euo pipefail

# One-shot LIBERO evaluation:
#   1. start policy server in the background
#   2. wait until the websocket port is listening
#   3. run LIBERO evaluation
#   4. always stop the policy server

STARVLA_DIR=${STARVLA_DIR:-/root/feihong/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/root/feihong/LIBERO}
starVLA_python=${starVLA_python:-/root/feihong/starVLA/.venv/bin/python}
LIBERO_python=${LIBERO_python:-/root/feihong/LIBERO/.venv/bin/python}

DEFAULT_CKPT=/root/nas/feihong/starVLA/Checkpoints/qwen_var_productvq_g16_s1248_structemb_nextscale_100k_fullcache/checkpoints/steps_100000_pytorch_model.pt

your_ckpt=${1:-${CKPT:-${DEFAULT_CKPT}}}
task_suite_name=${2:-${TASK_SUITE_NAME:-libero_goal}}
gpu_id=${3:-${GPU_ID:-0}}
base_port=${4:-${PORT:-6694}}

num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
max_tasks=${MAX_TASKS:--1}
host=${HOST:-127.0.0.1}
unnorm_key=${UNNORM_KEY:-franka}
server_wait_timeout=${SERVER_WAIT_TIMEOUT:-900}
image_history=${IMAGE_HISTORY:--1}
multiview_pack=${MULTIVIEW_PACK:-auto}

cd "${STARVLA_DIR}"

export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# Policy/eval workers are standalone processes. Cluster launchers may export
# distributed training variables that make accelerate initialize torch.distributed.
STANDALONE_ENV_CLEANUP=(
    -u WORLD_SIZE
    -u LOCAL_WORLD_SIZE
    -u RANK
    -u LOCAL_RANK
    -u NODE_RANK
    -u GROUP_RANK
    -u ROLE_RANK
    -u ROLE_WORLD_SIZE
    -u MASTER_ADDR
    -u MASTER_PORT
)

if [[ ! -f "${your_ckpt}" ]]; then
    echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
    exit 1
fi
if [[ ! -x "${starVLA_python}" ]]; then
    echo "[ERROR] starVLA python not executable: ${starVLA_python}" >&2
    exit 1
fi
if [[ ! -x "${LIBERO_python}" ]]; then
    echo "[ERROR] LIBERO python not executable: ${LIBERO_python}" >&2
    exit 1
fi

if [[ "${your_ckpt}" == *"/checkpoints/"* ]]; then
    model_root=$(echo "${your_ckpt}" | awk -F'/checkpoints/' '{print $1}')
elif [[ "$(basename "$(dirname "${your_ckpt}")")" == "final_model" ]]; then
    model_root=$(dirname "$(dirname "${your_ckpt}")")
else
    model_root=$(dirname "${your_ckpt}")
fi
folder_name=$(echo "${your_ckpt}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')

video_out_path="${model_root}/videos/${task_suite_name}/${folder_name}"
log_path="${model_root}/logs/${task_suite_name}"
mkdir -p "${video_out_path}" "${log_path}"

server_log="${log_path}/${folder_name}_server_gpu${gpu_id}_port${base_port}.log"
eval_log="${log_path}/${folder_name}.log"

server_pid=""
cleanup() {
    local status=$?
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "Stopping policy server PID ${server_pid}"
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "=========================================="
echo " LIBERO single evaluation"
echo "=========================================="
echo " checkpoint     : ${your_ckpt}"
echo " task suite     : ${task_suite_name}"
echo " GPU            : ${gpu_id}"
echo " port           : ${base_port}"
echo " trials/task    : ${num_trials_per_task}"
echo " max tasks      : ${max_tasks}"
echo " image history  : ${image_history}"
echo " multiview pack : ${multiview_pack}"
echo " server log     : ${server_log}"
echo " eval log       : ${eval_log}"
echo " video out path : ${video_out_path}"
echo "=========================================="

env "${STANDALONE_ENV_CLEANUP[@]}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${starVLA_python}" deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${base_port}" \
    --use_bf16 \
    > "${server_log}" 2>&1 &
server_pid=$!

echo "Started policy server PID ${server_pid}; waiting for ${host}:${base_port} ..."
wait_start=$(date +%s)
while ! ss -tlnp 2>/dev/null | grep -q ":${base_port} "; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "[ERROR] Policy server exited before port ${base_port} became ready." >&2
        tail -120 "${server_log}" >&2 || true
        exit 1
    fi
    if (( $(date +%s) - wait_start > server_wait_timeout )); then
        echo "[ERROR] Timed out waiting for policy server on port ${base_port} after ${server_wait_timeout}s." >&2
        tail -120 "${server_log}" >&2 || true
        exit 1
    fi
    sleep 5
done
echo "Policy server ready on port ${base_port}."

env "${STANDALONE_ENV_CLEANUP[@]}" \
    "${LIBERO_python}" ./examples/simBenchmarks/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path "${your_ckpt}" \
    --args.host "${host}" \
    --args.port "${base_port}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.max-tasks "${max_tasks}" \
    --args.unnorm-key "${unnorm_key}" \
    --args.video-out-path "${video_out_path}" \
    --args.image-history "${image_history}" \
    --args.multiview-pack "${multiview_pack}" \
    2>&1 | tee "${eval_log}"

echo "Evaluation completed. Videos saved to ${video_out_path}, log saved to ${eval_log}"
