#!/bin/bash
# run_parallel_eval.sh
#
# Automatically selects 4 free GPUs, launches 4 policy servers,
# splits 11 TASK_SUITES into 4 groups (3-3-3-2), and runs evaluations in parallel.
#
# Usage:
#   bash examples/VLA-Arena/eval_files/run_parallel_eval.sh [OPTIONS]
#
# Can be run from anywhere; paths are resolved from the script's location.

set -euo pipefail

# Resolve the directory this script lives in (eval_files/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# starVLA root is three levels up: eval_files -> VLA-Arena -> examples -> starVLA
STARVLA_HOME="${STARVLA_HOME:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

###########################################################################################
# === Configuration ===
your_ckpt=""
VLA_ARENA_ENV=""          # path to VLA-Arena uv project, e.g. /path/to/VLA-Arena/env/
NUM_SERVERS=4
BASE_PORT=10090           # ports will be BASE_PORT + gpu_id
GPU_MEM_THRESHOLD=2000    # GPUs with memory usage below this (MiB) are considered free
SERVER_STARTUP_WAIT=180   # seconds to wait for each server to start

# All 11 task suites
ALL_SUITES=(
    "safety_static_obstacles"
    "safety_cautious_grasp"
    "safety_hazard_avoidance"
    "safety_state_preservation"
    "safety_dynamic_obstacles"
    "distractor_static_distractors"
    "distractor_dynamic_distractors"
    "extrapolation_preposition_combinations"
    "extrapolation_task_workflows"
    "extrapolation_unseen_objects"
    "long_horizon"
)

# Split into 4 groups: 3-3-3-2
GROUP_0=("safety_static_obstacles" "safety_cautious_grasp" "safety_hazard_avoidance")
GROUP_1=("safety_state_preservation" "safety_dynamic_obstacles" "distractor_static_distractors")
GROUP_2=("distractor_dynamic_distractors" "extrapolation_preposition_combinations" "extrapolation_task_workflows")
GROUP_3=("extrapolation_unseen_objects" "long_horizon")
###########################################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Parallel evaluation: launches policy servers on free GPUs and runs VLA-Arena
task suites in parallel.

OPTIONS:
    -c, --checkpoint PATH       Path to pretrained checkpoint (required)
    --vla-arena-env PATH        Path to VLA-Arena uv project dir (required)
                                e.g. /path/to/VLA-Arena/envs/openpi
    --starvla-home PATH         starVLA root directory (default: auto-detected)
    --num-servers NUM            Number of parallel servers/groups (default: $NUM_SERVERS)
    --base-port PORT             Base port number (default: $BASE_PORT)
    --gpu-mem-threshold MiB      Free GPU memory threshold (default: $GPU_MEM_THRESHOLD)
    --server-wait SECONDS        Server startup timeout (default: $SERVER_STARTUP_WAIT)
    -h, --help                   Show this help message

ENVIRONMENT VARIABLES:
    STARVLA_HOME                 starVLA root (overridden by --starvla-home)
    VLA_ARENA_ENV                VLA-Arena uv project dir (overridden by --vla-arena-env)
    starVLA_python               Python interpreter (default: python)

EXAMPLES:
    $0 -c /path/to/ckpt.pt --vla-arena-env /path/to/VLA-Arena/envs/openpi
    STARVLA_HOME=/opt/starVLA $0 -c ckpt.pt --vla-arena-env /opt/VLA-Arena/envs/openpi
EOF
}

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)         your_ckpt="$2"; shift 2 ;;
        --vla-arena-env)         VLA_ARENA_ENV="$2"; shift 2 ;;
        --starvla-home)          STARVLA_HOME="$2"; shift 2 ;;
        --num-servers)           NUM_SERVERS="$2"; shift 2 ;;
        --base-port)             BASE_PORT="$2"; shift 2 ;;
        --gpu-mem-threshold)     GPU_MEM_THRESHOLD="$2"; shift 2 ;;
        --server-wait)           SERVER_STARTUP_WAIT="$2"; shift 2 ;;
        -h|--help)               show_usage; exit 0 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

# Also accept VLA_ARENA_ENV from environment variable if not set via flag
VLA_ARENA_ENV="${VLA_ARENA_ENV:-${VLA_ARENA_ENV_DEFAULT:-}}"

# Validate required arguments
if [[ -z "$your_ckpt" ]]; then
    print_error "Checkpoint path is required (-c / --checkpoint)"
    show_usage
    exit 1
fi
if [[ -z "$VLA_ARENA_ENV" ]]; then
    print_error "VLA-Arena env path is required (--vla-arena-env or VLA_ARENA_ENV env var)"
    show_usage
    exit 1
fi
if [[ ! -d "$VLA_ARENA_ENV" ]]; then
    print_error "VLA-Arena env directory does not exist: $VLA_ARENA_ENV"
    exit 1
fi
if [[ ! -d "$STARVLA_HOME" ]]; then
    print_error "starVLA home directory does not exist: $STARVLA_HOME"
    exit 1
fi

LOG_DIR="log"
mkdir -p "$LOG_DIR"

print_info "starVLA home  : $STARVLA_HOME"
print_info "VLA-Arena env : $VLA_ARENA_ENV"
print_info "Checkpoint    : $your_ckpt"
print_info "Log directory : $LOG_DIR"

# ---- Step 1: Find free GPUs ----
print_info "Detecting free GPUs (memory usage < ${GPU_MEM_THRESHOLD} MiB)..."

FREE_GPUS=()
while IFS=, read -r idx mem_used; do
    idx=$(echo "$idx" | xargs)
    mem_used=$(echo "$mem_used" | xargs | sed 's/ MiB//')
    if (( mem_used < GPU_MEM_THRESHOLD )); then
        FREE_GPUS+=("$idx")
    fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader)

if (( ${#FREE_GPUS[@]} < NUM_SERVERS )); then
    print_error "Need ${NUM_SERVERS} free GPUs but only found ${#FREE_GPUS[@]}: ${FREE_GPUS[*]}"
    print_info "All GPU memory usage:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
    exit 1
fi

# Take first N free GPUs
SELECTED_GPUS=("${FREE_GPUS[@]:0:$NUM_SERVERS}")
print_success "Selected GPUs: ${SELECTED_GPUS[*]}"

# ---- Step 2: Launch policy servers ----
export PYTHONPATH="${STARVLA_HOME}:${PYTHONPATH:-}"
starVLA_python="${starVLA_python:-python}"

SERVER_PIDS=()
EVAL_PIDS=()
EVAL_LOGS=()
PORTS=()
CLEANED_UP=0

cleanup() {
    if (( CLEANED_UP == 1 )); then
        return
    fi
    CLEANED_UP=1

    print_warning "Cleaning up server processes..."
    for pid in "${EVAL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            print_info "Killed eval/client PID $pid"
        fi
    done

    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            print_info "Killed server PID $pid"
        fi
    done

    # Escalate if some processes ignore SIGTERM.
    sleep 1
    for pid in "${EVAL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            print_info "Force killed eval/client PID $pid"
        fi
    done
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            print_info "Force killed server PID $pid"
        fi
    done

    wait 2>/dev/null || true
}

on_interrupt() {
    print_warning "Received interrupt signal, terminating all spawned server/client processes..."
    cleanup
    exit 130
}

trap cleanup EXIT
trap on_interrupt INT TERM

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    gpu_id=${SELECTED_GPUS[$i]}
    port=$((BASE_PORT + gpu_id))
    PORTS+=("$port")

    print_info "Launching server $i on GPU ${gpu_id}, port ${port}..."
    CUDA_VISIBLE_DEVICES=${gpu_id} ${starVLA_python} "${STARVLA_HOME}/deployment/model_server/server_policy.py" \
        --ckpt_path "${your_ckpt}" \
        --port "${port}" \
        --use_bf16 \
        > "${LOG_DIR}/server_gpu${gpu_id}.log" 2>&1 &
    SERVER_PIDS+=($!)
done

# ---- Step 3: Wait for servers to be ready ----
print_info "Waiting for servers to start (up to ${SERVER_STARTUP_WAIT}s)..."

wait_for_server() {
    local pid=$1
    local port=$2
    local log_file=$3
    local timeout=$4
    local elapsed=0

    while (( elapsed < timeout )); do
        # If process exited early, fail fast and surface log hints.
        if ! kill -0 "$pid" 2>/dev/null; then
            print_error "Server process PID ${pid} exited before ready."
            if [[ -f "$log_file" ]]; then
                print_info "Last 30 lines of ${log_file}:"
                tail -n 30 "$log_file" || true
            fi
            return 1
        fi

        # Primary readiness signal: server log confirms websocket listener is up.
        if [[ -f "$log_file" ]] && grep -Eq "server listening on .*:${port}" "$log_file"; then
            return 0
        fi

        # Fallback: check TCP listening state without opening websocket handshake.
        if command -v ss >/dev/null 2>&1; then
            if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(:|\])${port}$"; then
                return 0
            fi
        fi

        sleep 5
        elapsed=$((elapsed + 5))
    done
    return 1
}

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    pid=${SERVER_PIDS[$i]}
    port=${PORTS[$i]}
    gpu_id=${SELECTED_GPUS[$i]}
    log_file="${LOG_DIR}/server_gpu${gpu_id}.log"
    if wait_for_server "$pid" "$port" "$log_file" "$SERVER_STARTUP_WAIT"; then
        print_success "Server $i (GPU ${gpu_id}, port ${port}) is ready"
    else
        print_error "Server $i (GPU ${gpu_id}, port ${port}) failed to start. Check ${LOG_DIR}/server_gpu${gpu_id}.log"
        exit 1
    fi
done

print_success "All ${NUM_SERVERS} servers are running!"

# ---- Step 4: Launch eval processes in parallel ----
# Each eval process handles its group of task suites against its assigned server.

for i in $(seq 0 $((NUM_SERVERS - 1))); do
    port=${PORTS[$i]}
    gpu_id=${SELECTED_GPUS[$i]}

    # Build the suites string for this group
    eval "suites_array=(\"\${GROUP_${i}[@]}\")"
    suites_str="${suites_array[*]}"

    eval_log="${LOG_DIR}/eval_group${i}_gpu${gpu_id}.log"
    EVAL_LOGS+=("$eval_log")

    print_info "Starting eval group $i on port ${port}: ${suites_str}"
    uv run --project "${VLA_ARENA_ENV}" \
        bash "${SCRIPT_DIR}/eval_vla_arena.sh" \
        --checkpoint "${your_ckpt}" \
        --port "${port}" \
        --suites "${suites_str}" \
        > "${eval_log}" 2>&1 &
    EVAL_PIDS+=($!)
done

# ---- Step 5: Wait for all evaluations to finish ----
print_info "All ${NUM_SERVERS} eval processes launched. Waiting for completion..."

EVAL_FAILURES=0
for i in $(seq 0 $((NUM_SERVERS - 1))); do
    pid=${EVAL_PIDS[$i]}
    eval_log=${EVAL_LOGS[$i]}
    if wait "$pid"; then
        print_success "Eval group $i completed successfully. Log: ${eval_log}"
    else
        print_error "Eval group $i failed. Check ${eval_log}"
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    fi
done

# ---- Summary ----
echo ""
print_info "===== Parallel Evaluation Complete ====="
print_info "Servers used: GPUs ${SELECTED_GPUS[*]}, Ports ${PORTS[*]}"
print_info "Eval failures: ${EVAL_FAILURES} / ${NUM_SERVERS}"

if (( EVAL_FAILURES == 0 )); then
    print_success "All evaluations completed successfully!"
else
    print_warning "${EVAL_FAILURES} evaluation group(s) had failures. Check logs."
fi

print_info "Server logs: ${LOG_DIR}/server_gpu*.log"
print_info "Eval logs: ${LOG_DIR}/eval_group*.log"
