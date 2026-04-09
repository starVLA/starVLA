#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOTWIN_ALL_TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
    dump_bin_bigbin
    grab_roller
    handover_block
    handover_mic
    hanging_mug
    lift_pot
    move_can_pot
    move_pillbottle_pad
    move_playingcard_away
    move_stapler_pad
    open_laptop
    open_microwave
    pick_diverse_bottles
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_bread_basket
    place_bread_skillet
    place_burger_fries
    place_can_basket
    place_cans_plasticbox
    place_container_plate
    place_dual_shoes
    place_empty_cup
    place_fan
    place_mouse_pad
    place_object_basket
    place_object_scale
    place_object_stand
    place_phone_stand
    place_shoe
    press_stapler
    put_bottles_dustbin
    put_object_cabinet
    rotate_qrcode
    scan_object
    shake_bottle_horizontally
    shake_bottle
    stack_blocks_three
    stack_blocks_two
    stack_bowls_three
    stack_bowls_two
    stamp_seal
    turn_switch
)

used_ports=()
SLOT_GPUS=()
SLOT_PORTS=()
ACTIVE_PIDS=()
ACTIVE_TASKS=()
ACTIVE_SERVER_LOGS=()
ACTIVE_EVAL_LOGS=()
FAILED_TASKS=()

join_arr() {
    local sep="$1"; shift
    local out=""
    for item in "$@"; do
        out="${out:+${out}${sep}}${item}"
    done
    printf '%s' "${out}"
}

kill_descendants() {
    local target_pid="$1"
    local sig="${2:-TERM}"
    local child_pids
    child_pids="$(ps -o pid= --ppid "${target_pid}" 2>/dev/null)" || true
    local cpid
    for cpid in ${child_pids}; do
        kill_descendants "${cpid}" "${sig}"
    done
    kill -"${sig}" "${target_pid}" 2>/dev/null || true
}

cleanup_active_jobs() {
    trap '' INT TERM EXIT
    echo "[INFO] Cleaning up all child processes..."
    local pid
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill_descendants "${pid}" TERM
        fi
    done
    sleep 2
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill_descendants "${pid}" KILL
        fi
    done
    for pid in "${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup_active_jobs EXIT INT TERM

usage() {
    cat >&2 <<'EOF'
Usage:
  bash start_eval.sh -m <mode> -n <policy_name> -c <ckpt_path> [options] <tasks...>

Required flags:
  -m, --mode              Eval mode: demo_clean or demo_randomized
  -n, --name              Policy name (used for log directory naming)
  -c, --ckpt              Path to the checkpoint file

Tasks (positional):
  Remaining arguments after flags are treated as task names, the keyword
  `all` (all RoboTwin 2.0 tasks), or a single task-list file (one per line).

Optional flags:
  -s, --seed              Eval seed (default: 0, env: ROBOTWIN_SEED)
  -j, --jobs-per-gpu      Concurrent jobs per GPU (default: 1, env: ROBOTWIN_JOBS_PER_GPU)
  -p, --base-port         First port to allocate (default: 5694, env: ROBOTWIN_BASE_PORT)
      --server-timeout    Seconds to wait for server (default: 600, env: ROBOTWIN_SERVER_TIMEOUT)
      --install-deps      Run pip install bootstrap steps once
  -h, --help              Show this help message

Examples:
  bash start_eval.sh -m demo_clean -n test1 -c /path/to/ckpt.pt adjust_bottle
  bash start_eval.sh -m demo_randomized -n test1 -c /path/to/ckpt.pt all
  bash start_eval.sh --mode demo_clean --name my_run --ckpt /path/to/ckpt.pt task_list.txt
  bash start_eval.sh -m demo_clean -n test1 -c /path/to/ckpt.pt -j 2 adjust_bottle open_laptop

Environment variables (lower priority than flags):
  ROBOTWIN_PATH              Path to the RoboTwin repository
  ROBOTWIN_STARVLA_ENV       Conda env name for the policy server (default: starvla)
  ROBOTWIN_ENV               Conda env name for RoboTwin eval (default: robotwin)
  STARVLA_PYTHON             Explicit python path for starvla (skips conda env lookup)
  ROBOTWIN_PYTHON            Explicit python path for robotwin (skips conda env lookup)
EOF
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s\n' "${value}"
}

port_in_use() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -tuln 2>/dev/null | grep -q ":${port}[[:space:]]"
    elif command -v lsof >/dev/null 2>&1; then
        lsof -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v netstat >/dev/null 2>&1; then
        netstat -tuln 2>/dev/null | grep -q ":${port}[[:space:]]"
    else
        "${STARVLA_PYTHON:-python}" -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',${port})); s.close()" 2>/dev/null
    fi
}

port_reserved() {
    local port="$1"
    local reserved_port
    for reserved_port in "${used_ports[@]+"${used_ports[@]}"}"; do
        if [[ "${reserved_port}" == "${port}" ]]; then
            return 0
        fi
    done
    return 1
}

find_available_port() {
    local port="$1"
    while port_reserved "${port}" || port_in_use "${port}"; do
        port=$((port + 1))
    done
    used_ports+=("${port}")
    printf '%s\n' "${port}"
}

check_port_detection() {
    if command -v ss >/dev/null 2>&1 \
        || command -v lsof >/dev/null 2>&1 \
        || command -v netstat >/dev/null 2>&1; then
        return 0
    fi
    if "${STARVLA_PYTHON:-python}" -c "import socket" 2>/dev/null; then
        echo "[INFO] No ss/lsof/netstat found, using Python for port detection."
        return 0
    fi
    echo "[ERROR] No port detection method available (need ss, lsof, netstat, or Python)." >&2
    return 1
}

wait_for_server() {
    local port="$1"
    local timeout_s="${2:-600}"
    local elapsed=0
    while (( elapsed < timeout_s )); do
        if port_in_use "${port}"; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

detect_cuda_devices() {
    local -a devices=()
    local -a cleaned=()
    local gpu_count=""
    local idx=0
    local device=""

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra devices <<< "${CUDA_VISIBLE_DEVICES}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
        if [[ -n "${gpu_count}" && "${gpu_count}" != "0" ]]; then
            for (( idx = 0; idx < gpu_count; ++idx )); do
                devices+=("${idx}")
            done
        fi
    fi

    for device in "${devices[@]+"${devices[@]}"}"; do
        device="$(trim "${device}")"
        if [[ -n "${device}" ]]; then
            cleaned+=("${device}")
        fi
    done

    if (( ${#cleaned[@]} == 0 )); then
        cleaned=(0)
    fi

    printf '%s\n' "${cleaned[@]}"
}

resolve_tasks() {
    local -a raw_inputs=("$@")
    local -a resolved=()
    local -a split_inputs=()
    local input=""
    local task=""
    local line=""

    if (( ${#raw_inputs[@]} == 1 )) && [[ -f "${raw_inputs[0]}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="$(trim "${line%%#*}")"
            if [[ -n "${line}" ]]; then
                resolved+=("${line}")
            fi
        done < "${raw_inputs[0]}"
    else
        for input in "${raw_inputs[@]}"; do
            if [[ "${input}" == "all" ]]; then
                resolved+=("${ROBOTWIN_ALL_TASKS[@]}")
                continue
            fi
            IFS=',' read -ra split_inputs <<< "${input}"
            for task in "${split_inputs[@]}"; do
                task="$(trim "${task}")"
                if [[ -n "${task}" ]]; then
                    resolved+=("${task}")
                fi
            done
        done
    fi

    if (( ${#resolved[@]} == 0 )); then
        echo "No RoboTwin tasks were resolved from input." >&2
        return 1
    fi

    printf '%s\n' "${resolved[@]}"
}

find_conda_python() {
    local env_name="$1"
    local -a search_dirs=()

    if [[ -n "${CONDA_EXE:-}" ]]; then
        search_dirs+=("$(dirname "$(dirname "${CONDA_EXE}")")/envs")
    fi
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        search_dirs+=("$(dirname "${CONDA_PREFIX}")")
    fi
    search_dirs+=(
        "${HOME}/miniconda3/envs"
        "${HOME}/anaconda3/envs"
        "${HOME}/miniforge3/envs"
        "${HOME}/mambaforge/envs"
        "/opt/conda/envs"
    )

    local base
    for base in "${search_dirs[@]}"; do
        if [[ -x "${base}/${env_name}/bin/python" ]]; then
            printf '%s\n' "${base}/${env_name}/bin/python"
            return 0
        fi
    done

    echo "[ERROR] Cannot find Python for conda env '${env_name}'." >&2
    echo "  Searched: ${search_dirs[*]}" >&2
    echo "  Set STARVLA_PYTHON / ROBOTWIN_PYTHON to the full python path instead." >&2
    return 1
}

resolve_python() {
    local explicit_path="$1"
    local env_name="$2"

    if [[ -n "${explicit_path}" ]]; then
        if [[ ! -x "${explicit_path}" ]]; then
            echo "[ERROR] Specified python not executable: ${explicit_path}" >&2
            return 1
        fi
        printf '%s\n' "${explicit_path}"
        return 0
    fi

    find_conda_python "${env_name}"
}

prepare_runtime_dependencies() {
    if [[ "${ROBOTWIN_AUTO_INSTALL_DEPS:-0}" != "1" ]]; then
        return 0
    fi

    echo "[INFO] Installing runtime dependencies..."
    "${STARVLA_PYTHON}" -m pip install snntorch
    "${ROBOTWIN_PYTHON}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
}

launch_task_in_slot() {
    local slot_idx="$1"
    local task_name="$2"
    local gpu_id="${SLOT_GPUS[$slot_idx]}"
    local port="${SLOT_PORTS[$slot_idx]}"
    local launched_pid=""
    local task_safe="${task_name//\//_}"
    local slot_label="slot${slot_idx}_gpu${gpu_id}_port${port}"
    local server_log="${LOG_DIR}/${task_safe}_${TASK_CONFIG}_${slot_label}_server.log"
    local eval_log="${LOG_DIR}/${task_safe}_${TASK_CONFIG}_${slot_label}_eval.log"

    echo "[INFO] Launching task=${task_name} mode=${TASK_CONFIG} gpu=${gpu_id} port=${port}"

    (
        set -euo pipefail

        server_pid=""
        cleanup_server() {
            if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
                kill "${server_pid}" 2>/dev/null || true
                wait "${server_pid}" 2>/dev/null || true
            fi
        }
        trap cleanup_server EXIT INT TERM

        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
        export STARVLA_PYTHON="${STARVLA_PYTHON}"
        export ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}"

        bash "${SCRIPT_DIR}/run_policy_server.sh" "${CKPT_PATH}" "${gpu_id}" "${port}" > "${server_log}" 2>&1 &
        server_pid=$!

        if ! wait_for_server "${port}" "${ROBOTWIN_SERVER_TIMEOUT:-600}"; then
            echo "[ERROR] Policy server failed to become ready for task=${task_name} on port=${port}. See ${server_log}" >&2
            exit 1
        fi

        cd "${SCRIPT_DIR}"
        bash "${SCRIPT_DIR}/eval.sh" \
            "${task_name}" \
            "${TASK_CONFIG}" \
            "${POLICY_NAME}" \
            "${ROBOTWIN_SEED:-0}" \
            "${gpu_id}" \
            "${CKPT_PATH}" \
            "${port}" \
            > >(tee "${eval_log}" | grep --line-buffered "Success rate" | sed -u "s/^/[RESULT] ${task_name}: /") 2>&1
    ) &

    launched_pid=$!
    ACTIVE_PIDS[$slot_idx]="${launched_pid}"
    ACTIVE_TASKS[$slot_idx]="${task_name}"
    ACTIVE_SERVER_LOGS[$slot_idx]="${server_log}"
    ACTIVE_EVAL_LOGS[$slot_idx]="${eval_log}"
}

# --- Argument parsing ---

TASK_CONFIG=""
POLICY_NAME=""
CKPT_PATH=""
opt_seed=""
opt_jobs=""
opt_port=""
opt_timeout=""
opt_install=false

while (( $# > 0 )); do
    case "$1" in
        -m|--mode)          TASK_CONFIG="$2"; shift 2 ;;
        -n|--name)          POLICY_NAME="$2"; shift 2 ;;
        -c|--ckpt)          CKPT_PATH="$2"; shift 2 ;;
        -s|--seed)          opt_seed="$2"; shift 2 ;;
        -j|--jobs-per-gpu)  opt_jobs="$2"; shift 2 ;;
        -p|--base-port)     opt_port="$2"; shift 2 ;;
        --server-timeout)   opt_timeout="$2"; shift 2 ;;
        --install-deps)     opt_install=true; shift ;;
        -h|--help)          usage; exit 0 ;;
        -*)                 echo "Unknown option: $1" >&2; usage; exit 1 ;;
        *)                  break ;;
    esac
done

if [[ -z "${TASK_CONFIG}" || -z "${POLICY_NAME}" || -z "${CKPT_PATH}" ]]; then
    echo "Missing required flags: -m/--mode, -n/--name, -c/--ckpt" >&2
    usage
    exit 1
fi

if [[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" ]]; then
    echo "Unsupported mode: ${TASK_CONFIG} (expected demo_clean or demo_randomized)" >&2
    exit 1
fi

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint path does not exist: ${CKPT_PATH}" >&2
    exit 1
fi

if (( $# == 0 )); then
    echo "No tasks specified." >&2
    usage
    exit 1
fi

ROBOTWIN_SEED="${opt_seed:-${ROBOTWIN_SEED:-0}}"
ROBOTWIN_JOBS_PER_GPU="${opt_jobs:-${ROBOTWIN_JOBS_PER_GPU:-1}}"
ROBOTWIN_BASE_PORT="${opt_port:-${ROBOTWIN_BASE_PORT:-5694}}"
ROBOTWIN_SERVER_TIMEOUT="${opt_timeout:-${ROBOTWIN_SERVER_TIMEOUT:-600}}"
if ${opt_install}; then
    ROBOTWIN_AUTO_INSTALL_DEPS=1
fi

STARVLA_PYTHON="$(resolve_python "${STARVLA_PYTHON:-}" "${ROBOTWIN_STARVLA_ENV:-starvla}")"
ROBOTWIN_PYTHON="$(resolve_python "${ROBOTWIN_PYTHON:-}" "${ROBOTWIN_ENV:-robotwin}")"
export STARVLA_PYTHON ROBOTWIN_PYTHON

echo "[INFO] starvla python: ${STARVLA_PYTHON}"
echo "[INFO] robotwin python: ${ROBOTWIN_PYTHON}"

mapfile -t TASKS < <(resolve_tasks "$@")
mapfile -t CUDA_DEVICES < <(detect_cuda_devices)

NUM_GPUS=${#CUDA_DEVICES[@]}
JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU}"
TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
TOTAL_TASKS=${#TASKS[@]}
BASE_PORT="${ROBOTWIN_BASE_PORT}"

if (( TOTAL_SLOTS <= 0 )); then
    echo "No available execution slots were detected." >&2
    exit 1
fi

check_port_detection

prepare_runtime_dependencies

ckpt_name="$(basename "${CKPT_PATH}")"
ckpt_stem="${ckpt_name%.*}"
timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROBOTWIN_LOG_ROOT:-$(dirname "${CKPT_PATH}")/robotwin_eval_logs/${POLICY_NAME}_${TASK_CONFIG}_${ckpt_stem}_${timestamp}}"
mkdir -p "${LOG_DIR}"

next_port="${BASE_PORT}"
for gpu_id in "${CUDA_DEVICES[@]}"; do
    for (( slot_repeat = 0; slot_repeat < JOBS_PER_GPU; ++slot_repeat )); do
        assigned_port="$(find_available_port "${next_port}")"
        SLOT_GPUS+=("${gpu_id}")
        SLOT_PORTS+=("${assigned_port}")
        next_port=$((assigned_port + 1))
    done
done

echo "[INFO] mode=${TASK_CONFIG}  name=${POLICY_NAME}  seed=${ROBOTWIN_SEED}"
echo "[INFO] ckpt=${CKPT_PATH}"
echo "[INFO] logs=${LOG_DIR}"
echo "[INFO] gpus=$(join_arr ',' "${CUDA_DEVICES[@]}")  jobs_per_gpu=${JOBS_PER_GPU}  slots=${TOTAL_SLOTS}"
echo "[INFO] tasks (${TOTAL_TASKS}): $(join_arr ', ' "${TASKS[@]}")"

next_task_idx=0
completed_tasks=0

while (( completed_tasks < TOTAL_TASKS )); do
    for (( slot_idx = 0; slot_idx < TOTAL_SLOTS; ++slot_idx )); do
        current_pid="${ACTIVE_PIDS[$slot_idx]:-}"
        if [[ -n "${current_pid}" ]] && ! kill -0 "${current_pid}" 2>/dev/null; then
            finished_task="${ACTIVE_TASKS[$slot_idx]}"
            finished_eval_log="${ACTIVE_EVAL_LOGS[$slot_idx]}"
            if wait "${current_pid}"; then
                echo "[INFO] Finished task=${finished_task} slot=${slot_idx}"
            else
                exit_code=$?
                FAILED_TASKS+=("${finished_task}")
                echo "[ERROR] Task ${finished_task} failed with status ${exit_code}. See ${finished_eval_log} and ${ACTIVE_SERVER_LOGS[$slot_idx]}" >&2
            fi
            ACTIVE_PIDS[$slot_idx]=""
            ACTIVE_TASKS[$slot_idx]=""
            ACTIVE_SERVER_LOGS[$slot_idx]=""
            ACTIVE_EVAL_LOGS[$slot_idx]=""
            completed_tasks=$((completed_tasks + 1))
        fi

        if [[ -z "${ACTIVE_PIDS[$slot_idx]:-}" ]] && (( next_task_idx < TOTAL_TASKS )); then
            launch_task_in_slot "${slot_idx}" "${TASKS[$next_task_idx]}"
            next_task_idx=$((next_task_idx + 1))
        fi
    done

    if (( completed_tasks < TOTAL_TASKS )); then
        sleep 5
    fi
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[ERROR] RoboTwin evaluation finished with failures: $(join_arr ', ' "${FAILED_TASKS[@]}")" >&2
    echo "[ERROR] Logs are under ${LOG_DIR}" >&2
    exit 1
fi

echo "[INFO] RoboTwin evaluation finished successfully"
echo "[INFO] Logs are under ${LOG_DIR}"
