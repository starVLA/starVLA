#!/usr/bin/env zsh
emulate -L zsh
set -euo pipefail
setopt pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

typeset -ga ROBOTWIN_ALL_TASKS=(
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

typeset -ga used_ports=()
typeset -ga SLOT_GPUS=()
typeset -ga SLOT_PORTS=()
typeset -ga ACTIVE_PIDS=()
typeset -ga ACTIVE_TASKS=()
typeset -ga ACTIVE_SERVER_LOGS=()
typeset -ga ACTIVE_EVAL_LOGS=()
typeset -ga FAILED_TASKS=()

cleanup_active_jobs() {
    local pid=""
    for pid in "${ACTIVE_PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "${ACTIVE_PIDS[@]}"; do
        if [[ -n "${pid}" ]]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup_active_jobs EXIT INT TERM

usage() {
    cat >&2 <<'EOF'
Usage:
  zsh start_eval.sh <demo_clean|demo_randomized> <task ... | task_file> <policy_name> <ckpt_path>

Examples:
  zsh start_eval.sh demo_randomized adjust_bottle my_eval /path/to/ckpt.pt
  zsh start_eval.sh demo_clean adjust_bottle open_laptop my_eval /path/to/ckpt.pt
  zsh start_eval.sh demo_randomized task_list.txt my_eval /path/to/ckpt.pt

Notes:
  - The last two arguments are always treated as <policy_name> and <ckpt_path>.
  - When a single task argument is an existing file, tasks are read one-per-line.
  - Use `all` as a task argument to evaluate all RoboTwin 2.0 tasks.

Optional environment variables:
  ROBOTWIN_PATH              Path to the RoboTwin repository.
  ROBOTWIN_STARVLA_ENV       Conda env name for the policy server. Default: starvla
  ROBOTWIN_ENV               Conda env name for RoboTwin eval. Default: robotwin
  ROBOTWIN_BASE_PORT         First port to allocate. Default: 5694
  ROBOTWIN_JOBS_PER_GPU      Concurrent jobs per visible GPU. Default: 1
  ROBOTWIN_SERVER_TIMEOUT    Seconds to wait for the policy server. Default: 600
  ROBOTWIN_SEED              Eval seed. Default: 0
  ROBOTWIN_AUTO_INSTALL_DEPS Set to 1 to run pip install bootstrap steps once.
EOF
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    print -r -- "${value}"
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
        return 1
    fi
}

port_reserved() {
    local port="$1"
    local reserved_port
    for reserved_port in "${used_ports[@]}"; do
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
    print -r -- "${port}"
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
        devices=("${(@s:,:)CUDA_VISIBLE_DEVICES}")
    elif command -v nvidia-smi >/dev/null 2>&1; then
        gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
        if [[ -n "${gpu_count}" && "${gpu_count}" != "0" ]]; then
            for (( idx = 0; idx < gpu_count; ++idx )); do
                devices+=("${idx}")
            done
        fi
    fi

    for device in "${devices[@]}"; do
        device="$(trim "${device}")"
        if [[ -n "${device}" ]]; then
            cleaned+=("${device}")
        fi
    done

    if (( ${#cleaned[@]} == 0 )); then
        cleaned=(0)
    fi

    print -l -- "${cleaned[@]}"
}

resolve_tasks() {
    local -a raw_inputs=("$@")
    local -a resolved=()
    local -a split_inputs=()
    local input=""
    local task=""
    local line=""

    if (( ${#raw_inputs[@]} == 1 )) && [[ -f "${raw_inputs[1]}" ]]; then
        while IFS= read -r line || [[ -n "${line}" ]]; do
            line="$(trim "${line%%#*}")"
            if [[ -n "${line}" ]]; then
                resolved+=("${line}")
            fi
        done < "${raw_inputs[1]}"
    else
        for input in "${raw_inputs[@]}"; do
            if [[ "${input}" == "all" ]]; then
                resolved+=("${ROBOTWIN_ALL_TASKS[@]}")
                continue
            fi
            split_inputs=("${(@s:,:)input}")
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

    print -l -- "${resolved[@]}"
}

prepare_runtime_dependencies() {
    if [[ "${ROBOTWIN_AUTO_INSTALL_DEPS:-0}" != "1" ]]; then
        return 0
    fi

    echo "[INFO] Installing runtime dependencies into ${ROBOTWIN_STARVLA_ENV:-starvla} and ${ROBOTWIN_ENV:-robotwin}"
    source_shell_rc
    conda activate "${ROBOTWIN_STARVLA_ENV:-starvla}"
    pip install snntorch
    conda activate "${ROBOTWIN_ENV:-robotwin}"
    pip install -r "${SCRIPT_DIR}/requirements.txt"
}

source_shell_rc() {
    set +u
    if [[ -f ~/.zshrc ]]; then
        source ~/.zshrc
    fi
    set -u
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
        emulate -L zsh
        set -euo pipefail

        local server_pid=""
        cleanup_server() {
            if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
                kill "${server_pid}" 2>/dev/null || true
                wait "${server_pid}" 2>/dev/null || true
            fi
        }
        trap cleanup_server EXIT INT TERM

        export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

        source_shell_rc
        conda activate "${ROBOTWIN_STARVLA_ENV:-starvla}"
        bash "${SCRIPT_DIR}/run_policy_server.sh" "${CKPT_PATH}" "${gpu_id}" "${port}" > "${server_log}" 2>&1 &
        server_pid=$!

        if ! wait_for_server "${port}" "${ROBOTWIN_SERVER_TIMEOUT:-600}"; then
            echo "[ERROR] Policy server failed to become ready for task=${task_name} on port=${port}. See ${server_log}" >&2
            exit 1
        fi

        source_shell_rc
        conda activate "${ROBOTWIN_ENV:-robotwin}"
        cd "${SCRIPT_DIR}"
        bash "${SCRIPT_DIR}/eval.sh" \
            "${task_name}" \
            "${TASK_CONFIG}" \
            "${POLICY_NAME}" \
            "${ROBOTWIN_SEED:-0}" \
            "${gpu_id}" \
            "${CKPT_PATH}" \
            "${port}" \
            > "${eval_log}" 2>&1
    ) &

    launched_pid=$!
    ACTIVE_PIDS[$slot_idx]="${launched_pid}"
    ACTIVE_TASKS[$slot_idx]="${task_name}"
    ACTIVE_SERVER_LOGS[$slot_idx]="${server_log}"
    ACTIVE_EVAL_LOGS[$slot_idx]="${eval_log}"
}

if (( $# < 4 )); then
    usage
    exit 1
fi

TASK_CONFIG="$1"
shift

if [[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" ]]; then
    echo "Unsupported task config: ${TASK_CONFIG}" >&2
    exit 1
fi

if (( $# < 3 )); then
    usage
    exit 1
fi

argc=$#
POLICY_NAME="${argv[$((argc - 1))]}"
CKPT_PATH="${argv[$argc]}"
TASK_INPUTS=("${argv[1,$((argc - 2))]}")

if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Checkpoint path does not exist: ${CKPT_PATH}" >&2
    exit 1
fi

TASKS=("${(@f)$(resolve_tasks "${TASK_INPUTS[@]}")}")
CUDA_DEVICES=("${(@f)$(detect_cuda_devices)}")

NUM_GPUS=${#CUDA_DEVICES[@]}
JOBS_PER_GPU="${ROBOTWIN_JOBS_PER_GPU:-1}"
TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
TOTAL_TASKS=${#TASKS[@]}
BASE_PORT="${ROBOTWIN_BASE_PORT:-5694}"

if (( TOTAL_SLOTS <= 0 )); then
    echo "No available execution slots were detected." >&2
    exit 1
fi

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

echo "[INFO] task_config=${TASK_CONFIG}"
echo "[INFO] policy_name=${POLICY_NAME}"
echo "[INFO] ckpt_path=${CKPT_PATH}"
echo "[INFO] log_dir=${LOG_DIR}"
echo "[INFO] cuda_devices=${(j:,:)CUDA_DEVICES}"
echo "[INFO] jobs_per_gpu=${JOBS_PER_GPU}"
echo "[INFO] total_tasks=${TOTAL_TASKS}"
echo "[INFO] total_slots=${TOTAL_SLOTS}"

typeset -i next_task_idx=1
typeset -i completed_tasks=0

while (( completed_tasks < TOTAL_TASKS )); do
    for (( slot_idx = 1; slot_idx <= TOTAL_SLOTS; ++slot_idx )); do
        current_pid="${ACTIVE_PIDS[$slot_idx]:-}"
        if [[ -n "${current_pid}" ]] && ! kill -0 "${current_pid}" 2>/dev/null; then
            if wait "${current_pid}"; then
                echo "[INFO] Finished task=${ACTIVE_TASKS[$slot_idx]} slot=${slot_idx}"
            else
                exit_code=$?
                FAILED_TASKS+=("${ACTIVE_TASKS[$slot_idx]}")
                echo "[ERROR] Task ${ACTIVE_TASKS[$slot_idx]} failed with status ${exit_code}. See ${ACTIVE_EVAL_LOGS[$slot_idx]} and ${ACTIVE_SERVER_LOGS[$slot_idx]}" >&2
            fi
            ACTIVE_PIDS[$slot_idx]=""
            ACTIVE_TASKS[$slot_idx]=""
            ACTIVE_SERVER_LOGS[$slot_idx]=""
            ACTIVE_EVAL_LOGS[$slot_idx]=""
            completed_tasks=$((completed_tasks + 1))
        fi

        if [[ -z "${ACTIVE_PIDS[$slot_idx]:-}" ]] && (( next_task_idx <= TOTAL_TASKS )); then
            launch_task_in_slot "${slot_idx}" "${TASKS[$next_task_idx]}"
            next_task_idx=$((next_task_idx + 1))
        fi
    done

    if (( completed_tasks < TOTAL_TASKS )); then
        sleep 5
    fi
done

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "[ERROR] RoboTwin evaluation finished with failures: ${(j:, :)FAILED_TASKS}" >&2
    echo "[ERROR] Logs are under ${LOG_DIR}" >&2
    exit 1
fi

echo "[INFO] RoboTwin evaluation finished successfully"
echo "[INFO] Logs are under ${LOG_DIR}"
