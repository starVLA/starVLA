#!/bin/bash
# eval_vla_arena.sh
#
# Batch evaluation of a starVLA policy on one or more VLA-Arena task suites
# and difficulty levels.  The policy server must already be running
# (see run_policy_server.sh).
#
# Usage:
#   bash examples/VLA-Arena/eval_files/eval_vla_arena.sh [OPTIONS]
# Run from the starVLA root directory.

###########################################################################################
# === Please modify the following paths according to your environment ===
export VLA_ARENA_HOME=/path/to/VLA-Arena        # Root of the VLA-Arena repository
export VLA_ARENA_python=python                  # Python env that has vla_arena installed

export starVLA_HOME=$(pwd)
export PYTHONPATH=${VLA_ARENA_HOME}/vla_arena:${PYTHONPATH}
export PYTHONPATH=${starVLA_HOME}:${PYTHONPATH}
# === End of environment variable configuration ===
###########################################################################################

# --- Default configuration ---
your_ckpt=/path/to/checkpoint.pt
host="127.0.0.1"
port=10093

DEFAULT_NUM_TRIALS=10
DEFAULT_SEED=7
save_video_mode=first_success_failure   # all | first_success_failure | none

RUN_DATE=$(date +"%Y%m%d")
RUN_TIME=$(date +"%H%M%S")
RESULTS_DIR="./results/${RUN_DATE}/${RUN_TIME}"
TIMESTAMP="${RUN_DATE}_${RUN_TIME}"

# Visual perturbation (set to true to enable)
add_noise=false
adjust_light=false
randomize_color=false
camera_offset=false

# Safety constraint
apply_safety_constraint=false

# Initial state selection
init_state_offset_random=true

# Default task suites (comment/uncomment as needed)
TASK_SUITES=(
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

TASK_LEVELS=(0 1 2)

# --- Color output ---
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

Batch evaluation of a starVLA policy server against VLA-Arena task suites.

OPTIONS:
    -c, --checkpoint PATH      Path to pretrained checkpoint (required)
    -t, --trials NUM           Rollout episodes per task (default: $DEFAULT_NUM_TRIALS)
    -s, --seed NUM             Random seed (default: $DEFAULT_SEED)
    -o, --output-dir DIR       Output directory for logs and summary (default: $RESULTS_DIR)
    --host HOST                Policy server host (default: $host)
    --port PORT                Policy server port (default: $port)
    --suites "s1 s2 ..."       Space-separated list of task suites to evaluate
    --levels "0 1 2"           Space-separated list of difficulty levels to evaluate
    --save-video MODE          Video save mode: all|first_success_failure|none (default: $save_video_mode)
    --skip-existing            Skip suites whose log file already exists
    --dry-run                  Print commands without executing
    -h, --help                 Show this help message

EXAMPLES:
    # Evaluate a checkpoint on all safety suites at levels 0 and 1
    $0 -c /path/to/ckpt.pt --suites "safety_static_obstacles safety_cautious_grasp" --levels "0 1"

    # Evaluate with dry-run to preview commands
    $0 -c /path/to/ckpt.pt --dry-run
EOF
}

# --- Argument Parsing ---
NUM_TRIALS="$DEFAULT_NUM_TRIALS"
SEED="$DEFAULT_SEED"
OUTPUT_DIR="$RESULTS_DIR"
SKIP_EXISTING=false
DRY_RUN=false
CUSTOM_SUITES=""
CUSTOM_LEVELS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)    your_ckpt="$2"; shift 2 ;;
        -t|--trials)        NUM_TRIALS="$2"; shift 2 ;;
        -s|--seed)          SEED="$2"; shift 2 ;;
        -o|--output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --host)             host="$2"; shift 2 ;;
        --port)             port="$2"; shift 2 ;;
        --suites)           CUSTOM_SUITES="$2"; shift 2 ;;
        --levels)           CUSTOM_LEVELS="$2"; shift 2 ;;
        --save-video)       save_video_mode="$2"; shift 2 ;;
        --skip-existing)    SKIP_EXISTING=true; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        -h|--help)          show_usage; exit 0 ;;
        *) print_error "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

if [[ -n "$CUSTOM_SUITES" ]]; then TASK_SUITES=($CUSTOM_SUITES); fi
if [[ -n "$CUSTOM_LEVELS" ]]; then TASK_LEVELS=($CUSTOM_LEVELS); fi

mkdir -p "$OUTPUT_DIR"
SUMMARY_FILE="${OUTPUT_DIR}/summary_${TIMESTAMP}.csv"

# ---------------------------------------------------------------------------
# Data Extraction Functions
# Parses lines emitted by eval_vla_arena.py, e.g.:
#   "[suite_name] Final SR: 0.6000 (30/50)  avg_cost=1.2345"
# ---------------------------------------------------------------------------

extract_success_rate() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        local v=$(grep -i "Final SR:" "$log_file" | tail -1 | sed 's/.*Final SR: //' | awk '{print $1}')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_total_successes() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        # "(30/50)" → numerator
        local v=$(grep -i "Final SR:" "$log_file" | tail -1 | grep -o '([0-9]*/[0-9]*)' | tr -d '()' | cut -d'/' -f1)
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_total_episodes() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        # "(30/50)" → denominator
        local v=$(grep -i "Final SR:" "$log_file" | tail -1 | grep -o '([0-9]*/[0-9]*)' | tr -d '()' | cut -d'/' -f2)
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

extract_avg_cost() {
    local log_file="$1"
    if [[ -f "$log_file" ]]; then
        local v=$(grep -i "avg_cost=" "$log_file" | tail -1 | sed 's/.*avg_cost=//' | awk '{print $1}')
        echo "${v:-N/A}"
    else echo "N/A"; fi
}

print_error_details() {
    local log_file="$1"
    local suite="$2"
    local level="$3"
    print_error "Failed: $suite L$level"
    if [[ -f "$log_file" ]]; then
        echo "--- Last 30 lines of log ---"
        tail -30 "$log_file" | sed 's/^/  /'
        echo "----------------------------"
        if grep -q "Traceback" "$log_file"; then
            print_error "Python traceback found:"
            echo "--- Traceback ---"
            grep -A 20 "Traceback" "$log_file" | sed 's/^/  /'
            echo "-----------------"
        fi
    fi
}

# ---------------------------------------------------------------------------
# run_evaluation <suite> <level>
# ---------------------------------------------------------------------------
run_evaluation() {
    local suite="$1"
    local level="$2"

    local folder_name
    folder_name=$(echo "${your_ckpt}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
    local run_id="${suite}_L${level}_${folder_name}_${TIMESTAMP}"
    local suite_log_dir="${OUTPUT_DIR}/${suite}"
    local log_file="${suite_log_dir}/${run_id}.log"
    local video_out_path="${OUTPUT_DIR}/videos/${suite}_L${level}/${folder_name}"

    if [[ "$SKIP_EXISTING" == true && -f "$log_file" ]]; then
        print_warning "Skipping $suite L$level (log exists: $log_file)"
        return 0
    fi

    local cmd=(
        "${VLA_ARENA_python}" ./examples/VLA-Arena/eval_files/eval_vla_arena.py
        --args.pretrained-path "${your_ckpt}"
        --args.host "${host}"
        --args.port "${port}"
        --args.task-suite-name "${suite}"
        --args.task-level "${level}"
        --args.num-trials-per-task "${NUM_TRIALS}"
        --args.seed "${SEED}"
        --args.video-out-path "${video_out_path}"
        --args.save-video-mode "${save_video_mode}"
    )

    # For these switches, presence means true.
    [[ "${add_noise}" == true ]] && cmd+=(--args.add-noise)
    [[ "${adjust_light}" == true ]] && cmd+=(--args.adjust-light)
    [[ "${randomize_color}" == true ]] && cmd+=(--args.randomize-color)
    [[ "${camera_offset}" == true ]] && cmd+=(--args.camera-offset)
    [[ "${apply_safety_constraint}" == true ]] && cmd+=(--args.apply-safety-constraint)
    [[ "${init_state_offset_random}" == true ]] && cmd+=(--args.init-state-offset-random)

    if [[ "$DRY_RUN" == true ]]; then
        local cmd_str
        printf -v cmd_str '%q ' "${cmd[@]}"
        print_info "DRY RUN: ${cmd_str}"
        return 0
    fi

    mkdir -p "${suite_log_dir}"
    mkdir -p "${video_out_path}"
    print_info "Running: $suite  L$level"
    print_info "Log  → $log_file"

    if "${cmd[@]}" > "$log_file" 2>&1; then
        local sr ts te ac
        sr=$(extract_success_rate    "$log_file")
        ts=$(extract_total_successes "$log_file")
        te=$(extract_total_episodes  "$log_file")
        ac=$(extract_avg_cost        "$log_file")

        print_success "Done $suite L$level: SR=$sr ($ts/$te)  avg_cost=$ac"
        echo "$suite,L$level,$sr,$ts,$te,$ac,$log_file" >> "$SUMMARY_FILE"
        return 0
    else
        print_error_details "$log_file" "$suite" "$level"
        echo "$suite,L$level,FAILED,N/A,N/A,N/A,$log_file" >> "$SUMMARY_FILE"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Main batch loop
# ---------------------------------------------------------------------------
print_info "Batch evaluation started at $(date)"
print_info "Checkpoint : ${your_ckpt}"
print_info "Suites     : ${TASK_SUITES[*]}"
print_info "Levels     : ${TASK_LEVELS[*]}"
print_info "Trials/task: ${NUM_TRIALS}  |  Seed: ${SEED}"
print_info "Summary CSV: ${SUMMARY_FILE}"

echo "Task Suite,Level,Success Rate,Successes,Total Episodes,Avg Cost,Log File" > "$SUMMARY_FILE"

total_evaluations=$((${#TASK_SUITES[@]} * ${#TASK_LEVELS[@]}))
current_evaluation=0
successful_evaluations=0
failed_evaluations=0

for suite in "${TASK_SUITES[@]}"; do
    for level in "${TASK_LEVELS[@]}"; do
        current_evaluation=$((current_evaluation + 1))
        print_info "── Progress: $current_evaluation / $total_evaluations ──────────────────────────"

        if run_evaluation "$suite" "$level"; then
            successful_evaluations=$((successful_evaluations + 1))
        else
            failed_evaluations=$((failed_evaluations + 1))
        fi
        sleep 2
    done
done

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
echo ""
print_info "Batch evaluation finished at $(date)"
print_info "Successful: ${successful_evaluations} / ${total_evaluations}  |  Failed: ${failed_evaluations}"
print_success "Results saved to: ${SUMMARY_FILE}"