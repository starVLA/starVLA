#!/bin/bash
# auto_eval_vla_arena.sh
#
# Launches parallel evaluation jobs for all 11 VLA-Arena task suites at a
# chosen difficulty level.  Each suite gets its own GPU slot (round-robin).
#
# Usage:
#   bash auto_eval_vla_arena.sh
#
# Set your_ckpt and task_level below, then run from the starVLA root directory.

cd "$(dirname "$0")/../../../.."   # navigate to starVLA root

SCRIPT_PATH="./examples/VLA-Arena/eval_files/auto_eval_scripts/eval_vla_arena_parall.sh"

###########################################################################################
# === Please modify the following variables ===
your_ckpt=results/Checkpoints/vla_arena_qwenoft_all/checkpoints/steps_50000_pytorch_model.pt
task_level=0          # 0 | 1 | 2
run_index_base=400    # base index; each suite gets run_index_base + offset
###########################################################################################

# -----------------------------------------------------------------------
# Safety suites (5)
# -----------------------------------------------------------------------
bash "${SCRIPT_PATH}" "${your_ckpt}" safety_static_obstacles   "${task_level}" $((run_index_base + 0)) &
sleep 20
bash "${SCRIPT_PATH}" "${your_ckpt}" safety_cautious_grasp     "${task_level}" $((run_index_base + 1)) &
sleep 20
bash "${SCRIPT_PATH}" "${your_ckpt}" safety_hazard_avoidance   "${task_level}" $((run_index_base + 2)) &
sleep 20
bash "${SCRIPT_PATH}" "${your_ckpt}" safety_state_preservation "${task_level}" $((run_index_base + 3)) &
sleep 20
bash "${SCRIPT_PATH}" "${your_ckpt}" safety_dynamic_obstacles  "${task_level}" $((run_index_base + 4)) &
sleep 20

# -----------------------------------------------------------------------
# Distractor suites (2)
# -----------------------------------------------------------------------
bash "${SCRIPT_PATH}" "${your_ckpt}" distractor_static_distractors  "${task_level}" $((run_index_base + 5)) &
sleep 20
bash "${SCRIPT_PATH}" "${your_ckpt}" distractor_dynamic_distractors "${task_level}" $((run_index_base + 6)) &
sleep 20

# -----------------------------------------------------------------------
# Extrapolation suites (3)
# -----------------------------------------------------------------------
bash "${SCRIPT_PATH}" "${your_ckpt}" extrapolation_preposition_combinations "${task_level}" $((run_index_base + 7)) &
sleep 20
bash "${SCRIPT_PATH}" "${your_ckpt}" extrapolation_task_workflows            "${task_level}" $((run_index_base + 8)) &
sleep 20
bash "${SCRIPT_PATH}" "${your_ckpt}" extrapolation_unseen_objects            "${task_level}" $((run_index_base + 9)) &
sleep 20

# -----------------------------------------------------------------------
# Long-horizon suite (1)
# -----------------------------------------------------------------------
bash "${SCRIPT_PATH}" "${your_ckpt}" long_horizon "${task_level}" $((run_index_base + 10)) &

echo "All evaluation jobs launched.  Check logs under results/*/logs/"
