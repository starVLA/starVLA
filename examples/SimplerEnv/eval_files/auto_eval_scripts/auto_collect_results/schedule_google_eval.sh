cd /mnt/petrelfs/yejinhui/Projects/llavavla
# conda activate starvlaSAM

MODEL_DIR=/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1017_Qwen3PI

step=55000

SCR_ROOT=/mnt/petrelfs/yejinhui/Projects/llavavla/examples/SimplerEnv/eval_scripts/

MODEL_PATH=${MODEL_DIR}/checkpoints/steps_${step}_pytorch_model.pt
LOG_PATH=${MODEL_DIR}/checkpoints/client_logs/steps_${step}
mkdir -p $LOG_PATH


SCRIPT_PATH=${SCR_ROOT}/star_drawer_variant_agg.sh
# 2 * (378 / 42) = 18
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "${LOG_PATH}/drawer_variant.log" 2>&1 &
echo "Started drawer_variant_agg.sh with MODEL_PATH: $MODEL_PATH"
sleep 1

SCRIPT_PATH=${SCR_ROOT}/star_drawer_visual_matching.sh
# 216 / 32 = 6.75
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "${LOG_PATH}/drawer_visual_matching.log"  2>&1 &
echo "Started drawer_visual_matching.sh with MODEL_PATH: $MODEL_PATH"
sleep 1

SCRIPT_PATH=${SCR_ROOT}/star_move_near_variant_agg.sh
echo "Starting move_near_variant_agg.sh with MODEL_PATH: $MODEL_PATH"
# 10
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/move_near_variant.log" 2>&1 &
echo "Started move_near_variant_agg.sh with MODEL_PATH: $MODEL_PATH"
sleep 1

SCRIPT_PATH=${SCR_ROOT}/star_move_near_visual_matching.sh
# 4
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/move_near_visual_matching.log" 2>&1 &

echo "Started move_near_visual_matching.sh with MODEL_PATH: $MODEL_PATH"
sleep 1

SCRIPT_PATH=${SCR_ROOT}/star_pick_coke_can_variant_agg.sh
# 33
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/pick_coke_can_variant" 2>&1 &
echo "Started pick_coke_can_variant_agg.sh with MODEL_PATH: $MODEL_PATH"
sleep 1

SCRIPT_PATH=${SCR_ROOT}/star_pick_coke_can_visual_matching.sh
#  12
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/pick_coke_can_visual_matching.log" 2>&1 &
echo "Started pick_coke_can_visual_matching.sh with MODEL_PATH: $MODEL_PATH"
sleep 1

SCRIPT_PATH=${SCR_ROOT}/star_put_in_drawer_variant_agg.sh

# 7
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/put_in_drawer_variant.log" 2>&1 &
echo "Started put_in_drawer_variant_agg.sh with MODEL_PATH: $MODEL_PATH"
sleep 1

SCRIPT_PATH=${SCR_ROOT}/star_put_in_drawer_visual_matching.sh
# 12
nohup srun -p efm_p --gres=gpu:8 /bin/bash "$SCRIPT_PATH" "$MODEL_PATH" > "$LOG_PATH/put_in_drawer_visual_matching.log" 2>&1 &

echo "Started put_in_drawer_visual_matching.sh with MODEL_PATH: $MODEL_PATH"


