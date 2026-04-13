#!/bin/bash
#SBATCH --job-name=framework_test
#SBATCH --partition=preempt
#SBATCH --account=vonneumann1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=./tmp/sbatch_output_%j.log
#SBATCH --error=./tmp/sbatch_error_%j.log

# Activate conda
source ~/.bashrc
conda activate starVLA

cd /home/jye624/Projcets/starVLA

echo "=== Starting Framework Tests $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

FRAMEWORKS=(
    "QwenPI.py"
    "QwenAdapter.py"
    "M1.py"
    "QwenGR00T.py"
    "QwenDual.py"
    "QwenFast.py"
    "LangForce.py"
    "ABot_M0.py"
)

PASS=0
FAIL=0

for fw in "${FRAMEWORKS[@]}"; do
    echo "======================================"
    echo "Running: $fw  ($(date))"
    echo "======================================"
    
    python starVLA/model/framework/$fw > ./tmp/run_${fw%.py}.log 2>&1
    EXIT=$?
    
    if [ $EXIT -eq 0 ]; then
        echo "  RESULT: PASS (EXIT=$EXIT)"
        PASS=$((PASS+1))
    else
        echo "  RESULT: FAIL (EXIT=$EXIT)"
        echo "  Last 5 lines of error:"
        tail -5 ./tmp/run_${fw%.py}.log | sed 's/^/    /'
        FAIL=$((FAIL+1))
    fi
    echo ""
done

echo "======================================"
echo "SUMMARY: $PASS passed, $FAIL failed out of ${#FRAMEWORKS[@]} total"
echo "Finished at $(date)"
echo "======================================"
