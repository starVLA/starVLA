#!/usr/bin/env bash
# Install LIBERO environment for evaluation.
set -euo pipefail

###########################################################################################
# Usage
#
#   bash examples/LIBERO/eval_files/install_libero.sh
#
# Optional overrides:
#
#   LIBERO_CONDA_ENV=libero \
#   LIBERO_PARENT_DIR=/mnt/sda/wangbo \
#   bash examples/LIBERO/eval_files/install_libero.sh
#
# Or skip conda activation and use the current Python directly:
#
#   SKIP_CONDA_ACTIVATE=1 bash examples/LIBERO/eval_files/install_libero.sh
###########################################################################################

LIBERO_CONDA_ENV="${LIBERO_CONDA_ENV:-libero}"
LIBERO_PARENT_DIR="${LIBERO_PARENT_DIR:-$HOME}"
LIBERO_DIR="${LIBERO_DIR:-${LIBERO_PARENT_DIR}/LIBERO}"
SKIP_CONDA_ACTIVATE="${SKIP_CONDA_ACTIVATE:-0}"

echo "=== Step 1: Activate LIBERO Python environment ==="
if [[ "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda not found. Either initialize conda first or run with SKIP_CONDA_ACTIVATE=1."
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${LIBERO_CONDA_ENV}"
else
    echo "Skipping conda activation; using current python: $(command -v python)"
fi

echo "=== Step 2: Install MuJoCo and eval dependencies ==="
python -m pip install mujoco==3.2.3
python -m pip install tyro matplotlib mediapy websockets msgpack
python -m pip install numpy==1.24.4

echo "=== Step 3: Clone LIBERO if needed ==="
mkdir -p "${LIBERO_PARENT_DIR}"
if [[ ! -d "${LIBERO_DIR}" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
else
    echo "LIBERO already exists at ${LIBERO_DIR}"
fi

echo "=== Step 4: Install LIBERO (editable) ==="
cd "${LIBERO_DIR}"
python -m pip install -e .

echo "=== Step 5: Verify installation ==="
python -c "from libero.libero import benchmark; print('LIBERO OK:', benchmark)"
python -c "import mujoco; print('MuJoCo OK:', mujoco.__version__)"
python -c "import tyro; print('tyro OK')"
python -c "import websockets; print('websockets OK')"

echo "=== ALL DONE ==="
echo "LIBERO_DIR=${LIBERO_DIR}"
