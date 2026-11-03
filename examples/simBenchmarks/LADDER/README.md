# LADDER Long-Horizon Manipulation Benchmark Evaluation

This directory provides the StarVLA evaluation integration for [**LADDER**](https://ladderrobotics.ai) ([*Vision-Language-Action Models Are Surprisingly Good Zero-Shot Long-Horizon Controllers*](https://arxiv.org/abs/2610.18053), [code](https://github.com/insait-institute/ladder)), a long-horizon robot manipulation benchmark spanning **145 tasks** (1 to 10 sequential subgoals) across three simulation suites:

| Suite (`--suite`) | Benchmark Name | Simulator | Robot | Tasks | Default Episodes / Task |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `simpler` | **SimplerEnv-Long** | SAPIEN 3 / ManiSkill 3 | WidowX 250S | 54 (`all1_*` – `all9_*`) | 24 |
| `robot_arena` | **RobotArena-∞-Long** | Genesis | WidowX 250S | 47 (`scene*_1goal` – `scene*_8goal`) | 8 |
| `robolab` | **RoboLab-Long** | Isaac Sim 6 / Isaac Lab 3 | Franka Panda (DROID) | 44 (`comp_*_1` – `comp_*_10`) | 8 |

Every task reports both **Task Success Rate** (`success_rate`, whether all subgoals were completed) and **Subgoal Progress** (`subgoal_progress`, fraction of subgoals completed in `[0.0, 1.0]`).

---

## 0. Environment Setup

1. **StarVLA policy server environment**: Follow the root `README.md`.
2. **LADDER simulator environment**: Clone [`insait-institute/ladder`](https://github.com/insait-institute/ladder) and set `LADDER_ROOT`:
   ```bash
   git clone https://github.com/insait-institute/ladder.git ~/ladder
   export LADDER_ROOT=~/ladder
   ```
   - For **SimplerEnv-Long** (`--suite simpler`):
     ```bash
     pip install "mani_skill==3.0.1" "sapien==3.0.3" gymnasium opencv-python-headless scipy transforms3d trimesh pyyaml deepdiff frozendict overrides websockets msgpack
     ```
   - For **RobotArena-∞-Long** (`--suite robot_arena`):
     ```bash
     pip install "genesis-world==0.4.3" "quadrants==0.4.4" "numpy<2" "scipy<1.15" "opencv-python-headless<4.12" gymnasium transforms3d pyyaml deepdiff frozendict overrides websockets msgpack
     ```
   - For **RoboLab-Long** (`--suite robolab`):
     Follow the Isaac Sim 6.0 / Isaac Lab 3.0 installation in [`insait-institute/ladder`](https://github.com/insait-institute/ladder).

---

## 1. Evaluation

### Step 1 — Start the StarVLA Policy Server

Download a Bridge/RT-1 checkpoint such as [`StarVLA/Qwen3VL-GR00T-Bridge-RT-1`](https://huggingface.co/StarVLA/Qwen3VL-GR00T-Bridge-RT-1) and launch the WebSocket server:

```bash
your_ckpt=/path/to/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt \
port=10093 gpu_id=0 \
bash examples/simBenchmarks/LADDER/eval_files/run_policy_server.sh
```

### Step 2 — Run the LADDER Evaluator

Evaluate an entire suite (`--tasks all`) or a subset of tasks:

```bash
# SimplerEnv-Long (24 parallel environments per task)
SUITE=simpler TASKS="all1_plate,all1_towel,all2_plate" PORT=10093 \
bash examples/simBenchmarks/LADDER/eval_files/eval_ladder.sh

# RobotArena-∞-Long (8 parallel environments per task)
SUITE=robot_arena TASKS="scene54_1goal,scene76_1goal" PORT=10093 \
bash examples/simBenchmarks/LADDER/eval_files/eval_ladder.sh
```

To evaluate with oracle subgoal decomposition (`oracle_hlc`) instead of the full long-horizon task instruction (`flat`):

```bash
SUITE=simpler PROMPT_MODE=oracle_hlc PORT=10093 \
bash examples/simBenchmarks/LADDER/eval_files/eval_ladder.sh
```

---

## 2. Baseline Results (`StarVLA/Qwen3VL-GR00T-Bridge-RT-1`)

Sample zero-shot evaluation results using `StarVLA/Qwen3VL-GR00T-Bridge-RT-1` (`unnorm_key=oxe_bridge`, `prompt_mode=flat`):

| Suite | Task | Subgoals | Episodes | Success Rate | Subgoal Progress |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `simpler` | `all1_plate` | 1 | 24 | 79.2% | 79.2% |
| `simpler` | `all1_towel` | 1 | 24 | 83.3% | 83.3% |
| `simpler` | `all2_plate` | 2 | 24 | 0.0% | 47.9% |
| `robot_arena` | `scene54_1goal` | 1 | 8 | 37.5% | 37.5% |
| `robot_arena` | `scene76_1goal` | 1 | 8 | 12.5% | 12.5% |
