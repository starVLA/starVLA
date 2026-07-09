# MetaWorld MT50 Evaluation

This document provides instructions for evaluating starVLA on the [MetaWorld MT50](https://github.com/Farama-Foundation/Metaworld) benchmark (50 manipulation tasks across 4 difficulty buckets).

The evaluation uses the same client-server architecture as LIBERO: the starVLA policy server handles inference, while the MetaWorld client drives the simulation.

---

## 0. Environment Setup

### MetaWorld environment

```bash
pip install metaworld gymnasium
pip install tyro imageio opencv-python-headless numpy
```

> MetaWorld requires MuJoCo. If not already installed:
> ```bash
> pip install mujoco
> ```

### starVLA server environment

Follow the main starVLA installation instructions. The policy server (`deployment/model_server/server_policy.py`) is shared across all benchmarks.

---

## 1. Evaluation Workflow

Run from the **repository root** using **two separate terminals**.

### Step 1. Start the policy server (starVLA environment)

```bash
CKPT=/path/to/your/checkpoint.pt PORT=10095 GPU_ID=0 \
  bash examples/simBenchmarks/MetaWorld/eval_files/run_policy_server.sh
```

### Step 2. Run the evaluation (MetaWorld environment)

```bash
HOST=127.0.0.1 PORT=10095 EPISODES_PER_TASK=10 \
  bash examples/simBenchmarks/MetaWorld/eval_files/eval_metaworld.sh
```

Or call the Python script directly for finer control:

```bash
python examples/simBenchmarks/MetaWorld/eval_files/eval_metaworld.py \
  --args.host 127.0.0.1 \
  --args.port 10095 \
  --args.episodes-per-task 10 \
  --args.levels easy,medium,hard,very_hard \
  --args.video-out-path experiments/metaworld/logs
```

---

## 2. Key Differences from LIBERO

| Aspect | LIBERO | MetaWorld MT50 |
|--------|--------|----------------|
| Action space | 7D (xyz + rotation + gripper) | 4D (xyz + gripper) |
| Camera views | 2 (primary + wrist) | 1 (corner2) |
| Image preprocessing | `[::-1, ::-1]` flip | ROT180 + center\_crop(2/3) + resize 224 |
| Tasks | 4 suites x 10 tasks | 50 tasks, 4 difficulty buckets |
| Episodes/task | 50 | 10 |
| Max steps | 220-520 (per suite) | 400 |
| Gripper | Binarized | Continuous, clipped to `[-1, 1]` |

---

## 3. Output

- **Videos**: One `.mp4` per episode saved to `--video-out-path`
- **Summary**: `summary.json` with per-bucket and overall success rates
- **Metrics**: Overall SR = mean of per-bucket SRs (easy, medium, hard, very\_hard)
