# MetaWorld MT50 Evaluation

This document provides instructions for evaluating starVLA on the [MetaWorld MT50](https://github.com/Farama-Foundation/Metaworld) benchmark (50 manipulation tasks across 4 difficulty buckets).

The evaluation uses a client-server architecture: the starVLA policy server handles inference, while the MetaWorld client drives the simulation.

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

> Don't have a checkpoint yet? Grab a reference checkpoint from Section 4.

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

## 2. Benchmark Specifics

| Aspect | MetaWorld MT50 |
|--------|----------------|
| Action space | 4D (xyz + gripper) |
| Camera views | 1 (corner2) |
| Image preprocessing | ROT180 + center\_crop(2/3) + resize 224 |
| Tasks | 50 tasks, 4 difficulty buckets |
| Episodes/task | 10 |
| Max steps | 400 |
| Gripper | Continuous, clipped to `[-1, 1]` |

---

## 3. Output

- **Videos**: One `.mp4` per episode saved to `--video-out-path`
- **Summary**: `summary.json` with per-bucket and overall success rates
- **Metrics**: Overall SR = mean of per-bucket SRs (easy, medium, hard, very\_hard)

---

## 4. Reference Checkpoints & Results

We release two reference checkpoints for this benchmark. Both are QwenPI_v3
(base VLM: Qwen2.5-VL-3B-Instruct) finetuned on the MetaWorld MT50 LeRobot
dataset with the identical recipe:

| Checkpoint | Initialization |
|------------|----------------|
| [starvla_metaworld_qwenpiv3_baseline_finetune](https://huggingface.co/MINT-SJTU/starvla_metaworld_qwenpiv3_baseline_finetune) | direct finetune from the base VLM |
| [starvla_metaworld_qwenpiv3_la_finetune](https://huggingface.co/MINT-SJTU/starvla_metaworld_qwenpiv3_la_finetune) | finetune from an **LA-pretrained (LA4VLA)** backbone — [Language-Action pretraining](https://arxiv.org/abs/2606.27295), learning action prediction from language instructions and robot states |

```bash
# Download a checkpoint (keeps the layout the policy server expects;
# swap the repo id for the other checkpoint)
huggingface-cli download MINT-SJTU/starvla_metaworld_qwenpiv3_la_finetune \
  --local-dir ./ckpts/starvla_metaworld_qwenpiv3_la_finetune

# Serve it (Step 1 of the workflow above)
CKPT=./ckpts/starvla_metaworld_qwenpiv3_la_finetune/checkpoints/steps_60000_pytorch_model.pt \
  PORT=10095 GPU_ID=0 bash examples/simBenchmarks/MetaWorld/eval_files/run_policy_server.sh
```

Each repo contains the single evaluated weight plus the files the server resolves
from the run directory:

```
checkpoints/steps_60000_pytorch_model.pt   # evaluated weight (~8.8 GB)
config.yaml                                # model/run config
config.full.yaml
dataset_statistics.json                    # action de-normalization stats
```

Results, both checkpoints at `steps_60000`, evaluated with the exact workflow in
Section 1 (all 50 tasks × 10 episodes = 500 episodes, seed 4042, max 400 steps
per episode; overall SR = mean over the four difficulty buckets):

| Checkpoint | easy (28) | medium (11) | hard (6) | very_hard (5) | **Overall** |
|------------|-----------|-------------|----------|---------------|-------------|
| `la_finetune` | 0.846 | 0.673 | 0.517 | 0.800 | **0.709** |
| `baseline_finetune` | 0.857 | 0.600 | 0.467 | 0.440 | 0.591 |

**Per-task success rates** (10 episodes each; `la_ft` = la_finetune, `base_ft` = baseline_finetune):

**easy** (`la_ft` 0.846 / `base_ft` 0.857)

| Task | la_ft | base_ft | Task | la_ft | base_ft |
|------|-------|---------|------|-------|---------|
| button-press-topdown-v3 | 1.0 | 1.0 | handle-press-side-v3 | 1.0 | 1.0 |
| button-press-topdown-wall-v3 | 1.0 | 1.0 | handle-press-v3 | 1.0 | 1.0 |
| button-press-v3 | 1.0 | 1.0 | handle-pull-side-v3 | 0.1 | 0.4 |
| button-press-wall-v3 | 1.0 | 1.0 | handle-pull-v3 | 0.1 | 0.2 |
| coffee-button-v3 | 1.0 | 1.0 | lever-pull-v3 | 0.8 | 0.8 |
| dial-turn-v3 | 0.4 | 0.9 | plate-slide-v3 | 0.9 | 0.9 |
| door-close-v3 | 1.0 | 1.0 | plate-slide-side-v3 | 1.0 | 1.0 |
| door-lock-v3 | 1.0 | 0.3 | plate-slide-back-v3 | 0.7 | 0.8 |
| door-v3 | 1.0 | 1.0 | plate-slide-back-side-v3 | 0.9 | 1.0 |
| door-unlock-v3 | 1.0 | 1.0 | peg-unplug-side-v3 | 0.6 | 0.7 |
| drawer-close-v3 | 1.0 | 1.0 | reach-v3 | 0.4 | 0.1 |
| drawer-open-v3 | 1.0 | 1.0 | reach-wall-v3 | 0.9 | 0.9 |
| faucet-open-v3 | 1.0 | 1.0 | window-open-v3 | 1.0 | 1.0 |
| faucet-close-v3 | 0.9 | 1.0 | window-close-v3 | 1.0 | 1.0 |

**medium** (`la_ft` 0.673 / `base_ft` 0.600)

| Task | la_ft | base_ft | Task | la_ft | base_ft |
|------|-------|---------|------|-------|---------|
| basketball-v3 | 0.3 | 0.2 | peg-insertion-side-v3 | 0.5 | 0.3 |
| bin-picking-v3 | 0.5 | 0.7 | soccer-v3 | 0.3 | 0.0 |
| box-close-v3 | 0.7 | 0.7 | push-wall-v3 | 1.0 | 0.9 |
| coffee-pull-v3 | 0.8 | 0.9 | sweep-into-goal-v3 | 1.0 | 0.8 |
| coffee-push-v3 | 0.7 | 0.5 | sweep-v3 | 0.8 | 0.6 |
| hammer-v3 | 0.8 | 1.0 | | | |

**hard** (`la_ft` 0.517 / `base_ft` 0.467)

| Task | la_ft | base_ft | Task | la_ft | base_ft |
|------|-------|---------|------|-------|---------|
| nut-assembly-v3 | 0.6 | 0.7 | pick-place-v3 | 0.2 | 0.4 |
| hand-insert-v3 | 1.0 | 0.6 | push-v3 | 1.0 | 0.6 |
| pick-out-of-hole-v3 | 0.0 | 0.0 | push-back-v3 | 0.3 | 0.5 |

**very_hard** (`la_ft` 0.800 / `base_ft` 0.440)

| Task | la_ft | base_ft | Task | la_ft | base_ft |
|------|-------|---------|------|-------|---------|
| nut-disassemble-v3 | 0.8 | 0.5 | stick-pull-v3 | 0.7 | 0.3 |
| pick-place-wall-v3 | 0.9 | 0.7 | shelf-place-v3 | 0.6 | 0.3 |
| stick-push-v3 | 1.0 | 0.4 | | | |

> Reproduction notes: keep the client-side image preprocessing unchanged
> (ROT180 + center-crop 2/3 + resize to 224×224 — already built into
> `eval_metaworld.py`); the policy server must be started with the matching
> `dataset_statistics.json` from the checkpoint directory.
