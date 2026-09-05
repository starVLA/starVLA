# RoboCasa365 (PandaOmron)

This directory contains StarVLA training and evaluation recipes for the
official [RoboCasa365](https://robocasa.ai/) benchmark using the mobile
single-arm PandaOmron robot. It covers the 50 target tasks, four locally tested
StarVLA policy variants, and restartable multi-lane evaluation in target
kitchens.

> [!IMPORTANT]
> The NVIDIA GR1 tabletop fork under
> [`examples/simBenchmarks/Robocasa_tabletop`](../Robocasa_tabletop/README.md)
> is a different benchmark. Its checkpoints, observation/action contracts, and
> scores are not interchangeable with RoboCasa365.

## Status and headline results

The latest complete local result snapshot is **2026-08-30**. Every reported
variant below completed all **50 tasks × 50 rollouts = 2,500 rollouts** with
RoboCasa **v1.0.1** task-specific horizons in target kitchens.

- Best local overall: **StarVLA-PI at 150k — 40.64% (1,016/2,500)**.
- Best local Atomic-Seen: **StarVLA-GR00T at 120k — 76.67%**.
- Best local Composite-Seen: **StarVLA-PI at 150k — 39.38%**.
- Best local Composite-Unseen: **StarVLA-OFT at 220k — 15.12%**.

The aggregate and per-task scores for all four variants are included in
[Archived local research snapshot](#7-archived-local-research-snapshot).

## What is included

| Component | Checked-in support |
| --- | --- |
| Dataset | Official `target/human` LeRobot bundles: 18 atomic + 32 composite tasks |
| Robot | PandaOmron, single arm with mobile base |
| Training | Qwen3-VL-4B + `QwenOFT`; single-task and target-50 launchers |
| Evaluation | WebSocket policy server + upstream RoboCasa Gym environment |
| Minimal smoke test | OpenDrawer, 100 training steps, 2 evaluation episodes |

The repository also records an **archived local research snapshot** for four
StarVLA variants. Those results are retained below for reference, but the
corresponding checkpoints, complete YAML files, and raw evaluation logs are not
checked into this directory. They are not official RoboCasa leaderboard
submissions and cannot be reproduced from this README alone.

## 1. Environment setup

Training and simulation use separate environments:

- `starVLA`: training and WebSocket policy serving;
- `robocasa365`: RoboCasa, robosuite, MuJoCo, Gymnasium, and rendering.

Install the simulation environment using the official RoboCasa repositories:

```bash
conda create -n robocasa365 python=3.11 -y
conda activate robocasa365

mkdir -p playground/Code
cd playground/Code
git clone https://github.com/ARISE-Initiative/robosuite.git
git clone https://github.com/robocasa/robocasa.git robocasa365
pip install -e robosuite -e robocasa365
pip install lerobot mujoco

python -m robocasa.scripts.download_kitchen_assets
python robocasa365/robocasa/scripts/setup_macros.py
```

Set `DATASET_BASE_PATH` in
`playground/Code/robocasa365/robocasa/macros_private.py` to persistent storage,
for example `<starvla-root>/playground/Datasets/robocasa365`.

For headless rendering, set `MUJOCO_GL=egl`. On the tested robosuite stack,
`MUJOCO_EGL_DEVICE_ID` refers to the physical GPU ID; confirm that it matches
the device selected through `CUDA_VISIBLE_DEVICES` before launching a large
evaluation.

## 2. Dataset preparation

Download one task for the minimal walkthrough:

```bash
conda activate robocasa365
python -m robocasa.scripts.download_datasets \
  --tasks OpenDrawer \
  --split target \
  --source human
```

Download all 50 target/human bundles:

```bash
bash examples/simBenchmarks/Robocasa_365/train_files/download_target_human.sh
```

The expected dataset root is:

```text
playground/Datasets/robocasa365/
└── v1.0/target/
    ├── atomic/<Task>/<Date>/lerobot/
    └── composite/<Task>/<Date>/lerobot/
```

### Writable metadata

The StarVLA loader creates cache files such as `meta/stats_gr00t.json` and
`meta/steps_data_index.pkl`. Do not point it at a shared read-only dataset.
If the original download must remain read-only, create a writable overlay:

```text
<overlay>/v1.0/target/{atomic,composite}/<Task>/<Date>/lerobot/
├── data -> <downloaded-dataset>/data
├── videos -> <downloaded-dataset>/videos
└── meta/
    ├── info.json -> <downloaded-dataset>/meta/info.json
    ├── tasks.parquet -> <downloaded-dataset>/meta/tasks.parquet
    ├── episodes -> <downloaded-dataset>/meta/episodes
    ├── stats.json -> <downloaded-dataset>/meta/stats.json
    └── modality.json
```

Copy the official
`robocasa/models/assets/groot_dataset_assets/PandaOmron_modality.json` to each
writable `meta/modality.json`.

## 3. Observation and action contract

The loader defines the following PandaOmron contract:

| Modality | Shape/order |
| --- | --- |
| Images | left agent view, right agent view, eye-in-hand; 256×256 source images resized to 224×224 |
| State | 16-D: base position (3), base rotation (4), relative EEF position (3), relative EEF rotation (4), gripper qpos (2) |
| Model action | 12-D: EEF position (3), EEF rotation (3), gripper close (1), base motion (4), control mode (1) |
| Action horizon | 16 in the public QwenOFT example |
| Normalization | `min_max` per action key |

The state transform concatenates `sin(state)` and `cos(state)` when state is
enabled. The public YAML sets `include_state: false`.

> [!WARNING]
> The current training registry declares all three cameras, while the checked-in
> evaluation bridge sends only `video.robot0_agentview_left`. The bridge also
> constructs state unconditionally and does not validate camera count, state
> usage, or action horizon against server metadata. Align these contracts with
> the checkpoint before treating an evaluation result as valid.

## 4. Training

Before using either launcher, remove the placeholder `WANDB_API_KEY` assignment
or replace it with your own authenticated setup. Never commit a real key.

### Minimal OpenDrawer smoke test

The YAML defaults to 100 steps, but the shell launcher overrides it to 100,000.
Use explicit overrides for the short pipeline check:

```bash
conda activate starVLA
MAX_STEPS=100 SAVE_EVERY=100 NUM_GPUS=4 \
  bash examples/simBenchmarks/Robocasa_365/train_files/run_robocasa365.sh
```

The output should be written under:

```text
playground/Checkpoints/robocasa365_qwenoft_OpenDrawer/
```

The public configuration uses `QwenOFT` with a Qwen3-VL-4B backbone, a
12-dimensional MLP regression head, L1 action loss, horizon 16, and batch size
4 per GPU in YAML. The launcher overrides the batch size to 8 unless `BATCH` is
changed.

### Joint target-50 training

After all target/human bundles are available:

```bash
conda activate starVLA
NUM_GPUS=8 BATCH=8 MAX_STEPS=200000 \
  bash examples/simBenchmarks/Robocasa_365/train_files/run_robocasa365_all.sh
```

This command trains jointly on all 50 target/human tasks. Adjust GPU count,
per-device batch size, learning rate, and schedule for the actual cluster
instead of treating the example values as a published reference run.

## 5. Evaluation

Run the policy server and simulator client from the StarVLA repository root in
two terminals. Set the checkpoint path explicitly; the legacy default in
`run_eval.sh` may not match the output name produced by the current launcher.

Terminal 1 (`starVLA` environment):

```bash
conda activate starVLA
export CKPT=playground/Checkpoints/robocasa365_qwenoft_OpenDrawer/checkpoints/steps_100_pytorch_model.pt

python deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" \
  --port 5678 \
  --seed 7 \
  --use_bf16
```

Terminal 2 (`robocasa365` environment):

```bash
conda activate robocasa365
export CKPT=playground/Checkpoints/robocasa365_qwenoft_OpenDrawer/checkpoints/steps_100_pytorch_model.pt
export MUJOCO_GL=egl

python -m examples.simBenchmarks.Robocasa_365.eval_files.simulation_env \
  --args.pretrained-path "$CKPT" \
  --args.env-name robocasa/OpenDrawer \
  --args.port 5678 \
  --args.n-episodes 2 \
  --args.n-envs 1 \
  --args.max-episode-steps 500 \
  --args.n-action-steps 8
```

The client writes results next to the checkpoint:

```text
<checkpoint>.eval/robocasa_OpenDrawer.json
```

Videos default to `results/robocasa365_eval_test/videos/`.

### Current evaluation limitations

- `Args.seed` exists in `simulation_env.py`, but the current evaluator does not
  forward it to `gym.make()` or `env.reset()`. Setting the policy-server seed
  alone does not make RoboCasa environment rollouts seed-replayable.
- `run_eval.sh` still uses a legacy Python module path. The direct commands
  above use the current package path.
- The evaluator does not enforce checkpoint-bound camera, state, normalization,
  or horizon metadata. Verify them manually.
- Full 50-task evaluation requires task-specific episode horizons from the
  installed RoboCasa registry. Do not replace them with one global limit.

## 6. Benchmark protocol

The standard target evaluation contains 50 tasks:

| Group | Tasks | Meaning |
| --- | ---: | --- |
| Atomic-Seen | 18 | Atomic tasks represented in Human300 pretraining data |
| Composite-Seen | 16 | Composite tasks represented in Human300 pretraining data |
| Composite-Unseen | 16 | Composite tasks absent from Human300 pretraining data |

“Composite-Unseen” is defined relative to Human300. It is not automatically a
zero-shot split: a model trained on all target-50 demonstrations has seen those
16 target tasks. A seen-34 recipe holds them out.

RoboCasa v1.0.1 uses task-specific horizons. Record the installed RoboCasa
revision and registry-derived horizon together with every result. The public
benchmark reports per-task success over 50 rollouts; a 2-episode walkthrough is
only a pipeline smoke test.

## 7. Archived local research snapshot

The following snapshot is dated **2026-08-30**. Each selected result contains
50 tasks × 50 rollouts = 2,500 rollouts in target kitchens with RoboCasa v1.0.1.
The numbers are local research results, not leaderboard-verified results.

### Reported training/evaluation metadata

| Model | Training tasks | Initialization | GPUs × batch/GPU | Evaluated step | State | Model/execution horizon |
| --- | --- | --- | ---: | ---: | ---: | --- |
| StarVLA-OFT | target-50 | fresh Qwen3-VL | 8 × 24 | 220k | yes | 50 / 20 |
| StarVLA-OFT (no state) | target-50 | fresh Qwen3-VL | 8 × 26 | 140k | no | 16 / 16 |
| StarVLA-PI | seen-34 | fresh Qwen3-VL | 8 × 16 | 150k | yes | 50 / 16 |
| StarVLA-GR00T | seen-34 | fresh Qwen3-VL | 8 × 16 | 120k | no | 16 / 16 |

These four archived recipes are not all present in the current public files.
The state-enabled OFT run continues from a no-state/horizon-16 checkpoint, so
it is not a controlled state ablation.

### Selected aggregate results

| Model | Overall | Atomic-Seen | Composite-Seen | Composite-Unseen |
| --- | ---: | ---: | ---: | ---: |
| StarVLA-OFT | 33.32% (833/2500) | 65.22% (587/900) | 15.62% (125/800) | **15.12% (121/800)** |
| StarVLA-OFT (no state) | 30.92% (773/2500) | 63.78% (574/900) | 13.50% (108/800) | 11.38% (91/800) |
| **StarVLA-PI** | **40.64% (1016/2500)** | 76.56% (689/900) | **39.38% (315/800)** | 1.50% (12/800) |
| StarVLA-GR00T | 39.24% (981/2500) | **76.67% (690/900)** | 35.38% (283/800) | 1.00% (8/800) |

The reported evaluator accepted a `seed` argument but did not forward it to
the environment reset. These runs are therefore not paired or seed-replayable;
do not make paired statistical claims from their differences.

<details>
<summary>Detailed per-task results (50 rollouts per cell)</summary>

#### Atomic-Seen

| Task | StarVLA-OFT | StarVLA-OFT (no state) | StarVLA-PI | StarVLA-GR00T |
| --- | ---: | ---: | ---: | ---: |
| CloseBlenderLid | 16% | 12% | 30% | 36% |
| CloseFridge | 82% | 88% | 76% | 86% |
| CloseToasterOvenDoor | 80% | 86% | 86% | 76% |
| CoffeeSetupMug | 62% | 54% | 76% | 70% |
| NavigateKitchen | 82% | 74% | 82% | 44% |
| OpenCabinet | 88% | 84% | 86% | 86% |
| OpenDrawer | 86% | 80% | 92% | 86% |
| OpenStandMixerHead | 90% | 96% | 96% | 96% |
| PickPlaceCounterToCabinet | 74% | 54% | 56% | 80% |
| PickPlaceCounterToStove | 68% | 72% | 80% | 80% |
| PickPlaceDrawerToCounter | 38% | 58% | 80% | 60% |
| PickPlaceSinkToCounter | 58% | 58% | 82% | 84% |
| PickPlaceToasterToCounter | 70% | 66% | 90% | 96% |
| SlideDishwasherRack | 80% | 66% | 78% | 72% |
| TurnOffStove | 30% | 16% | 46% | 66% |
| TurnOnElectricKettle | 56% | 66% | 78% | 90% |
| TurnOnMicrowave | 66% | 62% | 80% | 88% |
| TurnOnSinkFaucet | 48% | 56% | 84% | 84% |

#### Composite-Seen

| Task | StarVLA-OFT | StarVLA-OFT (no state) | StarVLA-PI | StarVLA-GR00T |
| --- | ---: | ---: | ---: | ---: |
| DeliverStraw | 0% | 0% | 4% | 0% |
| GetToastedBread | 4% | 2% | 56% | 52% |
| KettleBoiling | 28% | 20% | 54% | 60% |
| LoadDishwasher | 30% | 24% | 64% | 42% |
| PackIdenticalLunches | 2% | 0% | 0% | 0% |
| PreSoakPan | 28% | 18% | 48% | 58% |
| PrepareCoffee | 2% | 0% | 28% | 28% |
| RinseSinkBasin | 28% | 42% | 76% | 80% |
| ScrubCuttingBoard | 8% | 2% | 54% | 18% |
| SearingMeat | 0% | 2% | 12% | 10% |
| SetUpCuttingStation | 10% | 20% | 30% | 40% |
| StackBowlsCabinet | 64% | 38% | 60% | 72% |
| SteamInMicrowave | 8% | 8% | 26% | 12% |
| StirVegetables | 0% | 2% | 10% | 24% |
| StoreLeftoversInBowl | 18% | 14% | 40% | 4% |
| WashLettuce | 20% | 24% | 68% | 66% |

#### Composite-Unseen

| Task | StarVLA-OFT | StarVLA-OFT (no state) | StarVLA-PI | StarVLA-GR00T |
| --- | ---: | ---: | ---: | ---: |
| ArrangeBreadBasket | 34% | 16% | 0% | 0% |
| ArrangeTea | 32% | 18% | 0% | 0% |
| BreadSelection | 16% | 10% | 0% | 0% |
| CategorizeCondiments | 6% | 2% | 0% | 0% |
| CuttingToolSelection | 42% | 34% | 20% | 6% |
| GarnishPancake | 0% | 2% | 0% | 0% |
| GatherTableware | 0% | 2% | 0% | 0% |
| HeatKebabSandwich | 0% | 0% | 0% | 0% |
| MakeIceLemonade | 0% | 0% | 0% | 0% |
| PanTransfer | 22% | 22% | 0% | 0% |
| PortionHotDogs | 0% | 2% | 0% | 0% |
| RecycleBottlesByType | 12% | 4% | 0% | 0% |
| SeparateFreezerRack | 0% | 0% | 0% | 0% |
| WaffleReheat | 42% | 40% | 0% | 0% |
| WashFruitColander | 6% | 16% | 4% | 10% |
| WeighIngredients | 30% | 14% | 0% | 0% |

</details>

## 8. Comparison guidance

Do not insert the archived StarVLA numbers into the official multi-task
leaderboard as ranks. The official leaderboard protocol uses different
pretraining data and evaluation kitchens, while the runs above use target-task
data and target kitchens.

### Illustrative comparison with the official leaderboard

The [official RoboCasa365 leaderboard](https://robocasa.ai/leaderboard.html)
was last updated on 2026-09-01 and lists 13 verified models. If the four local
StarVLA checkpoints were inserted simultaneously and sorted only by overall
success rate, their numerical positions would be:

| Illustrative position | Local checkpoint | Overall | Why this is not an official rank |
| ---: | --- | ---: | --- |
| 3 | **StarVLA-PI** | **40.64%** | seen-34 target-task training; target kitchens; not submitted |
| 6 | **StarVLA-GR00T** | **39.24%** | seen-34 target-task training; target kitchens; not submitted |
| 9 | **StarVLA-OFT** | **33.32%** | target-50 training; target kitchens; not submitted |
| 10 | **StarVLA-OFT (no state)** | **30.92%** | target-50 training; target kitchens; not submitted |

For context, the nearby verified leaderboard entries are Xiaomi-Robotics-1
(57.4%), ABot-M0.6 (46.6%), ABot-M0.5 (40.3%), PRTS (39.6%), RLDX-1
(36.0%), and WorldDreamer (35.3%). Protocol differences prevent an official
rank claim; this table is only an arithmetic placement against the current
published scores.

Useful public references include:

- the official
  [foundation-model learning protocol](https://github.com/robocasa/robocasa/blob/main/docs/benchmarking/foundation_model_learning.md);
- the [official RoboCasa365 leaderboard](https://robocasa.ai/leaderboard.html);
- the [AlphaBrain RoboCasa365 summary](https://github.com/AlphaBrainGroup/AlphaBrain/blob/main/benchmarks/Robocasa365/README.md).

Before comparing two results, align training data, policy count, task split,
checkpoint identity, camera/state contract, action and execution horizons,
environment version, task horizons, rollout count, and seed handling.

## 9. Reproducibility checklist

Record all of the following for every result:

- immutable checkpoint revision/hash and exact training step;
- initialization checkpoint and complete resolved training configuration;
- RoboCasa and robosuite versions;
- training scope (`target-50`, `seen-34`, or Human300) and task list;
- camera count/order and image preprocessing;
- state inclusion, key order, transform, and normalization;
- action key order, normalization, model horizon, and executed actions/query;
- registry-derived episode horizon for each task;
- exactly 50 valid rollouts per task and the per-task success vector;
- actual environment reset seeds, after seed forwarding is implemented;
- whether the result was submitted to and verified by the official leaderboard.

## References

- [RoboCasa365 project](https://robocasa.ai/)
- [RoboCasa code](https://github.com/robocasa/robocasa)
- [RoboCasa365 paper](https://arxiv.org/abs/2603.04356)
- [Official leaderboard repository](https://github.com/robocasa-benchmark/leaderboard)
- [Official dataset guide](https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md)
