# Evo-DexBench

[Evo-DexBench](https://github.com/MINT-SJTU/Evo-DexBench) evaluates dexterous
manipulation with Franka Panda arms and Shadow Hands in ManiSkill3: 30 tasks in
six functional families, 15 single-arm (`m01`–`m15`) and 15 dual-arm
(`b01`–`b15`). This example trains StarVLA policies on the public LeRobot v3
data and evaluates them with the benchmark's `policy_api`.

Links: [paper](https://arxiv.org/abs/2610.21180) ·
[project page](https://mint-sjtu.github.io/Evo-DexBench.io) ·
[benchmark code](https://github.com/MINT-SJTU/Evo-DexBench) ·
[dataset](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench-dataset) ·
[checkpoints](https://huggingface.co/MINT-SJTU/Evo-DexBench_StarVLA_checkpoints)

```text
examples/simBenchmarks/EvoDexBench/
├── data_preparation.sh            # download the LeRobot data and add the modality sidecars
├── train_files/
│   ├── data_registry/data_config.py   # data configs and 15-task mixtures (auto-registered)
│   ├── modality_single.json / modality_dual.json
│   ├── {single,dual}15_{qwenoft_h50,wan_h8_mean,cosmos_h8_mean}_q99.yaml
│   └── run_evodex_train.sh
├── eval_files/
│   ├── starvla_policy/            # benchmark client plugin and model-side state adapter
│   ├── run_policy_server.sh
│   └── eval_evodex.sh
├── pyproject.toml                 # installs the client plugin into the simulator environment
└── JOINT15_RESULTS.md             # per-task and per-condition results
```

## Results

One checkpoint per model and embodiment, trained jointly on the 15 tasks of that
embodiment (624 demonstrations: 50 for each of 12 Core tasks and 8 for each of 3
Gen tasks). Each task is evaluated over 100 episodes; values are fractions.

| Checkpoint | SR, 15 tasks | Stage, 15 tasks | SR, 12 Core | SR, 3 Gen |
| --- | ---: | ---: | ---: | ---: |
| QwenOFT single 600k | 0.157 | 0.257 | 0.173 | 0.090 |
| Wan-OFT single 300k | 0.121 | 0.198 | 0.146 | 0.020 |
| Cosmos-GR00T single 300k | 0.129 | 0.241 | 0.151 | 0.043 |
| QwenOFT dual 600k | 0.018 | 0.109 | 0.023 | 0.000 |
| Wan-OFT dual 300k | 0.005 | 0.099 | 0.006 | 0.003 |
| Cosmos-GR00T dual 300k | 0.037 | 0.168 | 0.044 | 0.010 |

Per-task, per-condition and M01–M04 results, the training settings and the
checkpoint hashes are in [JOINT15_RESULTS.md](JOINT15_RESULTS.md).

## 1. Environment

Use two environments: a model environment for StarVLA and a simulator
environment for Evo-DexBench. The model server and the simulator client talk
over a WebSocket (WebSockets 15.x on both sides).

**Model environment.** Install StarVLA's dependencies. QwenOFT uses Qwen3-VL
(Transformers 4.57.0). The released Wan-OFT and Cosmos-GR00T checkpoints were
trained and evaluated with Python 3.11, Torch 2.7.1, Diffusers 0.35.1,
Transformers 4.53.2 and Accelerate 1.10.1, plus the model extension described
[below](#wan-oft-and-cosmos-gr00t), which requires Python 3.11 or later.

**Simulator environment.** Follow the benchmark's
[install and asset steps](https://github.com/MINT-SJTU/Evo-DexBench#-quick-start),
then install the client plugin from this directory. The asset download needs
access to the gated
[PartNet-Mobility repository](https://huggingface.co/datasets/sapien-sim/PartNetMobility):
accept its terms and log in to Hugging Face first.

```bash
# in an Evo-DexBench checkout
tools/setup_environment.sh --profile sim --name evodex-sim
conda activate evodex-sim
hf auth login
dexbench assets download --asset-dir "$PWD/assets"
source assets/activate_evodex_assets.sh

# from the StarVLA repository root
python -m pip install -e examples/simBenchmarks/EvoDexBench
```

## 2. Data

```bash
bash examples/simBenchmarks/EvoDexBench/data_preparation.sh
```

The script downloads the task-wise LeRobot v3 export from
[MINT-SJTU/Evo-DexBench-dataset](https://huggingface.co/datasets/MINT-SJTU/Evo-DexBench-dataset)
into `playground/Datasets/EvoDexBench/raw` and writes one training copy per task
to `playground/Datasets/EvoDexBench/{single,dual}/<task>`: videos are hard-linked,
the depth columns are dropped from the parquet data (the recipes do not use depth,
and the Array2D depth features trigger a pandas extension-dtype error in
`datasets` 3.x), and the matching modality sidecar is added as
`meta/modality.json`. The sidecars map the released 111D/222D state to the
31D/62D policy state (end-effector pose and hand joint positions per arm), keep
the 30D/60D physical action, and select the three RGB cameras. Set `DATA_ROOT`
to use another location; the recipes read `playground/Datasets/EvoDexBench/{single,dual}`,
so then also pass `--datasets.vla_data.data_root_dir "$DATA_ROOT/single"` (or
`/dual`) to the trainer. Set `TASKS` (for example `"m01 m02"`) to prepare a
subset; the 15-task recipes need all 15 tasks of their embodiment.

## 3. Training

```bash
FRAMEWORK=QwenOFT EMBODIMENT=single \
  bash examples/simBenchmarks/EvoDexBench/train_files/run_evodex_train.sh \
  --framework.qwenvl.base_vlm /path/to/Qwen3-VL-4B-Instruct
```

All recipes sample the 15 tasks with equal weight, use global batch 4, seed 42,
bf16, 5,000 warmup steps, cosine decay to 5e-7 and gradient clipping of 1.0,
and freeze the backbone.

| Recipe | Framework | Steps | Horizon / executed | GPUs x batch |
| --- | --- | ---: | --- | --- |
| `{single,dual}15_qwenoft_h50_q99.yaml` | `QwenOFT` | 600k | 50 / 10 | 2 x 2 |
| `{single,dual}15_wan_h8_mean_q99.yaml` | `EvoWanOFT` | 300k | 8 / 8 | 1 x 4 |
| `{single,dual}15_cosmos_h8_mean_q99.yaml` | `EvoCosmosGR00T` | 300k | 8 / 8 | 1 x 4 |

Use `EMBODIMENT=dual` for the dual-arm recipes and `FRAMEWORK=EvoWanOFT` or
`FRAMEWORK=EvoCosmosGR00T` for the extension models; for those, set the
official initialization checkpoint and base configuration paths in the recipe
first. Pass `--trainer.max_train_steps 300000` to reproduce the QwenOFT 300k rows
in JOINT15_RESULTS.md. Choose a new `run_id` for every fresh run. Set
`WANDB_MODE=disabled` to train without Weights & Biases.

## 4. Evaluation

Download a released checkpoint. QwenOFT also needs its original Qwen3-VL base;
Wan-OFT and Cosmos-GR00T need `model_extension/` and `base_configs/`:

```bash
hf download MINT-SJTU/Evo-DexBench_StarVLA_checkpoints --include "qwenoft-single/*" \
  --local-dir ./playground/Checkpoints/evodex
hf download Qwen/Qwen3-VL-4B-Instruct --revision ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct
```

Start the model server in the model environment (from the StarVLA root):

```bash
CKPT_ROOT=./playground/Checkpoints/evodex CASE=qwenoft-single \
QWEN_BASE=./playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  bash examples/simBenchmarks/EvoDexBench/eval_files/run_policy_server.sh
```

Then run the 15 tasks in the simulator environment:

```bash
CASE=qwenoft-single PORT=10101 \
  bash examples/simBenchmarks/EvoDexBench/eval_files/eval_evodex.sh
```

Logs go to `results/<case>/<task>.jsonl`. Re-running the script skips tasks
whose log is complete, so it can resume after an interruption.

Each task runs the standard protocol: 100 episodes in five conditions of 20
(Nominal, Camera, Lighting, Physical, Spatial), layouts 34/33/33, seeds
1001–1005, four parallel environments, 30 Hz control, 120 Hz simulation and the
step limit in the task YAML. For a short diagnostic, call `dexbench evaluate`
with `--episodes 1 --num-envs 1 --max-steps 3` and a new log path.

| Item | Contract |
| --- | --- |
| Control | 30 Hz; chunk frames executed in order, without increment scaling |
| Cameras, single arm | context, interaction, wrist_right |
| Cameras, dual arm | context, wrist_left, wrist_right |
| Images | uint8 RGB, 224 × 224 per view |
| State | physical 31D single or 62D dual |
| Action | end-effector delta and absolute hand joint targets, 30D single or 60D dual |
| Normalization | the checkpoint's saved q99 statistics |

The client checks the server's mixture, horizon, image size and state/action
key order, and keeps the benchmark's physical-action guard. The model-side
adapter in `starvla_policy/server.py` applies the saved state transform once;
StarVLA's policy server de-normalizes the actions. `run_policy_server.sh` serves
the 15-task checkpoints under the single-dataset mixture name of the same robot
type (`evodex_{single,dual}_h50_q99` or `_h8_q99`), which the client expects;
the saved statistics are used unchanged.

## Wan-OFT and Cosmos-GR00T

The Wan and Cosmos checkpoints use `EvoWanOFT` and `EvoCosmosGR00T`, not the
`WanOFT` and `CosmoPredict2GR00T` classes in `starVLA/model/framework/WM4A`.
They encode the three camera views as independent VAE latents (posterior mean)
and add Evo state/action projections, so loading them with the upstream
classes changes the model. The implementation is distributed with the
checkpoints as `model_extension/`; install it in the model environment:

```bash
for pattern in "wan-single/*" "model_extension/*" "base_configs/wan/*"; do
  hf download MINT-SJTU/Evo-DexBench_StarVLA_checkpoints --include "${pattern}" \
    --local-dir ./playground/Checkpoints/evodex
done
python -m pip install -e ./playground/Checkpoints/evodex/model_extension --no-deps
```

Use `cosmos-*` and `base_configs/Cosmos-Predict2-2B-Video2World/*` for Cosmos.
`run_policy_server.sh` and `run_evodex_train.sh` select the extension's server
and trainer from `CASE` and `FRAMEWORK`.

The extension checks that `STARVLA_ROOT` is a git checkout of StarVLA that
contains commit `4507931a625c844404c4a76128fb116536d8ca7c`, the revision it was
validated on; this repository's history satisfies that. A shallow clone does not
include it; run `git fetch --unshallow` first.
