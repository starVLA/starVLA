# 🚀 DOMINO Training and Evaluation

> DOMINO is a dynamic manipulation benchmark where the robot must react to moving objects and time-varying scene changes across 35 tasks.

This document shows how to train and evaluate StarVLA on the [DOMINO benchmark](https://github.com/H-EmbodVis/DOMINO).

DOMINO reports two primary evaluation metrics: **Success Rate (SR)** and **Manipulation Score (MS)**, which measure task completion and dynamic-manipulation quality under penalty events.

---

## 📦 1. Environment Setup

Evaluation uses two environments:

- `starvla`: runs training and the policy server
- `domino`: runs the DOMINO simulator and benchmark

### starVLA environment

Follow the repository root [README.md](../../README.md) to install starVLA and prepare the training environment.

### DOMINO environment

Follow the upstream [DOMINO repository](https://github.com/H-EmbodVis/DOMINO) for simulator installation, assets, and benchmark-side setup.

Then install the bridge dependencies inside the DOMINO environment:

```bash
conda activate domino
pip install -r examples/DOMINO/eval_files/requirements.txt
```

Before running evaluation, set:

```bash
export DOMINO_PATH=/path/to/DOMINO
```

<details>
<summary><b>Optional environment overrides (Click to expand)</b></summary>

If your conda environment names are different from the launcher defaults, also set:

```bash
export DOMINO_STARVLA_ENV=your_starvla_env_name
export DOMINO_ENV=your_domino_env_name
```

Or provide explicit Python executables:

```bash
export STARVLA_PYTHON=/path/to/starvla/python
export DOMINO_PYTHON=/path/to/domino/python
```

</details>

---

## 🗂️ 2. Data Preparation

Prepare the converted dataset under `playground/Datasets/DOMINO/`:

```text
playground/Datasets/DOMINO/
├── Clean_Dynamic/<task>/...
├── Random_Dynamic/<task>/...
├── Clean/<task>/...        # optional, only for domino_cotrain
└── Randomized/<task>/...   # optional, only for domino_cotrain
```

Copy [modality.json](./train_files/modality.json) into the `meta/` directory of each dataset that will be used for training.

Supported `data_mix` options:

- `domino_clean_dynamic`: train with `Clean_Dynamic`
- `domino_random_dynamic`: train with `Random_Dynamic`
- `domino`: train with both dynamic splits
- `domino_cotrain`: train with DOMINO dynamic data plus static `Clean` and `Randomized`

Pick the `data_mix` that matches your experiment before launching training.

---

## 🚀 3. Training

Edit the user-facing settings in [run_domino_train.sh](./train_files/run_domino_train.sh), especially:

- `base_vlm`
- `run_root_dir`
- `data_mix`
- `run_id`
- `wandb_entity`

Then start training from the repository root:

```bash
bash examples/DOMINO/train_files/run_domino_train.sh
```

The training config used by the launcher is [starvla_train_domino.yaml](./train_files/starvla_train_domino.yaml).

---

## 🎯 4. Evaluation

Evaluation runs two processes in parallel:

- starVLA policy server in the `starvla` environment
- DOMINO benchmark in the `domino` environment

### Option A: Launcher script

This is the recommended entrypoint for full-benchmark runs.

Evaluate all 35 DOMINO tasks:

```bash
bash examples/DOMINO/eval_files/start_eval.sh \
    -m demo_clean_dynamic \
    -n my_run \
    -c /path/to/checkpoint.pt \
    all
```

Evaluate a subset of tasks:

```bash
bash examples/DOMINO/eval_files/start_eval.sh \
    -m demo_random_dynamic \
    -n my_run \
    -c /path/to/checkpoint.pt \
    adjust_bottle grab_roller
```

<details>
<summary><b>Launcher flags (Click to expand)</b></summary>

Common flags:

- `-m, --mode`: `demo_clean_dynamic` or `demo_random_dynamic`
- `-n, --name`: run name written into DOMINO result folders and local logs
- `-c, --ckpt`: checkpoint path
- `-j, --jobs-per-gpu`: number of concurrent tasks per GPU
- `-p, --base-port`: first policy-server port
- `--install-deps`: install eval-side Python dependencies once before launching

The task arguments can be:

- `all` for the full 35-task benchmark
- one or more task names
- a task-list file with one task per line

</details>

### Option B: Run server and benchmark manually

Use this mode when debugging a single task or the bridge itself.

Terminal 1, in the `starvla` environment:

```bash
bash examples/DOMINO/eval_files/run_policy_server.sh /path/to/checkpoint.pt 0 5694
```

Terminal 2, in the `domino` environment:

```bash
bash examples/DOMINO/eval_files/eval.sh \
    adjust_bottle demo_clean_dynamic my_run 0 0 /path/to/checkpoint.pt 5694
```


### Metrics

DOMINO uses the benchmark's native metric tracker in `DOMINO/script/eval_metrics.py`. The two primary metrics are:

- `Success Rate (SR)`: percentage of episodes that satisfy the benchmark success condition
- `Manipulation Score (MS)`: `route_completion × total_penalty_factor`, reflecting both progress and dynamic penalties

<details>
<summary><b>Metric details (Click to expand)</b></summary>

`route_completion` measures progress toward the dynamic target on a `0` to `100` scale. Successful episodes receive `100`, while failed episodes can still receive partial credit if the robot makes meaningful progress.

`total_penalty_factor` starts from `1.0` and is reduced by penalty events such as clutter collisions or target out-of-bounds. As a result, `MS` rewards progress while penalizing unstable or unsafe behavior during dynamic interaction.

For dynamic tasks, DOMINO also uses stricter success checks than a simple end-state test. The benchmark-side evaluation includes rules such as out-of-bounds failure detection and lifting verification, which help avoid false positives from accidental contact.

When you run evaluation through `start_eval.sh` or `eval.sh`, DOMINO summarizes:

- success rate over all episodes
- mean and standard deviation of manipulation score
- mean and standard deviation of route completion
- counts of penalty events such as clutter collisions and out-of-bounds

</details>


---

## 🕒 5. Historical Context Interface

Historical information is important for DOMINO because the benchmark focuses on dynamic manipulation rather than static single-frame understanding.

The starVLA-DOMINO bridge already exposes a generic history interface through [deploy_policy.yml](./eval_files/deploy_policy.yml) and [model2robotwin_interface.py](./eval_files/model2robotwin_interface.py).

You can enable and configure history in [deploy_policy.yml](./eval_files/deploy_policy.yml):

```yaml
history_k: 4
history_stride: 1
history_mode: "flow"   # "flow", "frames", or "none"
```

The current interface supports:

- `history_mode: "flow"`: build optical-flow RGB history from past frames
- `history_mode: "frames"`: pass raw historical RGB frames
- `history_mode: "none"`: disable temporal context and use the current frame only

At evaluation time, the bridge packages the temporal representation into `history_images` and forwards it through the WebSocket interface. This keeps the transport layer generic, so users can decide how their policy consumes historical observations.

<details>
<summary><b>Customize the history representation (Click to expand)</b></summary>

If you want a different temporal representation, subclass `ModelClient` or modify `_build_history_context()` in [model2robotwin_interface.py](./eval_files/model2robotwin_interface.py). For example, you can replace the provided flow/frame implementation with:

- multi-view history fusion
- object-centric motion crops
- learned temporal tokens
- any custom spatiotemporal preprocessing pipeline

In other words, the README keeps the history configuration visible for DOMINO users, while the code keeps the policy side abstract enough for custom dynamic-aware models.

</details>
