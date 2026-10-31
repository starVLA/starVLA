# RLBench: Robot Learning Benchmark [![Unit Tests](https://github.com/stepjam/RLBench/workflows/Unit%20Tests/badge.svg)](https://github.com/stepjam/RLBench/actions) [![Task Tests](https://github.com/stepjam/RLBench/workflows/Task%20Tests/badge.svg)](https://github.com/stepjam/RLBench/actions) [![Discord](https://img.shields.io/discord/694945190867370155.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/DXPCjmd)

![RLBench task grid](readme_files/task_grid.png)

**RLBench** is an ambitious large-scale benchmark and learning environment 
designed to facilitate research in a number of vision-guided manipulation
research areas, including: reinforcement learning, imitation learning,
multi-task learning, geometric computer vision, and in particular, 
few-shot learning. [Click here for website and paper.](https://sites.google.com/corp/view/rlbench)

# Integration

This directory integrates RLBench as a native-action benchmark for StarVLA.
RLBench owns action semantics: StarVLA consumes images and language, emits a
raw vector of the configured size, and the adapter validates that vector
against `ActionMode.action_shape(scene)` before passing it unchanged to
`TaskEnvironment.step()`.

This README documents the StarVLA integration workflow. It is not a replacement
for the [official RLBench documentation](https://github.com/stepjam/RLBench),
which also covers reinforcement learning, Gym, domain randomization, task
building, and other general-purpose workflows.

## Official Task Sets and This Evaluation

RLBench publishes task sets for standardized few-shot and multi-task learning.
For example, `FS10_V1` contains 10 training tasks and 5 held-out test tasks;
the official multi-task sets include `MT15_V1`, `MT30_V1`, `MT55_V1`, and
`MT100_V1`. These sets and their train/test splits should be identified by
their RLBench version when reporting results.

The nine-task protocol used in this directory selects nine tasks from the
official RLBench task catalog. It is a StarVLA evaluation protocol, not one of
the official few-shot or multi-task split objects. It uses
`push_buttons`, `stack_blocks`, `open_drawer`, `close_jar`, `stack_cups`,
`sweep_to_dustpan`, `slide_block_to_target`, `screw_nail`, and
`insert_onto_square_peg`. Report the exact task list, number of demonstrations,
rollouts per task, maximum episode length, and success criterion with every
result.

## Experimental Results

We use the established nine-task multi-task protocol below. The task labels
map to the RLBench task files as follows: Sweep Dirt -> `sweep_to_dustpan`,
Screw Bulb -> `screw_nail`, and Put in Board -> `insert_onto_square_peg`.

| Model | Action Mode | Steps | Epochs | Push Buttons | Stack Blocks | Open Drawer | Close Jar | Stack Cups | Sweep Dirt | Slide Block | Screw Bulb | Put in Board | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLARVA | Published RLBench protocol | - | - | 56 | 0 | 60 | 28 | 0 | 84 | 100 | 8 | 0 | 37.3 |
| PerAct | Published RLBench protocol | - | - | 48 | 36 | 80 | 60 | 0 | 56 | 72 | 24 | 16 | 43.6 |
| StarVLA-OFT | JointVelocity + Discrete | running | - | - | - | - | - | - | - | - | - | - | - |

The LLARVA and PerAct rows are published baselines copied from the classic
RLBench multi-task results referenced in [RLBench Multi-Task success rate (%)](https://openreview.net/pdf/43263302a3a8d0de5a63e6f77b7cd0e0eb6986ca.pdf). They are not StarVLA measurements. The StarVLA row remains blank until the currently running native-action experiment has completed evaluation.
The published rows do not by themselves specify the rollout count or every evaluation detail, so those values must be
verified from the cited protocol before claiming a strict comparison.

The action modes are explicitly listed in the table because `JointVelocity`, `JointPosition`, `JointTorque`, `EndEffectorPoseViaIK`, and `EndEffectorPoseViaPlanning` have different physical meanings; that is, even if their action vectors have the same dimensionality, the individual dimensions represent different quantities.

## 1. Environment Setup

Use a dedicated Python 3.10 environment for the benchmark. RLBench 1.2.0 and
CoppeliaSim 4.1.0 are the versions exercised by this integration.

```bash
conda create -n starvla-rlbench python=3.10 -y
conda activate starvla-rlbench
cd /path/to/starVLA
pip install -e .
pip install git+https://github.com/stepjam/RLBench.git

export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```

<details>
<summary><b>⚠️ Common Issues</b></summary>

- **Headless server has no display:** Run CoppeliaSim through `xvfb-run`, for
  example with `xvfb-run -a -s "-screen 0 1280x1024x24" ...`.
- **A job runs inside an old tmux session:** Do not reuse a stale `DISPLAY`
  value copied into that session. Start the command with a fresh Xvfb display
  (using `xvfb-run`) so the GUI and OpenGL environment are created together.
- **CoppeliaSim 4.1.0 vision sensors fail to initialize:** Do not set
  `QT_QPA_PLATFORM=offscreen`. Vision sensors require the OpenGL context
  provided by Xvfb; the offscreen Qt platform cannot create the required
  context.
- **OpenGL or camera failures are unclear:** Run the verification commands
  below. `glxinfo -B` should report a working renderer before running RLBench.

</details>

The following commands are useful smoke checks in the `starvla` environment:

```bash
xvfb-run -a -s "-screen 0 1280x1024x24" glxinfo -B
python -m unittest tests.test_rlbench_adapter -v
python examples/simBenchmarks/RLbench/eval_files/resolve_action_dim.py \
  --config examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml
```

The last command emits `RLBENCH_ACTION_DIM=8` for the default Panda
JointVelocity + Discrete configuration.

## 2. Action Mode Configuration

The selected RLBench mode determines the model action dimension at runtime.
The included [joint-velocity configuration](configs/env/joint_velocity.yaml) is:

```yaml
env:
  name: rlbench
  camera_names: [front_rgb]
  action_mode:
    type: MoveArmThenGripper
    arm: {type: JointVelocity, kwargs: {}}
    gripper: {type: Discrete, kwargs: {}}
```

Supported arm modes are `JointVelocity`, `JointPosition`, `JointTorque`,
`EndEffectorPoseViaIK`, and `EndEffectorPoseViaPlanning`. `Discrete` and
`Continuous` grippers are used when exposed by the installed RLBench version.
RLBench 1.2.0 exposes `Discrete` but not `Continuous`; a request for an
unavailable native mode fails explicitly.

For every run, `Environment.action_shape()` is the action-head contract.
The 8D result described here is specific to the current Panda configurations:
seven arm values plus a scalar gripper. It is not a universal RLBench action
dimension. The training dataset, dataset statistics, checkpoint, and evaluator
must use the exact same action-mode configuration and action semantics.

## 3. Checkpoint Download

Download a vision-language base model before training. For the default OFT
recipe, use Qwen3-VL-2B-Instruct:

```bash
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct \
  --local-dir playground/Pretrained_models/Qwen3-VL-2B-Instruct
```

## 4. Native-Action Data Preparation

Collect RLBench expert demonstrations with the same mode that will be used by
training and evaluation. The official generator currently uses
`JointVelocity + Discrete`, matching the default configuration:

```bash
export RLBENCH_ROOT=/path/to/RLBench
export RAW_ROOT=playground/Datasets/RLBench_raw/joint_velocity_9task

xvfb-run -a -s "-screen 0 1280x1024x24" \
  python "$RLBENCH_ROOT/rlbench/dataset_generator.py" \
  --save_path "$RAW_ROOT" \
  --tasks push_buttons stack_blocks open_drawer close_jar stack_cups \
          sweep_to_dustpan slide_block_to_target screw_nail insert_onto_square_peg \
  --image_size 128 128 --renderer opengl --processes 1 \
  --episodes_per_task 100 --variations 1
```

The raw generator output is not directly a StarVLA training dataset. Convert
it to LeRobot v2 only with a converter that writes the action labels for this
configured native mode and records that mode alongside the data. In particular,
never relabel all trajectories as `[xyz, rpy, gripper]`:

| RLBench arm mode | Stored native arm target | Full Panda action |
| --- | --- | --- |
| JointVelocity | 7 joint velocities | 7 joint velocities + gripper |
| JointPosition | 7 joint positions | 7 joint positions + gripper |
| JointTorque | 7 joint torques | 7 joint torques + gripper |
| EndEffectorPoseViaIK | xyz + quaternion | xyz + quaternion + gripper |
| EndEffectorPoseViaPlanning | xyz + quaternion | xyz + quaternion + gripper |

Before training, validate the converted dataset has a single action width
equal to `RLBENCH_ACTION_DIM`, its `meta/modality.json` action names describe
that native representation, and its normalization statistics were computed
from those same fields. Keep raw data under `playground/Datasets/RLBench_raw/`
until this validation passes.

The repository converter preserves every RGB camera as raw `uint8` PNG
frames, concatenates fixed-size robot proprioception fields into
`observation.state`, and keeps task-specific `task_low_dim_state` vectors in
their own column. It writes the native ActionSpec to
`meta/action_spec.json` and `meta/info.json`:

```bash
python -m pip install pyarrow
python examples/simBenchmarks/RLbench/eval_files/resolve_action_dim.py \
  --config examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml

python -m examples.simBenchmarks.RLbench.conversion.rlbench_to_lerobot \
  --raw-root playground/Datasets/RLBench_raw/joint_velocity_9task \
  --output-root playground/Datasets/RLBench_lerobot/joint_velocity_9task \
  --config examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml \
  --environment-action-shape 8 \
  --framework-action-dim 8 \
  --fps 20
```

The fps value is the effective rate of observations present in
low_dim_obs.pkl, not the CoppeliaSim timestep. With RLBench's default
record_gripper_closing=False, intermediate gripper-closing simulation steps
may be absent from the raw observation sequence and cannot be reconstructed by
the converter.
The supplied `joint_position.yaml` and `ee_pose_ik.yaml` select the corresponding modes.

The data flow has three distinct layers:

1. **RLBench raw demonstrations:** images, native actions, observations, and
   task metadata produced by the RLBench generator.
2. **Converted LeRobot data:** Parquet rows, PNG paths, modality metadata,
   action specifications, and task records written by this converter.
3. **StarVLA training input:** the fields selected by the StarVLA dataset
   loader and the active training/data configuration.

The converter combines all descriptions for an episode into one language
string, separated by ` | `. That string is written to the Parquet `task` field
and to `meta/tasks.jsonl`; `task_index` points to that entry in `tasks.jsonl`.
The loader uses the index to provide one string as the VLA language input.

### Observation and Dataset Gotchas

RLBench uses `128x128` image observations by default. Keep the image size used
for demonstration collection consistent with the live evaluation observation
size and the training preprocessing. Changing `ObservationConfig` dimensions
can change the saved demonstration images; re-collect demonstrations when the
target live resolution changes instead of assuming that an offline resize is
equivalent to a live camera observation.

`observation.state` contains the fixed robot proprioception fields selected by
the converter. `observation.task_low_dim_state` is a separate task-specific
field. Saving it does not make it a StarVLA policy input, and it must not be
treated as a general-purpose robot state without checking the active data
configuration. In the RLBench training configuration,
`language_keys` selects `annotation.human.action.task_description`. The loader
reads the row's `task_index`, looks up that index in `meta/tasks.jsonl`, and
returns the corresponding single `task` string as the VLA language input.
`task_low_dim_state` is not listed in the configured state keys, so the current
training path does not automatically pass it to the policy. RLBench also warns
that low-dimensional task observations can become out of distribution when
objects leave the workspace; image-based observations are the safer default for
this benchmark.

## 5. Training

The launcher reads these variables before resolving the native action
dimension. Set them explicitly when running from another checkout:

| Variable | Meaning | Default |
| --- | --- | --- |
| `STARVLA_DIR` | StarVLA repository root | Repository root inferred from the launcher |
| `RLBENCH_ENV_CONFIG` | RLBench ActionMode YAML | `$STARVLA_DIR/examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml` |
| `TRAIN_CONFIG` | StarVLA training YAML | Required; no implicit training recipe |
| `ACCELERATE_CONFIG` | Hugging Face Accelerate launch config | `$STARVLA_DIR/starVLA/config/deepseeds/deepspeed_zero2.yaml` |
| `TRAINING_SCRIPT` | Compatible VLA training entry point | `$STARVLA_DIR/starVLA/training/train_starvla.py` |

The repository already provides these compatible VLA combinations:

| Purpose | Accelerate config | Training script |
| --- | --- | --- |
| Default VLA/OFT | `deepspeed_zero2.yaml` | `train_starvla.py` |
| ZeRO-3 VLA | `deepspeed_zero3.yaml` | `train_starvla.py` |
| Alternate ZeRO-2 preset | `zero2.yaml` | `train_starvla.py` |
| VLA + multimodal co-training | Any compatible preset | `train_starvla_cotrain.py` |

Use either environment variables or launcher options for the last two:

```bash
export STARVLA_DIR="$(pwd)"
export RLBENCH_ENV_CONFIG=$STARVLA_DIR/examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml
export TRAIN_CONFIG=$STARVLA_DIR/examples/simBenchmarks/RLbench/configs/train/rlbench_starvla_training.yaml
export ACCELERATE_CONFIG=$STARVLA_DIR/starVLA/config/deepseeds/deepspeed_zero3.yaml
export TRAINING_SCRIPT=$STARVLA_DIR/starVLA/training/train_starvla.py
```

The example TRAIN_CONFIG path above is a placeholder and must point to an
existing YAML with the RLBench LeRobot dataset and the matching native action
schema. The launcher validates all four paths before starting training.

The launcher starts CoppeliaSim briefly, resolves `Environment.action_shape()`,
and injects the result into the StarVLA action head. It never hard-codes 7.
After placing a validated, native-action LeRobot dataset at `DATA_ROOT`, run:

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim

bash examples/simBenchmarks/RLbench/train_files/run_rlbench_train.sh \
  --accelerate-config "$ACCELERATE_CONFIG" \
  --training-script "$TRAINING_SCRIPT" \
  --datasets.vla_data.data_root_dir "$DATA_ROOT" \
  --datasets.vla_data.data_mix rlbench_joint_velocity_9task
```

The training YAML and data registry must define the same eight native action
keys. Changing `RLBENCH_ENV_CONFIG` selects another ActionMode and recomputes
the output dimension; it also requires a dataset and statistics generated for
that ActionMode.

## 6. Evaluation

Start the StarVLA policy server from the 8D RLBench checkpoint in one terminal:

```bash
python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/rlbench_checkpoint.pt --port 10095 --use_bf16
```

Then evaluate in the RLBench environment:

```bash
export COPPELIASIM_ROOT=/path/to/CoppeliaSim
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$COPPELIASIM_ROOT"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

xvfb-run -a -s "-screen 0 1280x1024x24" \
  python examples/simBenchmarks/RLbench/eval_files/eval_rlbench.py \
  --config examples/simBenchmarks/RLbench/configs/env/joint_velocity.yaml \
  --task PushButtons --episodes 50 --max-steps 40 --headless
```

You can use this approach to repeat the command for all nine tasks, record the number of successful rollouts and total rollouts, and then fill the StarVLA row with only the resulting percentage values.

## 7. Version Compatibility and Troubleshooting

This integration targets RLBench 1.2.0 with CoppeliaSim 4.1.0 and Python 3.10.
RLBench 1.2.0 has action-mode API changes and exposes the `Discrete` gripper
mode used by the default recipe; it does not expose the `Continuous` mode
assumed by some older examples. Check the installed RLBench version and
`Environment.action_shape()` before collecting data or launching training.

For general installation, headless rendering, task lists, custom tasks, arm
swapping, Gym, and domain randomization, use the
[official RLBench README](https://github.com/stepjam/RLBench/blob/master/README.md).

The StarVLA-specific checks above should still be run after following those
instructions.

## 8. Official Resources and Citation

- [RLBench website and paper](https://sites.google.com/corp/view/rlbench)
- [RLBench source repository](https://github.com/stepjam/RLBench)
- [RLBench official README](https://github.com/stepjam/RLBench/blob/master/README.md)

If RLBench is used in a paper, cite the benchmark as follows:

```bibtex
@article{james2019rlbench,
  title={RLBench: The Robot Learning Benchmark & Learning Environment},
  author={James, Stephen and Ma, Zicong and Rovick Arrojo, David and Davison, Andrew J.},
  journal={IEEE Robotics and Automation Letters},
  year={2020}
}
```