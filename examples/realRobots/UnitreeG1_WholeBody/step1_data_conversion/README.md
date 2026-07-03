# Step 1: Data Conversion

This step prepares the GR00T-WholeBodyControl demonstrations for StarVLA training. In the primary example route, GR00T-WholeBodyControl already exports a LeRobot v2.1-style dataset, so this step is mostly validation, cleanup, and StarVLA schema registration.

StarVLA's preferred boundary is LeRobot v2.1 plus a clear modality schema. If the upstream collector already produces LeRobot v2.1, do not reconvert it unnecessarily. Instead, validate it and add the StarVLA registry files.

This means LeRobot G1, Unitree `xr_teleoperate`, Unitree `unitree_lerobot`, GR00T-WholeBodyControl, a custom Unitree logger, a rosbag pipeline, or another teleop stack can all be valid inputs. The requirement is not the source tool. The requirement is a consistent LeRobot dataset with documented semantics.

## Primary Input from GR00T-WholeBodyControl

GR00T-WholeBodyControl's VLA collection flow writes a dataset shaped like:

```text
outputs/<timestamp>-G1-<robot_id>/
  data/
    train-00000.parquet
  videos/
    observation.images.<camera_name>/
      episode_000000.mp4
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
```

Copy or symlink it into:

```text
playground/Datasets/UnitreeG1_WholeBody/lerobot/<task_name>/
```

## Post-Process the NVlabs Dataset

Before StarVLA training, run the upstream cleanup step if needed:

```bash
source .venv_data_collection/bin/activate
python gear_sonic/scripts/process_dataset.py \
  --dataset-path outputs/<timestamp>-G1-<robot_id> \
  --output-path outputs/<task_name>_cleaned
```

Use the cleaned dataset for StarVLA. This removes discarded demonstrations and stale motion frames according to the upstream workflow.

## Generic Input

One of:

```text
playground/Datasets/UnitreeG1_WholeBody/raw/<task_name>/
```

or:

```text
playground/Datasets/UnitreeG1_WholeBody/lerobot/<task_name>/
```

## Output

```text
playground/Datasets/UnitreeG1_WholeBody/lerobot/<task_name>/
  data/
  videos/
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
```

## Conversion Strategy

Use the least invasive converter possible:

1. If GR00T-WholeBodyControl already exported LeRobot v2.1, keep it and validate it.
2. If LeRobot G1 already produced a LeRobot dataset, keep it.
3. If Unitree `unitree_lerobot` can convert the collected JSON data, use it.
4. If data is HDF5 / JSON / PKL / rosbag / vendor logs, convert to LeRobot.
5. If another converter already works, such as an `any4lerobot` style pipeline, use it and document the command here.
6. Only write a custom converter when the source format or action semantics require it.

## Source-Specific Notes

### From LeRobot G1

Usually no structural conversion is needed. Validate the dataset and write StarVLA's `data_config.py` against the existing keys.

### From Unitree XR Teleoperate

The recorded data may need conversion into LeRobot. Preserve the XR action semantics, arm/end-effector type, camera source, and episode success metadata.

### From Unitree LeRobot

Use their converter when possible, then inspect the output. Their tooling is useful for Unitree JSON data, dataset visualization, episode editing, and conversion into LeRobot-compatible datasets.

### From GR00T-WholeBodyControl

Keep the LeRobot export and action contract. If using SONIC latent actions, document the latent/action groups explicitly in StarVLA's registry.

For the primary example, treat the action as:

```text
action.sonic_latent: 64
action.left_hand: 7
action.right_hand: 7
```

These are the dimensions described by the NVlabs VLA workflow. Verify them against the actual dataset before training.

## StarVLA Files to Add Later

This folder is only the data conversion stage, but it should prepare the information needed for:

```text
../step2_training/train_files/
  data_registry/data_config.py
  modality.json
  starvla_cotrain_g1_wholebody.yaml
```

Recommended robot type names for the primary example:

```text
unitree_g1_sonic
```

Use `unitree_g1_sonic` for the GR00T-WholeBodyControl / SONIC path. Use a different robot type if your data comes from another action representation.

## Required Schema Notes

Write these down before training:

```text
camera_order:
  - video.<camera_0>
  - video.<camera_1>

state_keys:
  - state.<group_name>: dim, unit, coordinate frame, order

action_keys:
  - action.<group_name>: dim, unit, control mode, frequency

language_keys:
  - annotation.human.action.task_description

normalization:
  state: q99
  action: q99 / binary / passthrough
```

Use `q99` as the StarVLA-side default for continuous state and action groups. This means the dataset statistics must include `q01` and `q99` for each continuous key. Use `binary` only for true binary commands, and use `passthrough` only when a controller-specific field should not be normalized.

## Validation Checklist

Run these checks before training:

- Episode count matches collection logs.
- Discarded / failed episodes are removed or tagged intentionally.
- All frames have valid timestamps.
- Camera videos decode correctly.
- Camera order matches the intended training order.
- State vector shape is constant.
- Action vector shape is constant.
- Continuous state/action statistics include `q01` and `q99`.
- `modality.json` key names match the actual LeRobot fields.
- A sampled episode can be replayed without loading a real robot.

## Handoff to Step 2

Move to [step2_training](../step2_training/README.md) when:

- the LeRobot dataset exists,
- `meta/modality.json` is present,
- the action/state dimensions are final for this first experiment,
- you know the `data_root_dir`, `data_mix`, and `robot_type` strings.
