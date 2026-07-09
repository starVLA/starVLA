# Step 1: Data Conversion

This step prepares G1 WholeBody demonstrations for StarVLA training on the
local PC.

For this setup, the collected data is already stored in LeRobot v2.1 layout.
Do not reconvert it. Register it under StarVLA, validate the metadata, then
pass the schema into `step2_training`.

## Local Working Layout

Run StarVLA commands from the repository root:

```bash
cd starVLA
```

If you also use GR00T-WholeBodyControl locally, keep it next to `starVLA` or
set `GROOT` to its location:

```bash
GROOT=../GR00T-WholeBodyControl
```

The current dataset used for this example is under the StarVLA repository:

```text
playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic
```

If Git ever needs a local safe-directory entry for a copied workspace, use:

```bash
git config --global --add safe.directory "$(pwd)"
```

## Register Data Under StarVLA

The local example dataset is already present in StarVLA, so no extra import
step is required on this PC.

If you later export another copy of `test_sonic`, replace the source path in
the commands below with the archive location you saved on disk.

The expected target structure is:

```text
playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic/
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.ego_view/episode_000000.mp4
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
```

## Source Dataset Schema

The checked `test_sonic` dataset reports:

```text
codebase_version: v2.1
fps: 50
total_episodes: 49
total_frames: 56919
video key: observation.images.ego_view
task: pick the toy on the table, and put it into the box.
```

The important tensor dimensions from a sampled parquet episode are:

```text
observation.state: 43
observation.eef_state: 14
action.wbc: 43
action.motion_token: 64
teleop.left_hand_joints: 7
teleop.right_hand_joints: 7
teleop.smpl_pose: 63
teleop.vr_3pt_position: 9
teleop.vr_3pt_orientation: 18
```

The `meta/modality.json` maps these fields into groups:

```text
state:
  observation.state:
    left_leg 0:6
    right_leg 6:12
    waist 12:15
    left_arm 15:22
    left_hand 22:29
    right_arm 29:36
    right_hand 36:43
  observation.eef_state:
    left_wrist_pos 0:3
    left_wrist_abs_quat 3:7
    right_wrist_pos 7:10
    right_wrist_abs_quat 10:14

action:
  action.motion_token: 64
  teleop.left_hand_joints: 7
  teleop.right_hand_joints: 7
  teleop.delta_heading: 1
  teleop.smpl_pose: 63
  teleop.vr_3pt_position: 9
  teleop.vr_3pt_orientation: 18

video:
  observation.images.ego_view

annotation:
  annotation.human.task_description from task_index/tasks.jsonl
```

For the first StarVLA training pass, use the explicit robot type:

```text
unitree_g1_sonic_dex3
```

Recommended first action registry:

```text
action.motion_token: 64
teleop.left_hand_joints: 7
teleop.right_hand_joints: 7
```

This is a 78D controller target. Keep `action.wbc` available for analysis, but
do not train both `action.wbc` and `action.motion_token` in the same first
adapter unless the downstream policy head is designed for that.

## Validate the Registered Dataset

Metadata check with local system Python:

```bash
cd starVLA
export DATASET=playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic

python3 - <<'PY'
import json
import os
from pathlib import Path

dataset = Path(os.environ["DATASET"])
info = json.loads((dataset / "meta/info.json").read_text())
modality = json.loads((dataset / "meta/modality.json").read_text())
tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
features = info.get("features", {})
video_keys = [
    key for key, value in features.items()
    if value.get("dtype") in ("video", "image")
]

print("dataset:", dataset)
print("codebase_version:", info.get("codebase_version"))
print("fps:", info.get("fps"))
print("episodes:", info.get("total_episodes"))
print("frames:", info.get("total_frames"))
print("video_keys:", video_keys)
print("feature_keys:", sorted(features.keys()))
print("state_groups:", sorted(modality.get("state", {}).keys()))
print("action_groups:", sorted(modality.get("action", {}).keys()))
print("task:", tasks[0]["task"] if tasks else None)
PY
```

Parquet shape check using the local GR00T environment:

```bash
cd starVLA
GROOT=../GR00T-WholeBodyControl
export DATASET=playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic

"${GROOT}/.venv_data_collection/bin/python" - <<'PY'
import os
from pathlib import Path
import pandas as pd

dataset = Path(os.environ["DATASET"])
parquet = next((dataset / "data/chunk-000").glob("episode_*.parquet"))
df = pd.read_parquet(parquet)

keys = [
    "observation.state",
    "observation.eef_state",
    "action.wbc",
    "action.motion_token",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
    "teleop.smpl_pose",
    "teleop.vr_3pt_position",
    "teleop.vr_3pt_orientation",
]

print("parquet:", parquet)
print("rows:", len(df))
for key in keys:
    value = df[key].iloc[0]
    print(key, len(value))
PY
```

Video presence check:

```bash
cd starVLA
DATASET=playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic

find "${DATASET}/videos" -type f -name '*.mp4' | head
```

## Optional GR00T Cleanup

The local GR00T repository provides the cleaner here:

```text
<GROOT_ROOT>/gear_sonic/scripts/process_dataset.py
```

For this PC, the `.venv_data_collection` environment already has the needed
packages (`av`, `pandas`, `numpy`, `tyro`, `pyarrow`).

To clean a dataset in place:

```bash
cd ../GR00T-WholeBodyControl
source .venv_data_collection/bin/activate

python gear_sonic/scripts/process_dataset.py \
  --dataset-path ../starVLA/playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic \
  --output-path ../starVLA/playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic_cleaned \
  --remove-discarded
```

After cleanup, point the StarVLA path at the cleaned directory:

```bash
cd starVLA

rm -rf playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic
mv playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic_cleaned \
  playground/Datasets/UnitreeG1_WholeBody/lerobot/test_sonic
```

## Conversion Strategy

Use the least invasive route:

1. If the dataset is already LeRobot v2.1, keep it and validate it.
2. If the dataset only needs filtering, use `process_dataset.py`.
3. If a future dataset is HDF5, JSON, PKL, rosbag, or vendor logs, convert it
   to the same LeRobot v2.1 boundary first.
4. Only write a custom converter when the source format or action semantics
   cannot be represented by the existing LeRobot fields.

## Handoff to Step 2

Move to [step2_training](../step2_training/README.md) when these values are final:

```text
data_root_dir:
  playground/Datasets/UnitreeG1_WholeBody/lerobot

data_mix:
  test_sonic

robot_type:
  unitree_g1_sonic_dex3

video_keys:
  observation.images.ego_view

state_keys:
  observation.state
  observation.eef_state

action_keys:
  action.motion_token
  teleop.left_hand_joints
  teleop.right_hand_joints

language_keys:
  annotation.human.task_description

normalization:
  continuous state/action keys: q99
```

Before full training:

- Episode count matches `meta/info.json`.
- `meta/modality.json` and parquet tensor shapes agree.
- Camera videos decode and match the episode count.
- The selected action keys match the deployment-side SONIC controller contract.
- Keep 6D-hand datasets and 7D-hand datasets in separate robot-type configs.
