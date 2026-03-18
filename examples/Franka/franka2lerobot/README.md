# Converting Franka Data to LeRobot Format

This document explains how to organize real Franka robot data into a LeRobot dataset that can be used by StarVLA.

Recommended workflow:

1. Raw episode data -> **LeRobot v3.0-style aggregated dataset**
2. If needed, convert the v3.0 dataset into a **LeRobot v2.1-style directory layout**

## 1. Environment Setup

### 1.1 Install Dependencies

First, downgrade `datasets`:

```bash
pip install "datasets<4.0.0"
```

Reason: `datasets==4.0.0` introduced changes such as `List` and `Column`, which are incompatible with the current conversion logic.

Then install LeRobot:

```bash
pip install lerobot
lerobot-info
```

If `lerobot-info` runs successfully, LeRobot is installed correctly.

### 1.2 Additional Dependencies

If you need to convert from v3.0 to the v2.1 directory layout, you will typically also need:

```bash
pip install jsonlines pyarrow tqdm numpy pandas
```

If the dataset contains videos, the second conversion step also depends on `ffmpeg` to split video files:

```bash
ffmpeg -version
```

If this command is not available, install `ffmpeg` first.

## 2. Raw Data Requirements

The raw data does not have to use one fixed file format, but it is recommended to organize it by **episode**. Each episode should include at least:

- a multi-frame observation sequence
- per-frame actions
- camera images or image paths
- robot state
- task text or a task identifier

For example, the raw data can be stored as `demo_*.pkl` files:

```bash
demo_*.pkl
```

The directory can look like this:

```text
your_franka_dataset/
├── demo_0.pkl
├── demo_1.pkl
├── demo_2.pkl
└── ...
```

In this case, each `pkl` file is usually treated as one episode.

If your data is not in `pkl` format, that is also fine. Common sources include:

- HDF5
- ROS bag
- JSON / NPZ / Parquet
- your own binary format

As long as you can reorganize the contents into the LeRobot target format described below, the original source format does not matter.

Each frame should include at least the following fields:

- `observation.images.*`: one or more camera-view images
- `observation.state`: robot state, such as end-effector pose, velocity, gripper state, etc.
- `action`: the control action corresponding to the current frame
- `task`: task description text
- `timestamp` or frame index

For dual-arm tasks, the state and action dimensions will be expanded accordingly.

## 3. Target Format

It is recommended to produce one of the following two formats:

### 3.1 LeRobot v3.0-Style Aggregated Format

This format is suitable as an intermediate format. The directory roughly looks like:

```text
your_dataset_lerobot/
├── data/
├── meta/
├── videos/
└── ...
```

Characteristics: parquet files and videos are usually stored in chunk/file aggregated form, and metadata is also managed in aggregated form.

### 3.2 LeRobot v2.1-Style Legacy Format

If your training or downstream code requires LeRobot 2.1, the dataset should be further organized as:

```text
your_dataset_lerobot/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── camera_0/
│   │   │   ├── episode_000000.mp4
│   │   │   └── ...
├── meta/
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   └── info.json
└── ...
```

Some current StarVLA Franka workflows depend on this layout.

## 4. Conversion Workflow

### 4.1 Step 1: Raw Data -> LeRobot v3.0

The goal of this step is to aggregate multiple raw episodes into a single LeRobot dataset. This usually requires:

1. scanning raw episode files
2. reading observations, actions, and task information
3. organizing the data frame by frame into a unified schema
4. writing the data into a LeRobot dataset
5. generating metadata, statistics, and video information

Typical input parameters look like:

```bash
ROOT_PATH="/path/to/your/raw_franka_dataset"
OUT_PATH="/path/to/output_dir"
TASK_NAME="Franka pick and place"
```

Here, `ROOT_PATH` is the raw data directory, `OUT_PATH` is the output root directory, and `TASK_NAME` is the task description text.

Typical command form:

```bash
python your_converter.py \
    --root-path "$ROOT_PATH" \
    --out-path "$OUT_PATH" \
    --task-name "$TASK_NAME"
```

### 4.2 Output of Step 1

The output directory will typically be:

```text
OUT_PATH/
└── <ROOT_PATH>_lerobot/
```

example：

```text
ROOT_PATH=/data/pick_and_place_franka_data
OUT_PATH=/data/converted
```

Then the output is typically:

```text
/data/converted/pick_and_place_franka_data_lerobot/
```

This step produces a **LeRobot v3.0-style** dataset.

### 4.3 Step 2: LeRobot v3.0 -> LeRobot v2.1

If the training code depends on LeRobot 2.1, a second layout-conversion step is required. This mainly includes:

- splitting aggregated parquet data back into per-episode parquet files
- splitting aggregated videos back into per-episode mp4 files
- rebuilding legacy `episodes.jsonl`, `episodes_stats.jsonl`, and `tasks.jsonl`
- rewriting `info.json` with `codebase_version` set to `v2.1`

This step usually requires:

1. reading episode metadata from the v3.0 dataset
2. locating the start/end range of each episode inside aggregated parquet files
3. exporting each episode as `episode_xxxxxx.parquet`
4. if videos exist, splitting large videos into per-episode clips using timestamps
5. rebuilding the legacy `meta` files
6. updating `info.json` to match the v2.1 schema

Typical command form:

```bash
python your_v30_to_v21_converter.py \
    --root /path/to/your/lerobot_v30_dataset
```

A valid v3.0 -> v2.1 converter will usually:

1. read `meta/episodes/chunk-*/file-*.parquet`
    - recover the data range and video range for each episode
2. rewrite `info.json`
    - set `codebase_version` to `v2.1`
    - restore the legacy `data_path` template
    - restore the legacy `video_path` template
3. split aggregated parquet data into:
    - `data/chunk-xxx/episode_yyyyyy.parquet`
4. if videos exist:
    - split aggregated videos into per-episode mp4 files using `ffmpeg`
5. rebuild legacy metadata:
    - `meta/tasks.jsonl`
    - `meta/episodes.jsonl`
    - `meta/episodes_stats.jsonl`
6. copy ancillary directories (such as `images/`)
7. optionally replace the original directory:
    - first create a temporary `_v2.1` directory
    - remove the original v3.0 directory
    - move the v2.1 result back to the original `--root` path

In other words, after the conversion:

- the `--root` path name can remain unchanged
- but its contents will have changed from a v3.0 layout to a v2.1 layout

## 5. Key Consistency Requirements

Regardless of whether you use our internal scripts, the final dataset should satisfy the following consistency requirements:

### 5.1 Actions

- A single-arm Franka setup usually uses 7D actions.
- The action semantics must match the training configuration, for example:
    - `delta_ee`
    - or another control space that you define

### 5.2 Images

- The number of images must match the camera views used during training.
- The image ordering should remain consistent throughout the dataset.
- Image size and preprocessing should be as consistent as possible.

### 5.3 State

- The state dimension must match `state_dim` in the training configuration.
- Missing state fields or inconsistent ordering can cause silent training errors.

### 5.4 Task Text

- Each episode should have a corresponding task description.
- If natural-language annotations are unavailable, provide a stable task ID and map it to text.

### 5.5 Episode Boundaries

- The start and end indices of each episode must be correct.
- If videos exist, their time ranges must align with the corresponding episodes.


## 6. Example Commands

If you already have corresponding scripts, you can run them in the following order:

```bash
cd /project/vonneumann1/cyx/starVLA_franka/examples/Franka/franka2lerobot

bash convert.sh

python convert_dataset.py \
     --root /path/to/output_dir/<raw_dataset_name>_lerobot
```

## 7. Output Directory Examples

### After Step 1 (v3.0)

The directory will roughly contain:

```text
your_dataset_lerobot/
├── data/
├── meta/
├── videos/
└── ...
```

### After Step 2 (v2.1)

The directory will look more like the legacy LeRobot structure, for example:

```text
your_dataset_lerobot/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
├── videos/
│   ├── chunk-000/
│   │   ├── camera_0/
│   │   │   ├── episode_000000.mp4
│   │   │   └── ...
├── meta/
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   └── info.json
└── ...
```