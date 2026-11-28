"""Convert RLBench raw demonstrations to StarVLA's LeRobot v2 layout."""

from dataclasses import dataclass
import argparse
import json
from itertools import chain
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image
import yaml

from examples.simBenchmarks.RLbench.conversion.rlbench_reader import (
    RLBenchEpisode,
    RLBenchRawDataset,
)
from examples.simBenchmarks.RLbench.conversion.rlbench_schema import (
    ActionSpec,
    assert_action_dimension_contract,
)


_LEROBOT_CHUNK_SIZE = 1000


@dataclass(frozen=True)
class ConversionResult:
    total_frames: int
    total_episodes: int
    action_dim: int
    state_dim: int
    output_root: Path


def validate_conversion_action_mode(action_mode: Mapping[str, Any]) -> None:
    """Reject native semantics whose demonstration labels are not implemented.

    RLBench stores absolute observations. Relative JointPosition and pose modes
    require scene-state deltas during conversion, so accepting them here would
    silently write labels with the wrong physical meaning.
    """
    arm = action_mode.get("arm", {})
    arm_type = arm.get("type") if isinstance(arm, Mapping) else None
    kwargs = arm.get("kwargs", {}) if isinstance(arm, Mapping) else {}
    if arm_type in {"JointPosition", "EndEffectorPoseViaIK", "EndEffectorPoseViaPlanning"}:
        if isinstance(kwargs, Mapping) and kwargs.get("absolute_mode") is False:
            raise ValueError(
                "RLBench conversion does not support relative action labels "
                f"for {arm_type}; collect/convert an absolute-mode dataset instead"
            )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _modality_metadata(
    state_slices: Mapping[str, tuple[int, int]],
    camera_names: list[str],
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    """Describe the semantics and positions of state, action, and image in the final dataset."""
    state = {
        name: {
            "start": start,
            "end": end,
            "original_key": "observation.state",
            "dtype": "float32",
        }
        for name, (start, end) in state_slices.items()
    }
    return {
        "state": state,
        "action": {
            "native": {
                "start": 0,
                "end": action_dim,
                "original_key": "action",
                "dtype": "float32",
            }
        },

        "video": {
            camera: {"original_key": f"observation.images.{camera}"}
            for camera in camera_names
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        },
        "state_dim": state_dim,
    }


def _info_metadata(
    episodes: int,
    frames: int,
    tasks: int,
    chunks: int,
    action_spec: ActionSpec,
    state_dim: int,
    camera_shapes: Mapping[str, tuple[int, ...]],
    fps: int,
) -> dict[str, Any]:
    """Generate the overall metadata for the LeRobot dataset."""
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": [state_dim],
            "names": ["proprioception"],
        },
        "action": {
            "dtype": "float32",
            "shape": list(action_spec.action_shape),
            "names": ["native_action"],
        },
        # The official LeRobot source code by default includes metadata with names: None.
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "task": {"dtype": "string", "shape": [1], "names": None},
        # The task-specific low-dimensional state from RLBench
        "observation.task_low_dim_state": {
            "dtype": "float32",
            "shape": [-1],
            "names": ["task_state"],
        },
    }
    for camera, shape in camera_shapes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": list(shape),
            "names": ["height", "width", "rgb"],
        }
    return {
        "codebase_version": "v2.0",
        "robot_type": "rlbench_panda",
        "total_episodes": episodes,
        "total_frames": frames,
        "total_tasks": tasks,
        "total_videos": 0,
        "total_chunks": chunks,
        "chunks_size": _LEROBOT_CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{episodes}"},
        # The standard Parquet path in the official LeRobot_v2 source code.
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        # The official image-path layout use
        # images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpeg.
        # This converter stores each PNG path directly in the Parquet
        # observation.images.* entry. StarVLA's LeRobot loader still requires
        # info["video_path"] during _get_video_path_pattern() initialization,
        # but get_video() first checks that Parquet column and decodes its
        # embedded path. This field therefore preserves loader metadata
        # compatibility; it is not the path used to open a video file here.
        "video_path": "images/{video_key}/episode_{episode_index:06d}/{frame_index:06d}.png",
        "features": features,
        "action_spec": action_spec.as_dict(),
    }


def _copy_image(array: np.ndarray, path: Path) -> dict[str, str]:
    """Copy an RGB uint8 image to the LeRobot dataset layout and return its relative path."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected raw RGB uint8 HxWx3 image, received {array.shape} {array.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)
    # For output_root/images/observation.images.front/episode_000000/000000.png,
    # path.parents[3] is output_root, so this returns the relative path from output_root.
    return {"path": path.relative_to(path.parents[3]).as_posix()}


def _language_instruction(descriptions: tuple[str, ...]) -> str:
    """Encode all RLBench descriptions in LeRobot's scalar task field.
        Though RLBench supports multiple descriptions per episode as list[str],
        LeRobot only supports lang:str as input.
        This function joins all descriptions into one string with a pipe separator.
        For now, we will not consider issues such as semantic conflicts between multiple descriptions.
    """
    if not descriptions:
        raise ValueError("RLBench episode has no language descriptions")
    return " | ".join(descriptions)


def _episode_parquet_path(output_root: Path, episode_id: int) -> Path:
    """Return the LeRobot v2 path for an episode using its declared chunks."""
    chunk_index = episode_id // _LEROBOT_CHUNK_SIZE
    return (
        output_root
        / "data"
        / f"chunk-{chunk_index:03d}"
        / f"episode_{episode_id:06d}.parquet"
    )


def _episode_frame_table(
    episode: RLBenchEpisode,
    output_root: Path,
    global_index: int,
    episode_id: int,
    task_index: int,
    language_instruction: str,
    fps: int,
) -> tuple[pd.DataFrame, dict[str, tuple[int, ...]]]:
    rows = []
    camera_shapes: dict[str, tuple[int, ...]] = {}
    for frame_index, frame in enumerate(episode.frames):
        row: dict[str, Any] = {
            "observation.state": frame.state,
            "action": frame.action,
            # frame_index counts observations serialized in low_dim_obs.pkl,
            # not every CoppeliaSim step. RLBench may execute gripper-closing
            # steps without recording them when record_gripper_closing=False.
            # fps is the effective recorded-observation rate, not _DT or the
            # CoppeliaSim simulation timestep.
            "timestamp": np.float32(frame_index / fps),
            "frame_index": np.int64(frame_index),
            "episode_index": np.int64(episode_id),
            # The global index is the frame's position in the entire dataset, not just the episode.
            "index": np.int64(global_index + frame_index),
            # StarVLA resolves this scalar task_index through tasks.jsonl and
            # passes its string task value to the VLA as the language input.
            # LeRobot does not support an array of descriptions here, so the
            # converter joins every RLBench description into one instruction.
            "task_index": np.int64(task_index),
            "task": language_instruction,
            "observation.task_low_dim_state": frame.task_low_dim_state,
        }
        for camera, image in frame.images.items():
            camera_shapes[camera] = tuple(int(size) for size in image.shape)
            image_path = (
                output_root
                / "images"
                / f"observation.images.{camera}"
                / f"episode_{episode_id:06d}"
                / f"{frame_index:06d}.png"
            )
            row[f"observation.images.{camera}"] = _copy_image(image, image_path)
        rows.append(row)
    return pd.DataFrame(rows), camera_shapes


def convert(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    action_mode: Mapping[str, Any],
    environment_action_shape: tuple[int, ...] | int,
    framework_action_dim: int,
    fps: int,
) -> ConversionResult:
    """Convert raw demonstrations using an explicit recorded-observation rate.

    RLBench raw observations do not retain skipped simulator steps or a
    per-observation timestamp, so this rate cannot be recovered from the raw
    directory. It describes the recorded dataset timeline, not CoppeliaSim's
    simulation timestep.

    The output_root will contain a LeRobot v2 dataset with the following structure:
    output_root/
        data/
            chunk-000/
                episode_000000.parquet
                episode_000001.parquet
                ...
        images/
            observation.images.front/
                episode_000000/
                     000000.png
                     ...

    Metadata files written under `output_root/meta/`:
    - `action_spec.json`: Records the action specification derived from
    the configured action mode and the first frame's action shape. The
    same specification is also included in `info.json`.
    - `info.json`: Describes the overall LeRobot v2 dataset, including
    episode and frame counts, robot type, recorded-observation FPS,
    data path pattern, image path pattern, feature definitions, and
    action specification. Image features and dataset counts are
    collected during conversion.
    - `modality.json`: Describes the semantic layout of the state, action,
    image, and annotation modalities. State slices are derived from the
    first frame's `state_slices`; camera entries map to their
    `observation.images.*` feature keys.
    - `episodes.jsonl`: Stores one JSON object per converted episode,
    including its output episode index, associated task index, frame
    count, and half-open range of global dataset frame indices
    `[dataset_from_index, dataset_to_index)`.
    - `tasks.jsonl`: Stores one JSON object per unique converted language
    instruction, mapping `task_index` to the scalar `task` string.
    RLBench episode descriptions are joined with ` | ` before task
    indices are assigned.

    These metadata files describe the converted dataset and its indexing
    conventions. They do not contain the image files themselves, which
    are saved separately under `output_root/images/`, or the per-episode
    frame tables, which are saved under `output_root/data/chunk-000/`.
    """

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    if fps <= 0:
        raise ValueError(f"Recorded observation fps must be positive, got {fps}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"RLBench conversion output directory is not empty: {output_root}"
        )
    arm_type = str(action_mode["arm"]["type"])
    gripper_type = str(action_mode["gripper"]["type"])
    validate_conversion_action_mode(action_mode)
    # Build reader
    dataset = RLBenchRawDataset(
        raw_root,
        arm_type=arm_type,
        gripper_type=gripper_type,
    )

    episodes = iter(dataset)
    try:
        # Read the first episode to determine the action shape and state dimension.
        first_episode = next(episodes)
    except StopIteration:
        raise ValueError(f"No RLBench episodes found under {raw_root}")
    try:
        first_frame = first_episode.frames[0]
    except IndexError:
        raise ValueError(f"RLBench episode is empty: {first_episode.task_name}/{first_episode.episode_index}")
    action_shape = tuple(first_frame.action.shape)
    spec = ActionSpec(action_mode=action_mode, action_shape=action_shape)
    # These dimensions should be equal
    assert_action_dimension_contract(
        # From ActionSpec, which is derived from the first frame of the first episode
        raw_action_dim=spec.action_dim,
        # From the RLBench environment's ActionSpec.action_shape()
        environment_action_shape=environment_action_shape,
        # From the LeRobot dataset layout, which is flattened to a single dimension
        lerobot_action_shape=(len(first_frame.action),),
        # From the StarVLA framework's ActionSpec.action_dim
        framework_action_dim=framework_action_dim,
        spec=spec,
    )
    state_dim = len(first_frame.state)
    # Initialize the output directory and copy the raw dataset for reference.
    output_root.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    camera_shapes: dict[str, tuple[int, ...]] = {}
    episodes_jsonl = []
    tasks: dict[str, int] = {}
    episode_count = 0
    # Iterate through every episode in the raw dataset,
    # converting each to a Parquet table and copying its images to the LeRobot layout.
    for episode_id, episode in enumerate(chain((first_episode,), episodes)):
        if not episode.frames:
            raise ValueError(f"RLBench episode is empty: {episode.task_name}/{episode.episode_index}")
        if len(episode.frames[0].state) != state_dim:
            raise ValueError("State width changed between RLBench episodes")
        if episode.frames[0].action.shape != action_shape:
            raise ValueError("Action shape changed between RLBench episodes")
        # Trade-off: validate the first frame of each episode to detect
        # action-shape changes across episodes with minimal overhead.
        # Checking every frame would also detect shape changes within an
        # episode, but adds validation cost and may be unnecessary if RLBench
        # guarantees a consistent action shape within each episode.
        # if any(frame.action.shape != action_shape for frame in episode.frames):
        #     raise ValueError("Action shape changed within an RLBench episode")

        descriptions = episode.task_descriptions
        if not descriptions:
            raise ValueError(f"RLBench episode has no language descriptions: {episode.task_name}")
        language_instruction = _language_instruction(descriptions)
        task_index = tasks.setdefault(language_instruction, len(tasks))
        # Generate a Parquet table for the episode and copy its images to the LeRobot layout.
        table, episode_cameras = _episode_frame_table(
            episode,
            output_root,
            total_frames,
            episode_id,
            task_index,
            language_instruction,
            fps,
        )
        camera_shapes.update(episode_cameras)
        parquet_path = _episode_parquet_path(output_root, episode_id)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(parquet_path, index=False)
        length = len(table)
        episodes_jsonl.append(
            {
                "episode_index": episode_id,
                "tasks": [task_index],
                "length": length,
                "dataset_from_index": total_frames,
                # Don't include the last frame in the range, so this is the first index of the next episode.
                "dataset_to_index": total_frames + length,
            }
        )
        total_frames += length
        episode_count += 1

    _write_json(output_root / "meta/action_spec.json", spec.as_dict())
    _write_json(
        output_root / "meta/info.json",
        {
            **_info_metadata(
                episode_count,
                total_frames,
                len(tasks),
                (episode_count + _LEROBOT_CHUNK_SIZE - 1) // _LEROBOT_CHUNK_SIZE,
                spec,
                state_dim,
                camera_shapes,
                fps,
            ),
            "action_spec": spec.as_dict(),
        },
    )
    modality = _modality_metadata(
        first_frame.state_slices,
        sorted(camera_shapes),
        state_dim,
        spec.action_dim,
    )
    _write_json(output_root / "meta/modality.json", modality)
    (output_root / "meta").mkdir(exist_ok=True)
    with (output_root / "meta/episodes.jsonl").open("w", encoding="utf-8") as stream:
        for item in episodes_jsonl:
            stream.write(json.dumps(item) + "\n")
    with (output_root / "meta/tasks.jsonl").open("w", encoding="utf-8") as stream:
        for task, index in sorted(tasks.items(), key=lambda pair: pair[1]):
            stream.write(json.dumps({"task_index": index, "task": task}) + "\n")
    return ConversionResult(
        total_frames=total_frames,
        total_episodes=episode_count,
        action_dim=spec.action_dim,
        state_dim=state_dim,
        output_root=output_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--environment-action-shape",
        type=int,
        required=True,
        help="RLBENCH Environment.action_shape() flattened dimension",
    )
    parser.add_argument("--framework-action-dim", type=int, required=True)
    parser.add_argument(
        "--fps",
        type=int,
        required=True,
        help="Recorded observation rate; not the CoppeliaSim simulation timestep",
    )
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    result = convert(
        args.raw_root,
        args.output_root,
        action_mode=config["env"]["action_mode"],
        environment_action_shape=args.environment_action_shape,
        framework_action_dim=args.framework_action_dim,
        fps=args.fps,
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "total_episodes": result.total_episodes,
                "total_frames": result.total_frames,
                "action_dim": result.action_dim,
                "state_dim": result.state_dim,
            }
        )
    )


if __name__ == "__main__":
    main()
