"""Reader for the directory layout emitted by RLBench dataset_generator."""

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Iterator, Mapping
import warnings

import numpy as np
from PIL import Image

from .rlbench_schema import action_from_observation, state_from_observation


@dataclass(frozen=True)
class RLBenchFrame:
    """One frame decoded from an RLBench demonstration.

    task_low_dim_state is an optional, task-specific vector provided by
    RLBench's task implementation. It is separate from the common robot
    proprioception in state and does not have a shared dimension or
    universal meaning across tasks. For example, an OpenDrawer task may
    expose the drawer or handle joint state so that the degree of opening can
    be measured, while a multi-stage task such as PutItemInDrawer may expose
    values useful for identifying its current grasp, move, or place stage.
    The exact meaning must be interpreted using the corresponding RLBench
    task implementation.

    The value is None when the raw observation does not provide this
    task-specific state.
    """

    action: np.ndarray
    state: np.ndarray
    state_slices: Mapping[str, tuple[int, int]]
    images: Mapping[str, np.ndarray]
    task_low_dim_state: np.ndarray | None


@dataclass(frozen=True)
class RLBenchEpisode:
    """One episode decoded from an RLBench demonstration.

    task_descriptions are generated for the task variation when the episode
    is initialized and are shared by all frames in this episode.
    """
    task_name: str
    variation: int
    episode_index: int
    task_descriptions: tuple[str, ...]
    frames: tuple[RLBenchFrame, ...]


def _numeric_pngs(directory: Path) -> list[Path]:
    # RLbench saves frames as 0.png, 1.png, etc.
    # We sort numerically to avoid 10.png coming before 2.png.
    return sorted(directory.glob("*.png"), key=lambda path: int(path.stem))


class RLBenchRawDataset:
    """Iterate raw demonstrations without importing RLBench or CoppeliaSim."""

    def __init__(
        self,
        root: str | Path,
        *,
        arm_type: str = "JointVelocity",
        gripper_type: str = "Discrete",
    ) -> None:
        self.root = Path(root)
        self.arm_type = arm_type
        self.gripper_type = gripper_type
        if not self.root.is_dir():
            raise FileNotFoundError(f"RLBench raw dataset does not exist: {self.root}")

    def __iter__(self) -> Iterator[RLBenchEpisode]:
        for task_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            for variation_dir in sorted(task_dir.glob("variation*")):
                descriptions = self._descriptions(variation_dir)
                for episode_dir in sorted((variation_dir / "episodes").glob("episode*")):
                    yield self._read_episode(
                        task_dir.name,
                        int(variation_dir.name.removeprefix("variation")),
                        int(episode_dir.name.removeprefix("episode")),
                        episode_dir,
                        descriptions,
                    )

    @staticmethod
    def _descriptions(variation_dir: Path) -> list[str]:
        path = variation_dir / "variation_descriptions.pkl"
        # Use the directory name as a fallback if the descriptions file is missing or empty
        # For example, open_drawer/variation0/ as variation_dir,
        # but the variation_descriptions.pkl file is missing.
        # The fallback is "open drawer".
        fallback = variation_dir.parent.name.replace("_", " ")
        if not path.exists():
            warnings.warn(
                f"RLBench variation descriptions are missing at {path}; "
                f"falling back to task name {fallback!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return [fallback]
        with path.open("rb") as stream:
            descriptions = pickle.load(stream)
        descriptions = [str(item) for item in descriptions]
        if not descriptions:
            warnings.warn(
                f"RLBench variation descriptions are empty at {path}; "
                f"falling back to task name {fallback!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return [fallback]
        return descriptions

    def _read_episode(
        self,
        task_name: str,
        variation: int,
        episode_index: int,
        episode_dir: Path,
        descriptions: list[str],
    ) -> RLBenchEpisode:

        with (episode_dir / "low_dim_obs.pkl").open("rb") as stream:
            observations = pickle.load(stream)
        # An example camera_paths layout:
        # {
        #     "front_rgb": [
        #         Path("front_rgb/0.png"),
        #         Path("front_rgb/1.png"),
        #         ...
        #     ],
        #     "wrist_rgb": [
        #         Path("wrist_rgb/0.png"),
        #         Path("wrist_rgb/1.png"),
        #         ...
        #     ],
        # }
        camera_paths = {}

        for camera_dir in sorted(episode_dir.iterdir()):
            if not camera_dir.is_dir() or not camera_dir.name.endswith("_rgb"):
                continue

            paths = _numeric_pngs(camera_dir)
            if paths:
                camera_paths[camera_dir.name] = paths

        frames = []
        # Multiple descriptions are accessed, reader just returns all of them.
        for index, observation in enumerate(observations):
            state, state_slices = state_from_observation(observation)
            images = {}
            for camera, paths in camera_paths.items():
                # Check that the number of frames matches the number of observations
                if len(paths) != len(observations):
                    raise ValueError(
                        f"Camera {camera} has {len(paths)} frames, "
                        f"but episode has {len(observations)} observations: "
                        f"{episode_dir}"
                    )
                with Image.open(paths[index]) as image:
                    # image.shape: W,H to rgb.shape: H,W,C
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                # Store it contiguously in memory
                images[camera] = np.ascontiguousarray(rgb)
            frames.append(
                RLBenchFrame(
                    action=action_from_observation(observation, self.arm_type, self.gripper_type),
                    state=state,
                    state_slices=state_slices,
                    images=images,
                    task_low_dim_state=(
                        None
                        if getattr(observation, "task_low_dim_state", None) is None
                        else np.asarray(observation.task_low_dim_state, dtype=np.float32).reshape(-1)
                    ),
                )
            )
        return RLBenchEpisode(
            task_name,
            variation,
            episode_index,
            tuple(descriptions),
            tuple(frames),
        )
