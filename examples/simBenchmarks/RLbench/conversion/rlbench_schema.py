"""Schemas and dimension contracts for native RLBench actions."""

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    """The action contract shared by RLBench, LeRobot, and StarVLA."""

    action_mode: Mapping[str, Any]
    action_shape: tuple[int, ...]

    @property
    def action_dim(self) -> int:
        return int(prod(self.action_shape))

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": "RLBench",
            "action_mode": _json_copy(self.action_mode),
            "action_shape": list(self.action_shape),
            "action_dim": self.action_dim,
        }


def _json_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_copy(item) for item in value]
    return value


def normalize_shape(shape: Sequence[int] | int) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        shape = (int(shape),)
    result = tuple(int(size) for size in shape)
    if not result or any(size <= 0 for size in result):
        raise ValueError(f"Invalid action shape: {shape!r}")
    return result


def assert_action_dimension_contract(
    *,
    raw_action_dim: int,
    environment_action_shape: Sequence[int] | int,
    lerobot_action_shape: Sequence[int],
    framework_action_dim: int,
    spec: ActionSpec,
) -> None:
    """Fail early if any layer silently changes the native action width."""

    environment_shape = normalize_shape(environment_action_shape)
    lerobot_shape = tuple(int(size) for size in lerobot_action_shape)
    expected = spec.action_dim
    values = {
        "raw dataset": int(raw_action_dim),
        "RLBench environment": int(prod(environment_shape)),
        "LeRobot action": int(lerobot_shape[-1]) if lerobot_shape else -1,
        "StarVLA framework": int(framework_action_dim),
        "ActionSpec": expected,
    }
    if environment_shape != spec.action_shape:
        raise ValueError(
            f"RLBench environment shape {environment_shape} does not match "
            f"ActionSpec shape {spec.action_shape}"
        )
    if len(lerobot_shape) < 1 or any(size <= 0 for size in lerobot_shape):
        raise ValueError(f"Invalid LeRobot action shape: {lerobot_action_shape!r}")
    if len(set(values.values())) != 1:
        raise ValueError(f"Action dimension contract mismatch: {values}")


STATE_FIELDS = (
    "joint_positions",
    "joint_velocities",
    "joint_forces",
    "gripper_open",
    "gripper_joint_positions",
    "gripper_touch_forces",
    "gripper_pose",
    "gripper_matrix",
)


def flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"Expected numeric observation, received {array.dtype}")
    return array.reshape(-1)


def state_from_observation(observation: Any) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """Flatten supported RLBench state fields into one proprioceptive vector.

    The fields are concatenated in STATE_FIELDS order. The returned
    slices map each present field to its half-open range in state.
    For example, with 7D joint positions and 7D joint velocities:

        state.shape == (14,)
        slices == {
            "joint_positions": (0, 7),
            "joint_velocities": (7, 14),
        }
        state[0:7] == observation.joint_positions
        state[7:14] == observation.joint_velocities

    Missing fields are skipped with a RuntimeWarning.
    """
    values: list[np.ndarray] = []
    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for field in STATE_FIELDS:
        value = getattr(observation, field, None)
        if value is None:
            warnings.warn(
                f"RLBench observation is missing state field {field!r}; skipping it",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        flattened = flatten_numeric(value)
        values.append(flattened)
        slices[field] = (cursor, cursor + len(flattened))
        cursor += len(flattened)
    if not values:
        raise ValueError("RLBench observation contains no supported numeric state fields")
    return np.concatenate(values).astype(np.float32, copy=False), slices


def action_from_observation(observation: Any, arm_type: str, gripper_type: str) -> np.ndarray:
    arm_fields = {
        "JointVelocity": "joint_velocities",
        "JointPosition": "joint_positions",
        "JointTorque": "joint_forces",
        "EndEffectorPoseViaIK": "gripper_pose",
        "EndEffectorPoseViaPlanning": "gripper_pose",
    }
    if arm_type not in arm_fields:
        raise ValueError(f"Unsupported RLBench arm action mode: {arm_type!r}")
    arm = flatten_numeric(getattr(observation, arm_fields[arm_type]))
    if gripper_type not in {"Discrete", "Continuous"}:
        raise ValueError(f"Unsupported RLBench gripper action mode: {gripper_type!r}")
    # Whether the gripper is discrete or continuous, the gripper action is a single float.
    gripper = flatten_numeric(getattr(observation, "gripper_open"))
    return np.concatenate((arm, gripper)).astype(np.float32, copy=False)
