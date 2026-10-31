"""Model-agnostic bridge between StarVLA policy outputs and RLBench.

RLBench owns action semantics. This module builds its native ActionMode from
configuration, derives the StarVLA action dimension from that mode, and passes
validated policy output directly to TaskEnvironment.step.
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RLBenchModules:
    """Native RLBench classes required to build an action mode."""

    move_arm_then_gripper: Callable[..., Any]
    arm_modes: Mapping[str, Callable[..., Any]]
    gripper_modes: Mapping[str, Callable[..., Any]]


@dataclass(frozen=True)
class RLBenchActionSpec:
    """The action contract advertised by the selected RLBench ActionMode."""

    action_shape: tuple[int, ...]
    action_dim: int


_ARM_MODE_NAMES = (
    "JointVelocity",
    "JointPosition",
    "JointTorque",
    "EndEffectorPoseViaIK",
    "EndEffectorPoseViaPlanning",
)
_GRIPPER_MODE_NAMES = ("Discrete", "Continuous")
_DEFAULT_CAMERA_NAMES = (
    "front_rgb",
    "left_shoulder_rgb",
    "right_shoulder_rgb",
    "wrist_rgb",
    "overhead_rgb",
)
_DEFAULT_STATE_KEYS = (
    "state.joint_positions",
    "state.joint_velocities",
    "state.joint_forces",
    "state.gripper_open",
    "state.gripper_joint_positions",
    "state.gripper_touch_forces",
    "state.gripper_pose",
    "state.gripper_matrix",
)


def load_rlbench_modules() -> RLBenchModules:
    """Load RLBench lazily so adapter tests do not need CoppeliaSim."""
    try:
        action_mode = import_module("rlbench.action_modes.action_mode")
        arm_modes = import_module("rlbench.action_modes.arm_action_modes")
        gripper_modes = import_module("rlbench.action_modes.gripper_action_modes")
    except ImportError as exc:
        raise ImportError(
            "RLBench is required for evaluation. Install it in the benchmark "
            "environment and set COPPELIASIM_ROOT before creating an environment."
        ) from exc

    available_arm_modes = {
        name: getattr(arm_modes, name) for name in _ARM_MODE_NAMES if hasattr(arm_modes, name)
    }
    # some RLBench versions do not expose gripper modes, so we check for their presence
    available_gripper_modes = {
        name: getattr(gripper_modes, name)
        for name in _GRIPPER_MODE_NAMES
        if hasattr(gripper_modes, name)
    }
    return RLBenchModules(
        move_arm_then_gripper=action_mode.MoveArmThenGripper,
        arm_modes=available_arm_modes,
        gripper_modes=available_gripper_modes,
    )


def _mode_kwargs(config: Mapping[str, Any], kind: str) -> dict[str, Any]:
    # Adapter should not redefine RLBench semantics, so we pass through any kwargs to the native ActionMode.
    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"RLBench {kind}.kwargs must be a mapping, got {type(kwargs).__name__}")
    return dict(kwargs)


def build_action_mode(config: Mapping[str, Any], modules: RLBenchModules | None = None) -> Any:
    """Build native MoveArmThenGripper without changing its semantics."""
    modules = modules or load_rlbench_modules()
    if config.get("type") != "MoveArmThenGripper":
        raise ValueError("RLBench action_mode.type must be MoveArmThenGripper")

    arm_config = config.get("arm")
    gripper_config = config.get("gripper")
    if not isinstance(arm_config, Mapping) or not isinstance(gripper_config, Mapping):
        raise ValueError("RLBench action_mode requires mapping values for arm and gripper")

    arm_type = arm_config.get("type")
    gripper_type = gripper_config.get("type")
    if arm_type not in modules.arm_modes:
        raise ValueError(f"Unsupported RLBench arm action mode: {arm_type!r}")
    if gripper_type not in modules.gripper_modes:
        raise ValueError(f"Unsupported RLBench gripper action mode: {gripper_type!r}")

    # Build the native RLBench action mode with the provided configuration.
    arm_mode = modules.arm_modes[arm_type](**_mode_kwargs(arm_config, "arm"))
    # Build the native RLBench gripper mode with the provided configuration.
    gripper_mode = modules.gripper_modes[gripper_type](**_mode_kwargs(gripper_config, "gripper"))
    return modules.move_arm_then_gripper(
        arm_action_mode=arm_mode,
        gripper_action_mode=gripper_mode,
    )


def action_spec(source: Any, scene: Any = None) -> RLBenchActionSpec:
    """Derive the action contract from an RLBench environment or ActionMode."""
    try:
        raw_shape = source.action_shape
    except AttributeError as exc:
        raise ValueError("RLBench environment or ActionMode does not expose action_shape") from exc
    if callable(raw_shape):
        scene = scene if scene is not None else getattr(source, "_scene", None)
        if scene is None:
            raise ValueError(
                "RLBench ActionMode.action_shape requires a launched scene; "
                "construct RLBenchAdapter after Environment.launch()"
            )
        raw_shape = raw_shape(scene)
    if isinstance(raw_shape, (int, np.integer)):
        shape = (int(raw_shape),)
    else:
        shape = tuple(int(size) for size in raw_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError(
            f"Invalid RLBench action_shape: {raw_shape!r}"
        )

    return RLBenchActionSpec(
        action_shape=shape,
        action_dim=int(np.prod(shape)),
    )


def validate_native_action(
    action: Any, action_mode: Any, spec: RLBenchActionSpec | None = None
) -> np.ndarray:
    """Validate shape only; never normalize, clip, slice, or re-encode an action."""
    array = action if isinstance(action, np.ndarray) else np.asarray(action)
    spec = spec or action_spec(action_mode)
    if tuple(array.shape) != spec.action_shape:
        raise ValueError(
            f"RLBench ActionMode expected action shape {spec.action_shape}, "
            f"received {tuple(array.shape)}"
        )
    return array


def extract_first_action(
    response: Mapping[str, Any], action_mode: Any, spec: RLBenchActionSpec | None = None
) -> np.ndarray:
    """Read one StarVLA prediction while retaining the selected ActionMode contract."""
    try:
        actions = np.asarray(response["data"]["actions"])
    except (KeyError, TypeError) as exc:
        raise KeyError("Policy response must contain data.actions") from exc
    if actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[1] < 1:
        raise ValueError(
            "Policy response data.actions must have shape [batch, horizon, action_dim], "
            f"received {actions.shape}"
        )
    return validate_native_action(actions[0, 0], action_mode, spec)


def _observation_value(observation: Any, name: str) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name)
    return getattr(observation, name, None)


def _language(descriptions: Sequence[str] | str) -> str:
    if isinstance(descriptions, str):
        return descriptions
    if not descriptions:
        raise ValueError("RLBench reset returned no language descriptions")
    # Keep evaluation text identical to the conversion-time task encoding.
    return " | ".join(str(description) for description in descriptions)


class RLBenchAdapter:
    """Small environment-side adapter that preserves native RLBench actions."""

    def __init__(
        self,
        environment: Any,
        action_mode: Any,
        camera_names: Sequence[str] = _DEFAULT_CAMERA_NAMES,
        state_keys: Sequence[str] = (),
        gripper_type: str | None = None,
    ):
        self.environment = environment
        self.action_mode = action_mode
        self.camera_names = tuple(camera_names)
        self.state_keys = tuple(state_keys)
        self.gripper_type = gripper_type
        self.task = None
        # Prefer RLBench Environment.action_shape; retain the native ActionMode
        # path for lightweight adapters and custom environments.
        source = environment if hasattr(environment, "action_shape") else action_mode
        self.spec = action_spec(source, scene=getattr(environment, "_scene", None))
    def reset(self, task: Any) -> tuple[Sequence[str] | str, Any]:
        self.task = self.environment.get_task(task)
        return self.task.reset()

    def step(self, raw_action: Any) -> tuple[Any, float, bool]:
        if self.task is None:
            raise RuntimeError("Call reset(task) before step(action)")
        native_action = self.action_for_step(raw_action, self.gripper_type)
        return self.task.step(native_action)

    def action_for_step(self, raw_action: Any, gripper_type: str | None = None) -> np.ndarray:
        """Apply only native Discrete-gripper thresholding before env.step."""
        native_action = validate_native_action(raw_action, self.action_mode, self.spec)
        if gripper_type == "Discrete":
            native_action = native_action.copy()
            native_action[-1] = float(native_action[-1] > 0.5)
        return native_action

    def policy_example(self, observation: Any, descriptions: Sequence[str] | str) -> dict[str, Any]:
        images = []
        # Extract all configured camera images from the RLBench observation.
        for camera_name in self.camera_names:
            image = _observation_value(observation, camera_name)
            if image is not None:
                images.append(np.asarray(image))
        if not images:
            raise ValueError(
                "No configured RLBench camera image was found in observation. "
                f"Checked: {self.camera_names}"
            )
        example = {"image": images, "lang": _language(descriptions)}
        if self.state_keys:
            state_parts = []
            for full_key in self.state_keys:
                name = full_key.removeprefix("state.")
                value = _observation_value(observation, name)
                if value is None:
                    raise ValueError(f"RLBench observation is missing configured state key {full_key!r}")
                state_parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
            example["state"] = np.concatenate(state_parts, axis=0)[None, :]
        return example


class StarVLAPolicyAdapter:
    """Translate model-agnostic observation/language inputs into a native action."""

    def __init__(
        self,
        client: Any,
        action_mode: Any,
        unnorm_key: str | None = None,
        action_specification: RLBenchActionSpec | None = None,
    ):
        self.client = client
        self.action_mode = action_mode
        self.unnorm_key = unnorm_key
        self.action_specification = action_specification

    def predict(self, example: Mapping[str, Any]) -> np.ndarray:
        request = {"examples": [dict(example)], "do_sample": False}
        if self.unnorm_key is not None:
            request["unnorm_key"] = self.unnorm_key
        return extract_first_action(
            self.client.predict_action(request), self.action_mode, self.action_specification
        )
