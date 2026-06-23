"""Action-space metadata used by VAR Stage 1 training.

The Stage 1 tokenizer must agree with the VLA policy's action convention:
action order, chunk horizon, dimensionality, and normalization.  This module
keeps that metadata explicit and serializable so checkpoints can be consumed by
later Stage 2 code without relying on implicit LIBERO/pi0.5 assumptions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _strip_prefix(key: str, prefix: str) -> str:
    return key[len(prefix) :] if key.startswith(prefix) else key


def _default_dim_groups(action_keys: Sequence[str]) -> dict[str, list[int]]:
    """Build conservative dim groups from ordered action keys.

    The groups are only used for metrics/loss weighting; they do not change the
    underlying action order.  Unknown keys are left out rather than guessed.
    """

    groups: dict[str, list[int]] = {"position": [], "rotation": [], "gripper": []}
    for dim_idx, key in enumerate(action_keys):
        short = _strip_prefix(str(key), "action.").lower()
        if short in {"x", "y", "z"} or "position" in short or "pos" in short:
            groups["position"].append(dim_idx)
        elif short in {"roll", "pitch", "yaw", "rx", "ry", "rz"} or "rot" in short:
            groups["rotation"].append(dim_idx)
        elif "gripper" in short or "grip" in short:
            groups["gripper"].append(dim_idx)
    return {name: dims for name, dims in groups.items() if dims}


def _collect_normalization_modes(transform: Any) -> dict[str, str]:
    """Extract per-key normalization modes from a StarVLA transform pipeline."""

    modes: dict[str, str] = {}
    transforms = getattr(transform, "transforms", None)
    if transforms is None:
        transforms = [transform]
    for item in transforms:
        item_modes = getattr(item, "normalization_modes", None)
        if item_modes:
            modes.update(dict(item_modes))
    return modes


def _key_dims_from_data_config(data_config: Any, keys: Sequence[str], modality: str) -> dict[str, int]:
    """Return per-key dims from DataConfig when available, otherwise dim=1."""

    attr = f"{modality}_key_dims"
    if hasattr(data_config, attr):
        dims = dict(getattr(data_config, attr))
        return {key: int(dims.get(key, 1)) for key in keys}
    return {key: 1 for key in keys}


@dataclass
class ActionSpec:
    """Serializable description of the action chunks consumed by Stage 1."""

    action_dim: int
    horizon: int
    action_keys: list[str]
    state_keys: list[str] = field(default_factory=list)
    action_key_dims: dict[str, int] = field(default_factory=dict)
    state_key_dims: dict[str, int] = field(default_factory=dict)
    dim_groups: dict[str, list[int]] = field(default_factory=dict)
    normalization_modes: dict[str, str] = field(default_factory=dict)
    token_order: str = "scale_major"
    source: str = "starvla"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"Action horizon must be positive, got {self.horizon}.")
        if self.action_dim <= 0:
            raise ValueError(f"Action dim must be positive, got {self.action_dim}.")
        if not self.action_key_dims:
            self.action_key_dims = {key: 1 for key in self.action_keys}
        if sum(self.action_key_dims.values()) != self.action_dim:
            raise ValueError(
                "Action dim does not match action_key_dims: "
                f"action_dim={self.action_dim}, action_key_dims={self.action_key_dims}."
            )
        if not self.dim_groups:
            self.dim_groups = _default_dim_groups(self.action_keys)

    @classmethod
    def from_data_config(
        cls,
        data_config: Any,
        *,
        action_dim: int | None = None,
        horizon: int | None = None,
        source: str = "starvla_data_config",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionSpec":
        """Infer an action spec from an existing StarVLA DataConfig."""

        action_keys = list(getattr(data_config, "action_keys"))
        state_keys = list(getattr(data_config, "state_keys", []))
        action_key_dims = _key_dims_from_data_config(data_config, action_keys, "action")
        state_key_dims = _key_dims_from_data_config(data_config, state_keys, "state")
        inferred_dim = sum(action_key_dims.values())
        inferred_horizon = len(list(getattr(data_config, "action_indices")))
        transform = data_config.transform()

        if action_dim is not None and int(action_dim) != inferred_dim:
            raise ValueError(f"Configured action_dim={action_dim} but DataConfig implies {inferred_dim}.")
        if horizon is not None and int(horizon) != inferred_horizon:
            raise ValueError(f"Configured horizon={horizon} but DataConfig implies {inferred_horizon}.")

        return cls(
            action_dim=inferred_dim,
            horizon=inferred_horizon,
            action_keys=action_keys,
            state_keys=state_keys,
            action_key_dims=action_key_dims,
            state_key_dims=state_key_dims,
            dim_groups=_default_dim_groups(action_keys),
            normalization_modes=_collect_normalization_modes(transform),
            source=source,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_sample(
        cls,
        sample: Mapping[str, Any],
        *,
        action_keys: Sequence[str],
        state_keys: Sequence[str] | None = None,
        normalization_modes: Mapping[str, str] | None = None,
        source: str = "starvla_sample",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionSpec":
        """Infer horizon and dimensionality from a sample containing ``action``."""

        action = sample["action"]
        horizon = int(action.shape[0])
        action_dim = int(action.shape[1])
        return cls(
            action_dim=action_dim,
            horizon=horizon,
            action_keys=list(action_keys),
            state_keys=list(state_keys or []),
            action_key_dims={key: 1 for key in action_keys},
            state_key_dims={key: 1 for key in state_keys or []},
            dim_groups=_default_dim_groups(action_keys),
            normalization_modes=dict(normalization_modes or {}),
            source=source,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionSpec":
        return cls(**dict(payload))

