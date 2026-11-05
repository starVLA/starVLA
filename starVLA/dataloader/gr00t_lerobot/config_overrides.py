"""Runtime overrides shared by training and deployment data pipelines.

DataConfig classes remain the source of modality keys and default transforms,
while the training YAML may override the action chunk length and normalization
mode without mutating the globally registered DataConfig instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    Normalizer,
    StateActionTransform,
)


def _to_plain_value(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _positive_horizon(value: Any, *, source: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    try:
        horizon = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} must be a positive integer, got {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    if isinstance(value, str) and value.strip() != str(horizon):
        raise ValueError(f"{source} must be a positive integer, got {value!r}")
    if horizon <= 0:
        raise ValueError(f"{source} must be a positive integer, got {horizon}")
    return horizon


def resolve_action_horizon(
    *,
    model_action_horizon: Any = None,
    data_action_horizon: Any = None,
) -> int | None:
    """Resolve and validate the model/data action horizon.

    The model value is the canonical source used to construct
    ``range(action_horizon)``.  ``datasets.vla_data.action_horizon`` is an
    explicit consistency assertion when present.  A missing side falls back
    to the value supplied by the other side for backward compatibility.
    """

    model_horizon = _positive_horizon(
        model_action_horizon, source="model action_horizon"
    )
    data_horizon = _positive_horizon(
        data_action_horizon, source="datasets.vla_data.action_horizon"
    )

    if (
        model_horizon is not None
        and data_horizon is not None
        and model_horizon != data_horizon
    ):
        raise ValueError(
            "Action horizon mismatch: "
            f"model action_horizon={model_horizon}, but "
            f"datasets.vla_data.action_horizon={data_horizon}. "
            "Set datasets.vla_data.action_horizon to an interpolation of the "
            "model action_horizon (or the matching literal)."
        )

    return model_horizon if model_horizon is not None else data_horizon


def override_action_horizon(
    modality_configs: dict[str, ModalityConfig],
    action_horizon: int | None,
) -> dict[str, ModalityConfig]:
    """Return modality configs whose action indices are ``range(horizon)``."""

    if action_horizon is None:
        return modality_configs
    if "action" not in modality_configs:
        raise ValueError(
            "Cannot apply action_horizon because the DataConfig has no action modality"
        )

    action_cfg = modality_configs["action"]
    updated = dict(modality_configs)
    updated["action"] = ModalityConfig(
        delta_indices=list(range(action_horizon)),
        modality_keys=list(action_cfg.modality_keys),
    )
    return updated


def _validate_normalization_mode(mode: Any, *, key: str) -> str | None:
    if mode is None:
        return None
    if not isinstance(mode, str) or mode not in Normalizer.valid_modes:
        raise ValueError(
            f"Invalid normalization mode for {key!r}: {mode!r}. "
            f"Expected one of {Normalizer.valid_modes} or null."
        )
    return mode


def _state_action_transforms(
    transform: ComposedModalityTransform,
) -> list[StateActionTransform]:
    return [
        item
        for item in transform.transforms
        if isinstance(item, StateActionTransform)
    ]


def _set_exact_normalization_mode(
    transforms: list[StateActionTransform],
    full_key: str,
    mode: Any,
) -> None:
    normalized_mode = _validate_normalization_mode(mode, key=full_key)
    matches = [transform for transform in transforms if full_key in transform.apply_to]
    if not matches:
        available = sorted(
            {key for transform in transforms for key in transform.apply_to}
        )
        raise ValueError(
            f"Cannot override normalization for {full_key!r}; it is not handled by "
            f"a StateActionTransform. Available keys: {available}"
        )
    for transform in matches:
        if normalized_mode is None:
            transform.normalization_modes.pop(full_key, None)
        else:
            transform.normalization_modes[full_key] = normalized_mode


def _replace_existing_modes(
    transforms: list[StateActionTransform],
    mode: Any,
    *,
    modality: str | None = None,
) -> int:
    label = modality if modality is not None else "normalization_modes"
    normalized_mode = _validate_normalization_mode(mode, key=label)
    changed = 0
    prefix = f"{modality}." if modality is not None else None
    for transform in transforms:
        keys = list(transform.normalization_modes)
        for key in keys:
            if prefix is not None and not key.startswith(prefix):
                continue
            changed += 1
            if normalized_mode is None:
                transform.normalization_modes.pop(key, None)
            else:
                transform.normalization_modes[key] = normalized_mode
    return changed


def apply_normalization_mode_overrides(
    transform: ComposedModalityTransform,
    normalization_modes: Any,
) -> ComposedModalityTransform:
    """Apply YAML normalization overrides to a transform pipeline.

    Supported forms under ``datasets.vla_data.normalization_modes``::

        normalization_modes: q99

    replaces every normalization mode already selected by the DataConfig.
    A mapping can target a modality or individual keys::

        normalization_modes:
          action: q99
          state.joints: mean_std
          action.gripper: binary

    ``null`` disables normalization for the selected existing modality/key.
    Per-key entries may also be nested below ``action`` or ``state``.
    """

    normalization_modes = _to_plain_value(normalization_modes)
    if normalization_modes is None:
        return transform

    transforms = _state_action_transforms(transform)
    if not transforms:
        raise ValueError(
            "normalization_modes was configured, but the DataConfig transform "
            "pipeline contains no StateActionTransform"
        )

    if isinstance(normalization_modes, str):
        if _replace_existing_modes(transforms, normalization_modes) == 0:
            raise ValueError(
                "normalization_modes was configured, but the DataConfig has no "
                "existing normalization entries to override"
            )
        return transform

    if not isinstance(normalization_modes, Mapping):
        raise ValueError(
            "datasets.vla_data.normalization_modes must be a mode string, a "
            "mapping, or null"
        )

    for key, value in normalization_modes.items():
        key = str(key)
        value = _to_plain_value(value)
        if key in {"action", "state"}:
            if isinstance(value, Mapping):
                for subkey, submode in value.items():
                    subkey = str(subkey)
                    if subkey in {"*", "__all__"}:
                        if _replace_existing_modes(
                            transforms, submode, modality=key
                        ) == 0:
                            raise ValueError(
                                f"No existing {key} normalization entries to override"
                            )
                    else:
                        full_key = subkey if "." in subkey else f"{key}.{subkey}"
                        _set_exact_normalization_mode(transforms, full_key, submode)
            else:
                if _replace_existing_modes(transforms, value, modality=key) == 0:
                    raise ValueError(
                        f"No existing {key} normalization entries to override"
                    )
        elif "." in key:
            _set_exact_normalization_mode(transforms, key, value)
        else:
            raise ValueError(
                f"Invalid normalization override key {key!r}; use 'action', "
                "'state', or a full key such as 'action.gripper'."
            )

    return transform


def build_overridden_data_pipeline(
    data_config: Any,
    *,
    action_horizon: int | None = None,
    normalization_modes: Any = None,
) -> tuple[dict[str, ModalityConfig], ComposedModalityTransform]:
    """Build a DataConfig pipeline with non-mutating runtime overrides."""

    modality_configs = override_action_horizon(
        data_config.modality_config(), action_horizon
    )
    transform = data_config.transform()
    if not isinstance(transform, ComposedModalityTransform):
        transform = ComposedModalityTransform(transforms=[transform])
    transform = apply_normalization_mode_overrides(transform, normalization_modes)
    return modality_configs, transform
