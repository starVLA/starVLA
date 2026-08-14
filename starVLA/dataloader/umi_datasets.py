"""UMI-specific safety adapter around StarVLA's LeRobot datasets.

The source decoder remains ``LeRobotMixtureDataset``.  This module adds the
boundary checks needed by heterogeneous, converted UMI data without coupling
the generic loader to UMI conventions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UMISamplePolicy:
    action_horizon: int
    action_dim: int
    state_dim: int | None = None
    strict_dimensions: bool = True
    max_views: int | None = None
    max_abs_action: float | None = None
    reject_static_actions: bool = False
    static_action_epsilon: float = 1e-6
    retry_bad_samples: int = 20

    @classmethod
    def from_config(cls, config: Any, model_config: Any = None) -> "UMISamplePolicy":
        def get(name: str, default: Any = None) -> Any:
            if hasattr(config, "get"):
                return config.get(name, default)
            return getattr(config, name, default)

        def model_get(name: str) -> Any:
            if model_config is None:
                return None
            if hasattr(model_config, "get"):
                return model_config.get(name)
            return getattr(model_config, name, None)

        def resolved_dimension(name: str, required: bool = True) -> int | None:
            data_value, model_value = get(name), model_get(name)
            if data_value is not None and model_value is not None and int(data_value) != int(model_value):
                raise ValueError(
                    f"UMI {name} mismatch: datasets.vla_data={data_value}, "
                    f"framework.action_model={model_value}"
                )
            value = data_value if data_value is not None else model_value
            if value is None and required:
                raise ValueError(f"UMI dataloader requires {name} in data or action-model config")
            return None if value is None else int(value)

        action_horizon = resolved_dimension("action_horizon")
        action_dim = resolved_dimension("action_dim")
        include_state = get("include_state", None)
        state_disabled = include_state in (False, "False", "false", 0)
        state_dim = None if state_disabled else resolved_dimension("state_dim", required=False)
        if action_horizon is None or action_dim is None:
            raise ValueError("UMI dataloader requires action_horizon and action_dim")
        return cls(
            action_horizon=action_horizon,
            action_dim=action_dim,
            state_dim=state_dim,
            strict_dimensions=bool(get("strict_dimensions", True)),
            max_views=None if get("max_views") is None else int(get("max_views")),
            max_abs_action=get("max_abs_action"),
            reject_static_actions=bool(get("reject_static_actions", False)),
            static_action_epsilon=float(get("static_action_epsilon", 1e-6)),
            retry_bad_samples=int(get("retry_bad_samples", 20)),
        )


def _clean_language(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "Perform the demonstrated manipulation task."


def _fit_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    strict: bool,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty [T,D] array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    if strict and array.shape != (rows, columns):
        raise ValueError(f"{name} shape {array.shape} != configured {(rows, columns)}")

    # StarVLA labels align to the latest action chunk. Short chunks are edge
    # padded for a finite tensor, while the mask keeps padded cells invalid.
    source = array[-rows:, :columns]
    output = np.zeros((rows, columns), dtype=np.float32)
    mask = np.zeros((rows, columns), dtype=bool)
    row_offset = rows - source.shape[0]
    output[row_offset:, : source.shape[1]] = source
    mask[row_offset:, : source.shape[1]] = True
    if row_offset and source.shape[0]:
        output[:row_offset, : source.shape[1]] = source[0]
    return output.astype(np.float16), mask


class UMISampleAdapter(Dataset):
    """Validate and canonicalize samples emitted by a StarVLA VLA dataset."""

    def __init__(self, dataset: Dataset, policy: UMISamplePolicy, seed: int = 42):
        self.dataset = dataset
        self.policy = policy
        self.seed = int(seed)
        self.rejected_samples = 0
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def save_dataset_statistics(self, *args: Any, **kwargs: Any) -> Any:
        return self.dataset.save_dataset_statistics(*args, **kwargs)

    def _adapt(self, sample: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(sample, dict):
            raise TypeError(f"UMI sample must be dict, got {type(sample).__name__}")

        action, action_mask = _fit_matrix(
            sample.get("action"),
            rows=self.policy.action_horizon,
            columns=self.policy.action_dim,
            strict=self.policy.strict_dimensions,
            name="action",
        )
        valid_action = action[action_mask]
        if self.policy.max_abs_action is not None and valid_action.size:
            if float(np.max(np.abs(valid_action))) > float(self.policy.max_abs_action):
                raise ValueError("action exceeds max_abs_action")
        if self.policy.reject_static_actions and valid_action.size:
            if float(np.ptp(valid_action.astype(np.float32))) <= self.policy.static_action_epsilon:
                raise ValueError("action chunk is static")

        raw_images = sample.get("image", [])
        if isinstance(raw_images, Image.Image):
            raw_images = [raw_images]
        images = list(raw_images)
        if not images:
            raise ValueError("UMI sample has no image views")
        if self.policy.max_views is not None:
            images = images[: self.policy.max_views]
        images = [image.convert("RGB") if isinstance(image, Image.Image) else image for image in images]

        result = dict(sample)
        result.update(
            action=action,
            action_mask=action_mask,
            image=images,
            image_mask=np.ones(len(images), dtype=bool),
            lang=_clean_language(sample.get("lang")),
        )

        if self.policy.state_dim is not None:
            state = sample.get("state")
            if state is None:
                if self.policy.strict_dimensions:
                    raise ValueError("configured state_dim but sample has no state")
                result["state"] = np.zeros((1, self.policy.state_dim), dtype=np.float16)
                result["state_mask"] = np.zeros((1, self.policy.state_dim), dtype=bool)
            else:
                fitted_state, state_mask = _fit_matrix(
                    state,
                    rows=1,
                    columns=self.policy.state_dim,
                    strict=self.policy.strict_dimensions,
                    name="state",
                )
                result["state"] = fitted_state
                result["state_mask"] = state_mask
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        last_error: Exception | None = None
        dataset_length = len(self)
        if dataset_length <= 0:
            raise IndexError("cannot sample an empty UMI dataset")
        # An odd stride gives good coverage for common even-sized datasets;
        # epoch changes the starting offset while preserving reproducibility.
        stride = 2 * ((self.seed + index) % max(1, dataset_length // 2)) + 1
        for attempt in range(self.policy.retry_bad_samples + 1):
            candidate = (index + self.epoch + attempt * stride) % dataset_length
            try:
                return self._adapt(self.dataset[candidate])
            except (KeyError, TypeError, ValueError, OSError) as error:
                self.rejected_samples += 1
                last_error = error
        raise RuntimeError(
            f"UMI sample {index} remained invalid after {self.policy.retry_bad_samples + 1} attempts"
        ) from last_error


def umi_collate_fn(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep StarVLA's expected list-of-example batch contract."""
    if not batch:
        raise ValueError("cannot collate an empty UMI batch")
    return list(batch)


def make_umi_dataloader(cfg: Any) -> DataLoader:
    """Build a StarVLA-compatible loader with UMI validation enabled."""
    # Keep adapter-only imports/tests lightweight; the generic LeRobot stack
    # scans metadata and loads video backends only when a real loader is built.
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    data_cfg = cfg.datasets.vla_data
    action_model_cfg = getattr(getattr(cfg, "framework", None), "action_model", None)
    policy = UMISamplePolicy.from_config(data_cfg, action_model_cfg)

    # Refuse semantically incompatible mixtures before expensive metadata/video
    # setup. Tensor width alone does not make abs-pose, delta-EEF and joints
    # interchangeable targets.
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

    mixture = DATASET_NAMED_MIXTURES[data_cfg.data_mix]
    semantics = {
        getattr(ROBOT_TYPE_CONFIG_MAP[robot_type], "action_semantics", None)
        for _, _, robot_type in mixture
    }
    semantics.discard(None)
    if len(semantics) > 1 and not bool(data_cfg.get("allow_mixed_action_semantics", False)):
        raise ValueError(
            f"UMI mixture {data_cfg.data_mix!r} mixes action semantics {sorted(semantics)}; "
            "split it into compatible buckets or explicitly opt in"
        )
    base = get_vla_dataset(
        data_cfg=data_cfg,
        balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
        balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False),
        seed=int(data_cfg.get("seed", getattr(cfg, "seed", 42))),
    )
    dataset = UMISampleAdapter(base, policy, seed=int(data_cfg.get("seed", getattr(cfg, "seed", 42))))
    workers = int(data_cfg.get("num_workers", 4))
    kwargs: dict[str, Any] = {
        "batch_size": int(data_cfg.per_device_batch_size),
        "collate_fn": umi_collate_fn,
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **kwargs)
    loader.umi_dataset = dataset
    return loader
