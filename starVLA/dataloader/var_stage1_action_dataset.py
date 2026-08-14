"""Action-only dataset wrapper for VAR Stage 1 tokenizer training.

This module intentionally reuses StarVLA's existing LeRobot dataset pipeline.
It does not read LIBERO parquet files or normalize actions independently.  The
returned ``actions`` tensor is exactly the action chunk produced by the current
StarVLA data config and transform stack.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from starVLA.utils.action_spec import ActionSpec


def _as_data_cfg(config_or_data_cfg: Any) -> Any:
    """Accept either the full StarVLA config or the ``datasets.vla_data`` node."""

    if hasattr(config_or_data_cfg, "datasets"):
        return config_or_data_cfg.datasets.vla_data
    return config_or_data_cfg


def _to_plain_container(value: Any) -> Any:
    if value is None:
        return None
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get_expected_action_horizon(config_or_data_cfg: Any) -> int | None:
    if hasattr(config_or_data_cfg, "framework"):
        action_model = getattr(config_or_data_cfg.framework, "action_model", None)
        if action_model is not None and hasattr(action_model, "action_horizon"):
            return int(action_model.action_horizon)
    if hasattr(config_or_data_cfg, "action_horizon"):
        return int(config_or_data_cfg.action_horizon)
    return None


def _get_expected_action_dim(config_or_data_cfg: Any) -> int | None:
    if hasattr(config_or_data_cfg, "framework"):
        action_model = getattr(config_or_data_cfg.framework, "action_model", None)
        if action_model is not None and hasattr(action_model, "action_dim"):
            return int(action_model.action_dim)
    if hasattr(config_or_data_cfg, "action_dim"):
        return int(config_or_data_cfg.action_dim)
    return None


def _to_numpy_action(action: Any) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2:
        raise ValueError(f"Expected action chunk with shape [T, D], got {action.shape}.")
    return action


def _is_abs_action_mode(dataset: Any) -> bool:
    return getattr(dataset, "_action_mode", "abs") in (None, "abs")


def _trajectory_data_exists(dataset: Any, trajectory_id: int) -> bool:
    """Return whether an eagerly enumerated LeRobot v2.0 trajectory file exists.

    For dataset versions or implementations we do not recognize, return True so
    the normal dataset reader remains the source of truth.
    """

    if getattr(dataset, "_lerobot_version", None) != "v2.0":
        return True

    dataset_path = getattr(dataset, "dataset_path", None)
    data_path_pattern = getattr(dataset, "data_path_pattern", None)
    get_episode_chunk = getattr(dataset, "get_episode_chunk", None)
    if dataset_path is None or data_path_pattern is None or get_episode_chunk is None:
        return True

    try:
        trajectory_id = int(trajectory_id)
        chunk_index = get_episode_chunk(trajectory_id)
        parquet_path = Path(dataset_path) / str(data_path_pattern).format(
            episode_chunk=chunk_index,
            episode_index=trajectory_id,
        )
    except Exception:
        return True
    return parquet_path.exists()


def _libero_task_metadata(dataset: Any, base_index: int) -> dict[str, Any]:
    """Extract stable task-level identifiers from the already loaded trajectory."""

    trajectory_data = getattr(dataset, "curr_traj_data", None)
    if trajectory_data is None or "task_index" not in trajectory_data.columns:
        return {}

    row_index = min(max(int(base_index), 0), len(trajectory_data) - 1)
    task_index = int(trajectory_data["task_index"].iloc[row_index])
    dataset_name = str(getattr(dataset, "dataset_name", "unknown"))
    suite_name = dataset_name.rsplit("/", 1)[-1]
    for suffix in ("_no_noops_1.0.0_lerobot", "_1.0.0_lerobot", "_lerobot"):
        if suite_name.endswith(suffix):
            suite_name = suite_name[: -len(suffix)]
            break

    result: dict[str, Any] = {
        "suite_name": suite_name,
        "task_index": task_index,
        "task_name": f"{suite_name}::task_{task_index:02d}",
    }
    tasks = getattr(dataset, "tasks", None)
    try:
        task_row = tasks.loc[task_index]
        description = task_row["task"] if hasattr(task_row, "__getitem__") else None
        if description is not None:
            result["task_description"] = str(description)
    except Exception:
        pass
    return result


class VARStage1ActionDataset(Dataset):
    """Expose normalized action chunks from a StarVLA VLA dataset.

    Args:
        config_or_data_cfg: Full OmegaConf config or ``cfg.datasets.vla_data``.
        mode: Dataset mode passed to the underlying mixture dataset.
        balance_dataset_weights: Forwarded to ``get_vla_dataset``.  ``False``
            keeps LIBERO suite weights equal when mixture weights are equal.
        balance_trajectory_weights: Forwarded to ``get_vla_dataset``.
        return_raw_actions: Also return pre-transform actions. These are still
            after StarVLA's optional action_mode conversion, but before tensor
            conversion and normalization.
        window_mode: ``"full"`` enumerates only complete action windows. This is
            the preferred Stage 1 mode. ``"padded"`` mirrors StarVLA policy
            training and pads near episode ends with the last action.
    """

    def __init__(
        self,
        config_or_data_cfg: Any,
        *,
        mode: str = "train",
        balance_dataset_weights: bool = False,
        balance_trajectory_weights: bool = False,
        seed: int = 42,
        return_raw_actions: bool = True,
        window_mode: str = "full",
    ) -> None:
        self.config_or_data_cfg = config_or_data_cfg
        self.data_cfg = _as_data_cfg(config_or_data_cfg)
        self.expected_action_horizon = _get_expected_action_horizon(config_or_data_cfg)
        self.expected_action_dim = _get_expected_action_dim(config_or_data_cfg)
        self.mode = mode
        self.return_raw_actions = return_raw_actions
        if window_mode not in {"full", "padded"}:
            raise ValueError(f"window_mode must be 'full' or 'padded', got {window_mode!r}.")
        self.window_mode = window_mode
        self.source_dataset = get_vla_dataset(
            data_cfg=self.data_cfg,
            mode=mode,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
        )

        self.action_spec = self._build_action_spec()
        self._full_windows = self._build_full_windows() if self.window_mode == "full" else []

    def _first_robot_type(self) -> str:
        mixture = DATASET_NAMED_MIXTURES[self.data_cfg.data_mix]
        robot_types = [entry[2] for entry in mixture]
        if len(set(robot_types)) != 1:
            raise ValueError(
                "VARStage1ActionDataset currently expects a single action convention per run. "
                f"Got robot_types={sorted(set(robot_types))} for data_mix={self.data_cfg.data_mix}."
            )
        return robot_types[0]

    def _build_action_spec(self) -> ActionSpec:
        robot_type = self._first_robot_type()
        data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode_apply_keys = _to_plain_container(self.data_cfg.get("action_mode_apply_keys", None))
        action_mode_state_map = _to_plain_container(self.data_cfg.get("action_mode_state_map", None)) or {}
        return ActionSpec.from_data_config(
            data_config,
            action_dim=self.expected_action_dim,
            horizon=self.expected_action_horizon,
            source="starvla_lerobot",
            metadata={
                "data_root_dir": str(self.data_cfg.data_root_dir),
                "data_mix": str(self.data_cfg.data_mix),
                "robot_type": robot_type,
                "mode": self.mode,
                "expected_action_horizon": self.expected_action_horizon,
                "expected_action_dim": self.expected_action_dim,
                "action_mode": str(action_mode),
                "action_mode_apply_keys": action_mode_apply_keys,
                "action_mode_state_map": action_mode_state_map,
            },
        )

    def __len__(self) -> int:
        if self.window_mode == "full":
            return len(self._full_windows)
        return len(self.source_dataset)

    def _build_full_windows(self) -> list[tuple[int, int, int]]:
        """Enumerate every non-padded action window in the source mixture."""

        windows: list[tuple[int, int, int]] = []
        skipped_missing: dict[str, int] = {}
        horizon = int(self.action_spec.horizon)
        for dataset_index, dataset in enumerate(self.source_dataset.datasets):
            for trajectory_id, trajectory_length in zip(dataset.trajectory_ids, dataset.trajectory_lengths):
                if not _trajectory_data_exists(dataset, int(trajectory_id)):
                    dataset_name = str(getattr(dataset, "dataset_name", dataset_index))
                    skipped_missing[dataset_name] = skipped_missing.get(dataset_name, 0) + 1
                    continue
                max_start = int(trajectory_length) - horizon + 1
                if max_start <= 0:
                    continue
                for base_index in range(max_start):
                    windows.append((dataset_index, int(trajectory_id), base_index))
        if skipped_missing:
            skipped_summary = ", ".join(
                f"{dataset_name}: {count}" for dataset_name, count in sorted(skipped_missing.items())
            )
            warnings.warn(
                "Skipped missing trajectory parquet files while building full windows: "
                f"{skipped_summary}.",
                RuntimeWarning,
            )
        if not windows:
            raise RuntimeError(
                f"No complete action windows found for horizon={horizon}. "
                "Use window_mode='padded' if this is expected."
            )
        return windows

    def _sample_location(self, index: int) -> tuple[Any, int, int, int | None]:
        if self.window_mode == "full":
            dataset_index, trajectory_id, base_index = self._full_windows[index]
            return self.source_dataset.datasets[dataset_index], trajectory_id, base_index, dataset_index

        dataset, trajectory_id, base_index = self.source_dataset.sample_step(index)
        return dataset, int(trajectory_id), int(base_index), None

    def _get_action_only_data(self, dataset: Any, trajectory_id: int, base_index: int) -> dict[str, Any]:
        dataset.curr_traj_data = dataset.get_trajectory_data(trajectory_id)
        dataset.curr_traj_id = trajectory_id
        data: dict[str, Any] = {}

        for action_key in dataset.modality_keys["action"]:
            data[action_key] = dataset.get_state_or_action(trajectory_id, "action", action_key, base_index)

        for state_key in dataset.modality_keys.get("state", []):
            data[state_key] = dataset.get_state_or_action(trajectory_id, "state", state_key, base_index)

        if not _is_abs_action_mode(dataset):
            for action_key in dataset.modality_keys["action"]:
                state_key = dataset._infer_state_key_for_action(action_key)
                if state_key is not None and state_key in dataset.modality_keys.get("state", []):
                    data[state_key] = dataset.get_state_or_action(trajectory_id, "state", state_key, base_index)

        return dataset._apply_action_mode(data)

    def _sample_raw_and_transformed(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Read and transform action chunks without loading image or language data."""

        dataset, trajectory_id, base_index, dataset_index = self._sample_location(index)
        raw_data = self._get_action_only_data(dataset, trajectory_id, base_index)

        raw_actions = []
        for action_key in dataset.modality_keys["action"]:
            raw_actions.append(np.asarray(raw_data[action_key], dtype=np.float32))
        raw_action = np.concatenate(raw_actions, axis=1)

        transformed = dataset.transforms(dict(raw_data))
        actions = []
        for action_key in dataset.modality_keys["action"]:
            value = transformed[action_key]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            actions.append(np.asarray(value, dtype=np.float32))
        action = np.concatenate(actions, axis=1)

        metadata = {
            "dataset_name": dataset.dataset_name,
            "robot_tag": dataset.tag,
            "trajectory_id": int(trajectory_id),
            "base_index": int(base_index),
            "window_mode": self.window_mode,
        }
        metadata.update(_libero_task_metadata(dataset, base_index))
        if dataset_index is not None:
            metadata["dataset_index"] = int(dataset_index)
        return action, raw_action, metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        action, raw_action, metadata = self._sample_raw_and_transformed(index)
        actions = _to_numpy_action(action)

        if actions.shape != (self.action_spec.horizon, self.action_spec.action_dim):
            raise ValueError(
                "Action chunk shape does not match ActionSpec: "
                f"actions={actions.shape}, expected={(self.action_spec.horizon, self.action_spec.action_dim)}."
            )

        item: dict[str, Any] = {
            "actions": torch.from_numpy(actions),
            "metadata": metadata,
        }
        if self.return_raw_actions:
            item["actions_raw"] = torch.from_numpy(_to_numpy_action(raw_action))
        return item


def build_var_stage1_action_dataset_from_yaml(
    config_yaml: str | Path,
    *,
    mode: str = "train",
    **kwargs: Any,
) -> VARStage1ActionDataset:
    """Convenience helper for scripts and smoke tests."""

    cfg = OmegaConf.load(config_yaml)
    return VARStage1ActionDataset(cfg, mode=mode, **kwargs)
