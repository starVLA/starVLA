"""LIBERO Stage 2 dataset for VAR action-token policy training."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.model.modules.action_tokenizer import Stage1Artifact, load_frozen_var_action_tokenizer
from starVLA.training.train_var_stage1 import load_starvla_base_config


def collate_var_stage2_token_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep StarVLA's example-list convention for VLA framework training."""

    return batch


class VARStage2TokenDataset(Dataset):
    """Expose LIBERO observations with frozen Stage 1 action-token labels.

    This first implementation is intentionally LIBERO/pi0.5 focused. It reuses
    ``VARStage1ActionDataset`` for deterministic window enumeration and
    ActionSpec construction, while reading the full StarVLA sample to provide
    images, language, optional state, and normalized expert actions.
    """

    def __init__(
        self,
        config_or_stage1_cfg: Any,
        *,
        stage1_artifact: Stage1Artifact | None = None,
        stage1_artifact_path: str | Path | None = None,
        mode: str = "train",
        balance_dataset_weights: bool = False,
        balance_trajectory_weights: bool = False,
        seed: int = 42,
        window_mode: str = "full",
        device: str | torch.device = "cpu",
        return_token_text: bool = False,
        token_text_codec: Any | None = None,
        token_cache_path: str | Path | None = None,
        validate_cache_online: bool = False,
        max_samples: int | None = None,
        sample_indices: list[int] | tuple[int, ...] | None = None,
        skip_bad_samples: bool = False,
        max_read_retries: int = 8,
    ) -> None:
        if stage1_artifact is None:
            if stage1_artifact_path is None:
                raise ValueError("Either stage1_artifact or stage1_artifact_path must be provided.")
            stage1_artifact = load_frozen_var_action_tokenizer(stage1_artifact_path, device=device)

        self.stage1_artifact = stage1_artifact
        self.tokenizer = stage1_artifact.tokenizer
        self.return_token_text = return_token_text
        self.token_text_codec = token_text_codec
        self.token_cache = self._load_token_cache(token_cache_path) if token_cache_path is not None else None
        self.validate_cache_online = bool(validate_cache_online)
        self.skip_bad_samples = bool(skip_bad_samples)
        self.max_read_retries = max(1, int(max_read_retries))
        self.sample_indices = self._normalize_sample_indices(sample_indices)
        self.max_samples = None if max_samples is None else max(0, int(max_samples))

        self.stage1_dataset = VARStage1ActionDataset(
            config_or_stage1_cfg,
            mode=mode,
            balance_dataset_weights=balance_dataset_weights,
            balance_trajectory_weights=balance_trajectory_weights,
            seed=seed,
            return_raw_actions=False,
            window_mode=window_mode,
        )
        self._length = len(self.stage1_dataset)
        self._validate_stage1_contract()
        if self.token_cache is not None:
            self._validate_token_cache()
        self._apply_subset()

    def _normalize_sample_indices(self, sample_indices: list[int] | tuple[int, ...] | None) -> list[int] | None:
        if sample_indices is None:
            return None
        return [int(index) for index in sample_indices]

    def _apply_subset(self) -> None:
        if self.sample_indices is not None:
            if any(index < 0 or index >= self._length for index in self.sample_indices):
                raise ValueError(f"sample_indices must be within [0, {self._length}); got {self.sample_indices[:10]}.")
            self._length = len(self.sample_indices)
        if self.max_samples is not None:
            self._length = min(self._length, self.max_samples)

    def _validate_stage1_contract(self) -> None:
        dataset_spec = self.stage1_dataset.action_spec
        artifact_spec = self.stage1_artifact.action_spec
        if dataset_spec.action_dim != artifact_spec.action_dim:
            raise ValueError(f"Action dim mismatch: dataset={dataset_spec.action_dim}, artifact={artifact_spec.action_dim}")
        if dataset_spec.horizon != artifact_spec.horizon:
            raise ValueError(f"Action horizon mismatch: dataset={dataset_spec.horizon}, artifact={artifact_spec.horizon}")
        if dataset_spec.action_keys != artifact_spec.action_keys:
            raise ValueError(f"Action key mismatch: dataset={dataset_spec.action_keys}, artifact={artifact_spec.action_keys}")

    @property
    def action_spec(self):
        return self.stage1_dataset.action_spec

    @property
    def token_dim(self) -> int:
        return int(self.tokenizer.token_dim)

    def __len__(self) -> int:
        return int(self._length)

    def _source_index(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(f"Stage2 index {index} outside [0, {len(self)}).")
        if self.sample_indices is None:
            return int(index)
        return int(self.sample_indices[int(index)])

    def _load_token_cache(self, path: str | Path) -> dict[str, Any]:
        cache_path = Path(path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Stage 2 token cache not found: {cache_path}")
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            cache = torch.load(cache_path, map_location="cpu")
        if not isinstance(cache, dict):
            raise ValueError(f"Expected token cache to be a dict, got {type(cache).__name__}.")
        cache["path"] = str(cache_path)
        return cache

    def _validate_token_cache(self) -> None:
        assert self.token_cache is not None
        metadata = dict(self.token_cache.get("metadata", {}))
        if metadata.get("stage1_artifact_id") != self.stage1_artifact.artifact_id:
            raise ValueError(
                "Stage 2 token cache artifact mismatch: "
                f"cache={metadata.get('stage1_artifact_id')!r}, current={self.stage1_artifact.artifact_id!r}"
            )
        if metadata.get("stage1_checkpoint_sha256") is not None and metadata["stage1_checkpoint_sha256"] != self.stage1_artifact.checkpoint_sha256:
            raise ValueError(
                "Stage 2 token cache checkpoint hash mismatch: "
                f"cache={metadata['stage1_checkpoint_sha256']}, current={self.stage1_artifact.checkpoint_sha256}"
            )
        if int(metadata.get("token_dim", -1)) != self.token_dim:
            raise ValueError(f"Stage 2 token cache token_dim mismatch: cache={metadata.get('token_dim')}, current={self.token_dim}")
        tokens = self.token_cache.get("tokens")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[1] != self.token_dim:
            raise ValueError(f"Stage 2 token cache tokens must have shape [N, {self.token_dim}], got {getattr(tokens, 'shape', None)}.")
        if tokens.shape[0] > len(self.stage1_dataset):
            raise ValueError(f"Stage 2 token cache is longer than source dataset: cache={tokens.shape[0]}, source={len(self.stage1_dataset)}")
        cached_len = int(metadata.get("cached_len", tokens.shape[0]))
        if cached_len != int(tokens.shape[0]):
            raise ValueError(f"Stage 2 token cache cached_len mismatch: metadata={cached_len}, tokens={tokens.shape[0]}")
        source_dataset_len = metadata.get("source_dataset_len")
        if source_dataset_len is not None and int(source_dataset_len) != len(self.stage1_dataset):
            raise ValueError(
                "Stage 2 token cache source_dataset_len mismatch: "
                f"cache={source_dataset_len}, current={len(self.stage1_dataset)}"
            )
        self._length = int(tokens.shape[0])

    def _read_starvla_sample(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        dataset, trajectory_id, base_index, dataset_index = self.stage1_dataset._sample_location(index)
        raw_data = dataset.get_step_data(int(trajectory_id), int(base_index))
        transformed = dataset.transforms(raw_data)
        sample = dataset._pack_sample(transformed)

        metadata = {
            "dataset_name": dataset.dataset_name,
            "robot_tag": dataset.tag,
            "trajectory_id": int(trajectory_id),
            "base_index": int(base_index),
            "window_mode": self.stage1_dataset.window_mode,
            "stage1_artifact_id": self.stage1_artifact.artifact_id,
        }
        if dataset_index is not None:
            metadata["dataset_index"] = int(dataset_index)
        return sample, metadata

    @torch.no_grad()
    def _encode_actions(self, actions: np.ndarray | torch.Tensor) -> torch.LongTensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)
        actions = actions.to(device=next(self.tokenizer.parameters()).device, dtype=torch.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected actions with shape [T, D], got {tuple(actions.shape)}.")
        tokens = self.tokenizer.encode(actions.unsqueeze(0))[0].detach().cpu().long()
        if tokens.numel() != self.token_dim:
            raise ValueError(f"Expected {self.token_dim} Stage 1 tokens, got {tokens.numel()}.")
        return tokens

    def _resolve_read_index(self, index: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
        source_index = self._source_index(int(index))
        if not self.skip_bad_samples:
            sample, metadata = self._read_starvla_sample(source_index)
            metadata["source_index"] = int(source_index)
            return int(source_index), sample, metadata

        last_error: Exception | None = None
        for attempt in range(self.max_read_retries):
            subset_index = (int(index) + attempt) % len(self)
            read_index = self._source_index(subset_index)
            try:
                sample, metadata = self._read_starvla_sample(read_index)
                metadata["source_index"] = int(read_index)
                if attempt > 0:
                    metadata["requested_index"] = int(index)
                    metadata["replacement_subset_index"] = int(subset_index)
                    metadata["replacement_index"] = int(read_index)
                    metadata["replacement_attempt"] = int(attempt)
                    warnings.warn(
                        f"Skipped unreadable Stage2 sample {index}; using {read_index} after {attempt} retries. "
                        f"Last error: {type(last_error).__name__}: {last_error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return read_index, sample, metadata
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            f"Failed to read Stage2 sample {index} after {self.max_read_retries} retries. "
            f"Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def __getitem__(self, index: int) -> dict[str, Any]:
        read_index, sample, metadata = self._resolve_read_index(index)
        actions = np.asarray(sample["action"], dtype=np.float32)
        expected_shape = (self.action_spec.horizon, self.action_spec.action_dim)
        if actions.shape != expected_shape:
            raise ValueError(f"Action chunk shape mismatch: got {actions.shape}, expected {expected_shape}.")

        if self.token_cache is not None:
            cached_tokens = self.token_cache["tokens"][read_index].long()
            action_tokens = cached_tokens
            if self.validate_cache_online:
                encoded_tokens = self._encode_actions(actions)
                if not torch.equal(cached_tokens, encoded_tokens):
                    raise ValueError(
                        "Stage 2 token cache mismatch at index "
                        f"{read_index}: cached={cached_tokens[:5].tolist()}, encoded={encoded_tokens[:5].tolist()}"
                    )
        else:
            action_tokens = self._encode_actions(actions)
        sample["action"] = actions.astype(np.float32)
        sample["action_tokens"] = action_tokens
        sample["stage1_artifact_id"] = self.stage1_artifact.artifact_id
        sample["metadata"] = metadata

        if self.return_token_text:
            if self.token_text_codec is None:
                raise ValueError("return_token_text=True requires token_text_codec.")
            sample["action_token_text"] = self.token_text_codec.ids_to_text(action_tokens.tolist())

        return sample


def build_var_stage2_token_dataset_from_yaml(
    config_yaml: str | Path,
    *,
    stage1_artifact_path: str | Path,
    mode: str = "train",
    **kwargs: Any,
) -> VARStage2TokenDataset:
    """Build a Stage 2 token dataset from a Stage 1 YAML config."""

    cfg = OmegaConf.load(config_yaml)
    base_cfg = load_starvla_base_config(cfg)
    return VARStage2TokenDataset(
        base_cfg,
        stage1_artifact_path=stage1_artifact_path,
        mode=mode,
        balance_dataset_weights=bool(cfg.data.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(cfg.data.get("balance_trajectory_weights", False)),
        seed=int(cfg.experiment.get("seed", 42)),
        window_mode=str(cfg.data.get("window_mode", "full")),
        **kwargs,
    )
