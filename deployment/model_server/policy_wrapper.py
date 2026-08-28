# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

import numpy as np
import torch

from starVLA.model.framework.base_framework import baseframework, merge_config_overrides
from starVLA.model.framework.share_tools import read_mode_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


def _training_obs_image_size(model_cfg: Dict[str, Any]) -> Optional[List[int]]:
    """Return explicitly configured training image dimensions for eval checks."""
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    size = vla_data_cfg.get("obs_image_size") or vla_data_cfg.get("image_size")
    if size is None and "default_image_resolution" in vla_data_cfg:
        resolution = vla_data_cfg["default_image_resolution"]
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            size = resolution[-2:]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    return [int(size[0]), int(size[1])]


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        config_overrides: Sequence[str] | None = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path, config_overrides=config_overrides)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        # Co-located metadata.
        model_cfg, _ = read_mode_config(self._ckpt_path)
        model_cfg = merge_config_overrides(model_cfg, config_overrides)
        self._model_cfg = model_cfg
        self._action_mode, self._action_mode_apply_keys, self._action_mode_state_map = self._resolve_action_mode_metadata(model_cfg)

        self._action_chunk_size = self._resolve_action_chunk_size(model_cfg, framework)
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}
        self._predict_count = 0

        # Peek at available keys without building a full processor.
        _, _ns = read_mode_config(self._ckpt_path)
        self._available_unnorm_keys: List[str] = list(_ns.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s, action_mode=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
                self._action_mode,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )


    def _load_stage1_config_payload(self, stage1_cfg_path: str | None) -> Dict[str, Any]:
        if not stage1_cfg_path:
            return {}
        path = Path(stage1_cfg_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            logging.warning("PolicyServerWrapper: stage1_config not found for action_mode metadata: %s", path)
            return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _pick_first_config_value(*containers: Dict[str, Any], key: str) -> Any:
        for container in containers:
            if isinstance(container, dict) and container.get(key, None) is not None:
                return container[key]
        return None

    @staticmethod
    def _canonical_action_key(key: str) -> str:
        key = str(key)
        return key if key.startswith("action.") else f"action.{key}"

    @staticmethod
    def _canonical_state_key(key: str) -> str:
        key = str(key)
        return key if key.startswith("state.") else f"state.{key}"

    def _resolve_action_mode_metadata(self, model_cfg: Dict[str, Any]) -> tuple[str, Any, Dict[str, str]]:
        framework_cfg = model_cfg.get("framework", {}) or {}
        stage1_tokenizer_cfg = framework_cfg.get("stage1_tokenizer", {}) or {}
        dataset_cfg = ((model_cfg.get("datasets", {}) or {}).get("vla_data", {}) or {})
        stage1_cfg = self._load_stage1_config_payload(stage1_tokenizer_cfg.get("stage1_config"))
        stage1_data_cfg = (stage1_cfg.get("data", {}) or {}) if isinstance(stage1_cfg, dict) else {}
        stage1_dataset_cfg = (((stage1_cfg.get("datasets", {}) or {}).get("vla_data", {}) or {}) if isinstance(stage1_cfg, dict) else {})

        mode = self._pick_first_config_value(
            stage1_tokenizer_cfg,
            dataset_cfg,
            stage1_data_cfg,
            stage1_dataset_cfg,
            key="action_mode",
        )
        if mode is None:
            mode = "abs"
        mode = str(mode).strip().lower()
        if mode not in {"abs", "delta", "rel"}:
            logging.warning("PolicyServerWrapper: unknown action_mode=%r; treating it as abs", mode)
            mode = "abs"

        apply_keys = self._pick_first_config_value(
            stage1_tokenizer_cfg,
            dataset_cfg,
            stage1_data_cfg,
            stage1_dataset_cfg,
            key="action_mode_apply_keys",
        )
        if apply_keys is not None:
            apply_keys = [self._canonical_action_key(key) for key in apply_keys]

        state_map = self._pick_first_config_value(
            stage1_tokenizer_cfg,
            dataset_cfg,
            stage1_data_cfg,
            stage1_dataset_cfg,
            key="action_mode_state_map",
        )
        state_map = state_map or {}
        state_map = {self._canonical_action_key(k): self._canonical_state_key(v) for k, v in dict(state_map).items()}
        return mode, apply_keys, state_map


    def _resolve_action_chunk_size(self, model_cfg: Dict[str, Any], framework: baseframework) -> int:
        framework_cfg = model_cfg.get("framework", {})
        action_model_cfg = framework_cfg.get("action_model") or {}

        if "action_horizon" in action_model_cfg:
            return int(action_model_cfg["action_horizon"])
        if "future_action_window_size" in action_model_cfg:
            return int(action_model_cfg["future_action_window_size"]) + 1

        stage1_cfg_path = (framework_cfg.get("stage1_tokenizer") or {}).get("stage1_config")
        if stage1_cfg_path:
            path = Path(stage1_cfg_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    stage1_cfg = yaml.safe_load(f) or {}
                horizon = (stage1_cfg.get("data") or {}).get("expected_action_horizon")
                if horizon is not None:
                    return int(horizon)

        framework_horizon = getattr(framework, "action_horizon", None)
        if framework_horizon is not None:
            return int(framework_horizon)

        raise ValueError(
            "PolicyServerWrapper: could not resolve action chunk size from "
            "framework.action_model, framework.stage1_tokenizer.stage1_config, or framework.action_horizon "
            f"for {self._ckpt_path}"
        )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "action_mode": self._action_mode,
            "returned_action_mode": "abs",
            "action_mode_apply_keys": self._action_mode_apply_keys,
            "action_mode_state_map": self._action_mode_state_map,
            "training_data_mix": self._model_cfg.get("datasets", {}).get("vla_data", {}).get("data_mix"),
            "training_obs_image_size": _training_obs_image_size(self._model_cfg),
            "eval_image_contract": (
                "Eval clients must explicitly choose image count and order. "
                "The server does not infer or reorder camera views from training config."
            ),
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
            base["action_key_dims"] = proc.action_key_dims
            base["state_key_dims"] = proc.state_key_dims
        return base

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            **kwargs: forwarded to the framework's ``predict_action``
                (``do_sample``, ``use_ddim``, ``num_ddim_steps``, ...).

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        model_examples = self._prepare_examples_for_model(examples, proc)

        out = self._framework.predict_action(examples=model_examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        unnorm = self._convert_training_actions_to_env_actions(unnorm, examples, proc)
        self._predict_count += 1
        stats_every = int(os.environ.get("STARVLA_NORM_ACTION_STATS_EVERY", "0") or 0)
        if stats_every > 0 and self._predict_count % stats_every == 0:
            state_stats = None
            if model_examples and model_examples[0].get("state") is not None:
                state = np.asarray(model_examples[0]["state"], dtype=np.float32)
                state_stats = {
                    "shape": list(state.shape),
                    "min": float(np.nanmin(state)),
                    "max": float(np.nanmax(state)),
                    "mean": float(np.nanmean(state)),
                }
            logging.info(
                "policy_stats count=%d norm_shape=%s norm_min=%.6f norm_max=%.6f norm_mean=%.6f "
                "unnorm_shape=%s unnorm_min=%.6f unnorm_max=%.6f unnorm_mean=%.6f state=%s",
                self._predict_count,
                list(normalized.shape),
                float(np.nanmin(normalized)),
                float(np.nanmax(normalized)),
                float(np.nanmean(normalized)),
                list(unnorm.shape),
                float(np.nanmin(unnorm)),
                float(np.nanmax(unnorm)),
                float(np.nanmean(unnorm)),
                state_stats,
            )
        return {"actions": unnorm}

    def _split_current_raw_state_by_key(
        self,
        example: dict,
        proc: PolicyNormProcessor,
    ) -> Dict[str, np.ndarray]:
        if "state" not in example or example["state"] is None:
            raise ValueError(f"action_mode={self._action_mode!r} requires raw state in each example.")
        state = np.asarray(example["state"], dtype=np.float32)
        if state.ndim == 1:
            current_state = state
        elif state.ndim == 2:
            current_state = state[-1]
        else:
            raise ValueError(f"Expected raw state shape [D] or [T, D], got {state.shape}.")

        state_by_key: Dict[str, np.ndarray] = {}
        cursor = 0
        for state_key in proc.state_keys:
            dim = int(proc.state_key_dims.get(state_key, 1))
            state_by_key[state_key] = current_state[cursor : cursor + dim]
            cursor += dim
        if cursor != current_state.shape[-1]:
            raise ValueError(
                f"State dim mismatch while inverting {self._action_mode} actions: "
                f"metadata_dim={cursor}, state_dim={current_state.shape[-1]}, state_keys={proc.state_keys}"
            )
        return state_by_key

    def _convert_training_actions_to_env_actions(
        self,
        actions: np.ndarray,
        examples: List[dict],
        proc: PolicyNormProcessor,
    ) -> np.ndarray:
        if self._action_mode == "abs":
            return actions
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape [B, T, D], got {actions.shape}.")
        if len(examples) != actions.shape[0]:
            raise ValueError(f"Batch mismatch while inverting actions: examples={len(examples)}, actions={actions.shape[0]}.")

        converted = np.asarray(actions, dtype=np.float32).copy()
        apply_keys = set(self._action_mode_apply_keys or proc.action_keys)
        for batch_idx, example in enumerate(examples):
            state_by_key = self._split_current_raw_state_by_key(example, proc)
            cursor = 0
            for action_key in proc.action_keys:
                dim = int(proc.action_key_dims.get(action_key, 1))
                action_slice = slice(cursor, cursor + dim)
                if action_key in apply_keys:
                    state_key = self._action_mode_state_map.get(
                        action_key,
                        action_key.replace("action.", "state.", 1),
                    )
                    if state_key not in state_by_key:
                        raise ValueError(
                            f"Cannot invert {self._action_mode} action for {action_key}: "
                            f"missing {state_key} in state metadata."
                        )
                    state0 = state_by_key[state_key]
                    if state0.shape[-1] != dim:
                        raise ValueError(
                            f"State/action dim mismatch for {action_key}: "
                            f"state={state0.shape[-1]}, action={dim}."
                        )
                    if self._action_mode == "delta":
                        converted[batch_idx, :, action_slice] = (
                            np.cumsum(converted[batch_idx, :, action_slice], axis=0) + state0[None, :]
                        )
                    elif self._action_mode == "rel":
                        converted[batch_idx, :, action_slice] = converted[batch_idx, :, action_slice] + state0[None, :]
                cursor += dim
            if cursor != converted.shape[-1]:
                raise ValueError(
                    f"Action dim mismatch while inverting {self._action_mode} actions: "
                    f"metadata_dim={cursor}, action_dim={converted.shape[-1]}, action_keys={proc.action_keys}"
                )
        return converted

    def _prepare_examples_for_model(
        self,
        examples: List[dict],
        proc: PolicyNormProcessor,
    ) -> List[dict]:
        """Apply training-time input transforms before framework inference."""
        if not proc.state_keys:
            return examples

        prepared: List[dict] = []
        for example in examples:
            if "state" not in example or example["state"] is None:
                prepared.append(example)
                continue

            state = np.asarray(example["state"], dtype=np.float32)
            if state.ndim == 1:
                state = state[None, :]
            normalized_state = proc.apply_state(state)

            next_example = dict(example)
            next_example["state"] = normalized_state.astype(np.float32, copy=False)
            prepared.append(next_example)

        return prepared
