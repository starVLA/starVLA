"""Serve current StarVLA checkpoints through the vla-eval protocol."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from PIL import Image
from vla_eval.model_servers.base import SessionContext
from vla_eval.model_servers.predict import PredictModelServer
from vla_eval.specs import (
    GRIPPER_CLOSE_POS,
    IMAGE_RGB,
    LANGUAGE,
    POSITION_DELTA,
    RAW,
    ROTATION_AA,
    DimSpec,
)
from vla_eval.types import Action, Observation

logger = logging.getLogger(__name__)

ActionProfile = Literal["raw", "libero", "robotwin"]
GripperTransform = Literal["none", "open01_to_close_positive"]
ResizeMethod = Literal["bilinear", "area"]


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    matches = re.findall(r"\d+", path.stem)
    return (int(matches[-1]) if matches else -1, path.name)


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a local checkpoint or a Hugging Face repository to one weight file."""
    path = Path(checkpoint).expanduser()
    if path.is_file():
        if path.suffix not in {".pt", ".safetensors"}:
            raise ValueError(f"Unsupported checkpoint extension: {path}")
        # HF snapshot weights are symlinks into blobs; preserve the snapshot
        # path so PolicyServerWrapper can find its sibling config and stats.
        return _validate_checkpoint(path)

    if path.exists():
        root = path
    else:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover - part of the standard model environment
            raise RuntimeError("Install huggingface-hub to use a repository checkpoint ID") from exc
        logger.info("Downloading StarVLA checkpoint repository %s", checkpoint)
        root = Path(
            snapshot_download(
                checkpoint,
                allow_patterns=[
                    "config.yaml",
                    "dataset_statistics.json",
                    "checkpoints/*.pt",
                    "checkpoints/*.safetensors",
                ],
            )
        )

    candidates = [candidate for suffix in ("*.pt", "*.safetensors") for candidate in (root / "checkpoints").glob(suffix)]
    if not candidates:
        raise FileNotFoundError(f"No .pt or .safetensors file found under {root / 'checkpoints'}")
    return _validate_checkpoint(max(candidates, key=_checkpoint_sort_key))


def _validate_checkpoint(path: Path) -> str:
    for name in ("config.yaml", "dataset_statistics.json"):
        metadata_path = path.parent.parent / name
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint metadata: {metadata_path}")
    return str(path.absolute())


def _build_policy(
    checkpoint_path: str,
    *,
    use_bf16: bool,
    unnorm_key: str | None,
    config_overrides: list[str] | None,
) -> Any:
    # Keep the CLI and contract tests importable without loading torch and the
    # full StarVLA model stack. The actual server imports it during startup.
    from deployment.model_server.policy_wrapper import PolicyServerWrapper

    return PolicyServerWrapper(
        checkpoint_path,
        use_bf16=use_bf16,
        unnorm_key=unnorm_key,
        config_overrides=config_overrides,
    )


def _resize_image(image: Any, size: tuple[int, int] | None, method: ResizeMethod) -> Image.Image:
    pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    pil_image = pil_image.convert("RGB")
    if size is None or pil_image.size == (size[1], size[0]):
        return pil_image
    if method == "area":
        import cv2

        resized = cv2.resize(np.asarray(pil_image), (size[1], size[0]), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized)
    return pil_image.resize((size[1], size[0]), Image.Resampling.BILINEAR)


def postprocess_actions(
    actions: np.ndarray,
    *,
    action_indices: Sequence[int] | None = None,
    gripper_transform: GripperTransform = "none",
) -> np.ndarray:
    """Apply benchmark-facing transforms to an unnormalized action chunk."""
    result = np.asarray(actions, dtype=np.float32).copy()
    if result.ndim != 2:
        raise ValueError(f"Expected action chunk shaped (T, D), got {result.shape}")
    if action_indices is not None:
        result = result[:, list(action_indices)]
    if gripper_transform == "open01_to_close_positive":
        result[:, -1] = 1.0 - 2.0 * (result[:, -1] > 0.5)
    return result


class StarVLAHarnessServer(PredictModelServer):
    """vla-eval server backed by StarVLA's canonical checkpoint and transform code."""

    def __init__(
        self,
        checkpoint: str,
        *,
        base_vlm: str | None = None,
        unnorm_key: str | None = None,
        use_bf16: bool = False,
        config_overrides: list[str] | None = None,
        image_size: list[int] | None = None,
        image_keys: list[str] | None = None,
        resize_method: ResizeMethod = "bilinear",
        action_indices: list[int] | None = None,
        gripper_transform: GripperTransform = "none",
        action_profile: ActionProfile = "raw",
        observation_params: dict[str, Any] | None = None,
        max_batch_size: int = 1,
        max_wait_time: float = 0.01,
    ) -> None:
        checkpoint_path = resolve_checkpoint(checkpoint)
        effective_overrides = list(config_overrides or [])
        if base_vlm is not None:
            effective_overrides.append(f"framework.qwenvl.base_vlm={base_vlm}")
        self._policy = _build_policy(
            checkpoint_path,
            use_bf16=use_bf16,
            unnorm_key=unnorm_key,
            config_overrides=effective_overrides,
        )
        metadata = self._policy.metadata
        available_keys = metadata.get("available_unnorm_keys", [])
        effective_key = unnorm_key or metadata.get("default_unnorm_key")
        if effective_key is None and len(available_keys) > 1:
            raise ValueError(f"Checkpoint has multiple statistics keys; set unnorm_key to one of {available_keys}")
        if effective_key is not None and available_keys and effective_key not in available_keys:
            raise ValueError(f"Unknown unnorm_key={effective_key!r}; available={available_keys}")
        effective_chunk_size = int(metadata["action_chunk_size"])
        super().__init__(
            chunk_size=effective_chunk_size,
            max_batch_size=max_batch_size,
            max_wait_time=max_wait_time,
        )

        self._unnorm_key = effective_key
        self._image_size = tuple(image_size) if image_size is not None else None
        if self._image_size is not None and len(self._image_size) != 2:
            raise ValueError("image_size must be [height, width]")
        self._image_keys = list(image_keys) if image_keys else None
        self._resize_method = resize_method
        self._action_indices = action_indices
        self._gripper_transform = gripper_transform
        self._action_profile = action_profile
        self._observation_params = dict(observation_params or {})

        logger.info(
            "Loaded StarVLA checkpoint=%s chunk_size=%d cameras=%s unnorm_key=%s",
            checkpoint_path,
            effective_chunk_size,
            self._image_keys,
            unnorm_key or metadata.get("default_unnorm_key"),
        )

    def _make_example(self, obs: Observation) -> dict[str, Any]:
        image_source = obs.get("images")
        if isinstance(image_source, dict):
            keys = self._image_keys or list(image_source)
            missing = [key for key in keys if key not in image_source]
            if missing:
                raise KeyError(f"Missing configured cameras {missing}; available={list(image_source)}")
            images = [image_source[key] for key in keys]
        elif image_source is not None:
            if self._image_keys and len(self._image_keys) != 1:
                raise ValueError("Named image_keys require an image dictionary")
            images = list(image_source) if isinstance(image_source, (list, tuple)) else [image_source]
        else:
            raise KeyError("Observation has no 'images' field")

        example: dict[str, Any] = {
            "image": [_resize_image(image, self._image_size, self._resize_method) for image in images],
            "lang": str(obs.get("task_description", "")),
        }
        return example

    def predict_batch(self, obs_batch: list[Observation], ctx_batch: list[SessionContext]) -> list[Action]:
        del ctx_batch
        examples = [self._make_example(obs) for obs in obs_batch]
        result = self._policy.predict_action(
            examples=examples,
            unnorm_key=self._unnorm_key,
        )
        action_batch = np.asarray(result["actions"])
        if action_batch.ndim != 3 or action_batch.shape[0] != len(obs_batch):
            raise ValueError(f"Expected actions shaped (B, T, D) for B={len(obs_batch)}, got {action_batch.shape}")
        if action_batch.shape[1] < self.chunk_size:
            raise ValueError(f"Policy returned fewer actions than chunk_size={self.chunk_size}")
        return [
            {
                "actions": postprocess_actions(
                    action_batch[index],
                    action_indices=self._action_indices,
                    gripper_transform=self._gripper_transform,
                )
            }
            for index in range(len(obs_batch))
        ]

    def get_observation_params(self) -> dict[str, Any]:
        return dict(self._observation_params)

    def get_action_spec(self) -> dict[str, DimSpec]:
        if self._action_profile == "libero":
            return {
                "position": POSITION_DELTA,
                "rotation": ROTATION_AA,
                "gripper": GRIPPER_CLOSE_POS,
            }
        if self._action_profile == "robotwin":
            return {"joints": DimSpec("joints", 14, "joint_positions")}
        return {"action": RAW}

    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec = {key: IMAGE_RGB for key in self._image_keys or ["image"]}
        spec["language"] = LANGUAGE
        return spec


if __name__ == "__main__":
    from vla_eval.model_servers.serve import run_server

    run_server(StarVLAHarnessServer)
