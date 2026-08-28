"""Utilities for consuming frozen VAR Stage 1 tokenizer artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from starVLA.model.modules.action_tokenizer.var_action_tokenizer import VARActionTokenizer
from starVLA.utils.action_spec import ActionSpec


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_manifest_and_checkpoint(path: str | Path) -> tuple[Path | None, Path, dict[str, Any]]:
    artifact_path = Path(path)
    if artifact_path.is_dir():
        manifest_path = artifact_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Stage 1 artifact directory has no manifest.json: {artifact_path}")
        manifest = _read_json(manifest_path)
        checkpoint_path = artifact_path / str(manifest.get("checkpoint", "checkpoint.ckpt"))
        return manifest_path, checkpoint_path, manifest

    if artifact_path.suffix == ".json":
        manifest_path = artifact_path
        manifest = _read_json(manifest_path)
        checkpoint_path = Path(str(manifest["checkpoint"]))
        if not checkpoint_path.is_absolute():
            checkpoint_path = manifest_path.parent / checkpoint_path
        return manifest_path, checkpoint_path, manifest

    return None, artifact_path, {}


def _validate_manifest(manifest: dict[str, Any], checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    if not manifest:
        return

    if manifest.get("checkpoint_sha256") is not None:
        actual = _sha256(checkpoint_path)
        if actual != manifest["checkpoint_sha256"]:
            raise ValueError(
                "Stage 1 checkpoint hash mismatch: "
                f"expected={manifest['checkpoint_sha256']}, actual={actual}, path={checkpoint_path}"
            )

    model_config = dict(checkpoint["model_config"])
    action_spec = ActionSpec.from_dict(checkpoint["action_spec"])
    product_codebook_groups = int(model_config.get("product_codebook_groups", 1))
    expected = {
        "action_dim": action_spec.action_dim,
        "action_horizon": action_spec.horizon,
        "token_dim": int(sum(model_config["scales"])) * product_codebook_groups,
        "codebook_size": int(model_config["codebook_size"]),
        "token_order": checkpoint.get("token_order", action_spec.token_order),
    }
    for key, actual_value in expected.items():
        if key in manifest and manifest[key] != actual_value:
            raise ValueError(f"Stage 1 manifest mismatch for {key}: manifest={manifest[key]!r}, checkpoint={actual_value!r}")

    if "scales" in manifest and list(manifest["scales"]) != list(model_config["scales"]):
        raise ValueError(f"Stage 1 manifest mismatch for scales: manifest={manifest['scales']}, checkpoint={model_config['scales']}")


@dataclass(frozen=True)
class Stage1Artifact:
    """Loaded metadata for a frozen Stage 1 tokenizer artifact."""

    tokenizer: VARActionTokenizer
    action_spec: ActionSpec
    checkpoint: dict[str, Any]
    checkpoint_path: Path
    manifest: dict[str, Any]
    manifest_path: Path | None = None

    @property
    def artifact_id(self) -> str:
        return str(self.manifest.get("artifact_id") or self.checkpoint_path.stem)

    @property
    def token_dim(self) -> int:
        return int(self.tokenizer.token_dim)

    @property
    def codebook_size(self) -> int:
        return int(self.tokenizer.codebook_size)

    @property
    def checkpoint_sha256(self) -> str:
        if self.manifest.get("checkpoint_sha256") is not None:
            return str(self.manifest["checkpoint_sha256"])
        return _sha256(self.checkpoint_path)


def load_frozen_var_action_tokenizer(
    artifact_or_checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    validate_manifest: bool = True,
) -> Stage1Artifact:
    """Load a frozen Stage 1 VAR action tokenizer.

    ``artifact_or_checkpoint_path`` may be either a checkpoint file, a
    ``manifest.json`` file, or an artifact directory containing ``manifest.json``.
    """

    manifest_path, checkpoint_path, manifest = _resolve_manifest_and_checkpoint(artifact_or_checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    checkpoint = _safe_torch_load(checkpoint_path)
    if validate_manifest:
        _validate_manifest(manifest, checkpoint, checkpoint_path)

    model = VARActionTokenizer(**dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return Stage1Artifact(
        tokenizer=model,
        action_spec=ActionSpec.from_dict(checkpoint["action_spec"]),
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        manifest=manifest,
        manifest_path=manifest_path,
    )
