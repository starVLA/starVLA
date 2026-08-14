"""Export a VAR Stage 1 checkpoint as a versioned artifact for Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


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


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_artifact(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {checkpoint_path}")

    checkpoint = _safe_torch_load(checkpoint_path)
    action_spec = dict(checkpoint["action_spec"])
    model_config = dict(checkpoint["model_config"])
    scales = list(model_config["scales"])
    product_codebook_groups = int(model_config.get("product_codebook_groups", 1))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_checkpoint = output_dir / "checkpoint.ckpt"
    shutil.copy2(checkpoint_path, exported_checkpoint)

    _write_json(output_dir / "action_spec.json", action_spec)
    _write_json(output_dir / "model_config.json", model_config)
    if "stage1_config" in checkpoint:
        OmegaConf.save(OmegaConf.create(checkpoint["stage1_config"]), output_dir / "stage1_config.yaml", resolve=True)

    optional_files = {
        "reconstruction_eval": args.reconstruction_eval,
        "oracle_replay_eval": args.oracle_replay_eval,
    }
    copied_reports = {}
    for key, source in optional_files.items():
        if source is None:
            continue
        source_path = Path(source)
        if source_path.exists():
            dest = output_dir / source_path.name
            shutil.copy2(source_path, dest)
            copied_reports[key] = dest.name

    manifest = {
        "artifact_id": args.artifact_id,
        "status": args.status,
        "checkpoint": exported_checkpoint.name,
        "checkpoint_sha256": _sha256(exported_checkpoint),
        "source_checkpoint": str(checkpoint_path),
        "action_dim": int(action_spec["action_dim"]),
        "action_horizon": int(action_spec["horizon"]),
        "action_keys": list(action_spec.get("action_keys", [])),
        "scales": scales,
        "quantization_mode": str(model_config.get("quantization_mode", "vq")),
        "product_codebook_groups": product_codebook_groups,
        "token_dim": int(sum(scales)) * product_codebook_groups,
        "codebook_size": int(model_config["codebook_size"]),
        "token_order": checkpoint.get("token_order", action_spec.get("token_order", "scale_major")),
        "model_config": "model_config.json",
        "action_spec": "action_spec.json",
        "reports": copied_reports,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a VAR Stage 1 checkpoint artifact for Stage 2.")
    parser.add_argument("--checkpoint", type=Path, default=Path("playground/Checkpoints/var_stage1_pi05_libero/best_recon.ckpt"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--artifact_id", type=str, required=True)
    parser.add_argument("--status", type=str, default="accepted_for_stage2")
    parser.add_argument("--reconstruction_eval", type=Path, default=Path("playground/Checkpoints/var_stage1_pi05_libero/reconstruction_eval.json"))
    parser.add_argument("--oracle_replay_eval", type=Path, default=None)
    args = parser.parse_args()

    manifest = export_artifact(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
