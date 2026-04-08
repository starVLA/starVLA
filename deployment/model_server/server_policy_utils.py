from pathlib import Path

import torch


def resolve_server_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")

    return requested_device


def build_server_metadata(vla, ckpt_path: str, device: str) -> dict:
    return {
        "env": "simpler_env",
        "device": device,
        "framework": vla.__class__.__name__,
        "checkpoint": Path(ckpt_path).name,
        "supports_vlm_cache": hasattr(vla, "get_inference_cache_stats"),
    }
