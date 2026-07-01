"""Utilities for historical-frame sampling and optical-flow computation.

Adapted from DOMINO/policy/PUMA for use in the starVLA eval interface.
These helpers are **not** required when running with history disabled
(the default); they are loaded lazily the first time a history-enabled
``ModelClient`` is instantiated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2 as cv
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# History offset sampling
# ---------------------------------------------------------------------------

def sample_history_offsets(history_k: int, history_stride: int) -> list[int]:
    """Return ``history_k`` negative offsets spaced by ``history_stride``.

    Example: ``sample_history_offsets(3, 2)`` → ``[-6, -4, -2]``.
    """
    history_k = max(0, int(history_k))
    stride = max(1, int(history_stride))
    return [-(history_k - i) * stride for i in range(history_k)]


def parse_hw_size(
    size_value: Optional[Union[Sequence[int], str]],
    default_size: Tuple[int, int] = (128, 128),
) -> Tuple[int, int]:
    """Parse a ``[H, W]`` size from various input types."""
    if size_value is None:
        return default_size
    if isinstance(size_value, str):
        try:
            size_value = json.loads(size_value)
        except Exception:
            return default_size
    if isinstance(size_value, (list, tuple)) and len(size_value) == 2:
        try:
            h, w = int(size_value[0]), int(size_value[1])
        except (TypeError, ValueError):
            return default_size
        if h > 0 and w > 0:
            return (h, w)
    return default_size


# ---------------------------------------------------------------------------
# Optical flow  (Farneback)
# ---------------------------------------------------------------------------

def _ensure_rgb_uint8(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    else:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with shape [H, W, C], got {image.shape}")
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255.0 + 0.5).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    return image


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(ang * 0.5, 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    max_mag = float(np.percentile(mag, 99.0))
    if max_mag < 1e-6:
        hsv[..., 2] = 0
    else:
        hsv[..., 2] = np.clip((mag / max_mag) * 255.0, 0, 255).astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)


def compute_flow_rgb_farneback(
    prev_rgb: Union[np.ndarray, Image.Image],
    curr_rgb: Union[np.ndarray, Image.Image],
    compute_size: Tuple[int, int],
    farneback_cfg: Optional[dict] = None,
) -> np.ndarray:
    """Compute dense optical flow between two RGB images and return an RGB flow visualisation."""
    prev = _ensure_rgb_uint8(prev_rgb)
    curr = _ensure_rgb_uint8(curr_rgb)
    h, w = compute_size
    prev = cv.resize(prev, (w, h), interpolation=cv.INTER_AREA)
    curr = cv.resize(curr, (w, h), interpolation=cv.INTER_AREA)
    prev_gray = cv.cvtColor(prev, cv.COLOR_RGB2GRAY)
    curr_gray = cv.cvtColor(curr, cv.COLOR_RGB2GRAY)
    cfg = farneback_cfg or {}
    flow = cv.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=float(cfg.get("pyr_scale", 0.5)),
        levels=int(cfg.get("levels", 3)),
        winsize=int(cfg.get("winsize", 15)),
        iterations=int(cfg.get("iterations", 3)),
        poly_n=int(cfg.get("poly_n", 5)),
        poly_sigma=float(cfg.get("poly_sigma", 1.2)),
        flags=int(cfg.get("flags", 0)),
    )
    return _flow_to_rgb(flow)
