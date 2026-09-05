"""Randomness helpers shared by policy servers and evaluation clients."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed_everywhere(seed: int) -> None:
    """Seed the Python, NumPy, and PyTorch random number generators."""
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError(f"seed must be between 0 and {np.iinfo(np.uint32).max}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
