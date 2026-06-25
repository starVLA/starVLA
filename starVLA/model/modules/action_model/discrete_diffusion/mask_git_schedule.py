"""
MaskGIT-style masking schedule for discrete diffusion (ref: ref_dd_mode.py).
Training: rand_time -> mask_ratio -> num_mask positions.
Decode: ratio -> mask_ratio (fraction of unknown to keep masked).
"""

import torch

# In the input action token sequence, this value means "masked" (model uses MASK embedding).
IGNORE_TOKEN = -100


def train_mask_schedule(rand_time: torch.Tensor, method: str = "cosine") -> torch.Tensor:
    """
    Training-time mask schedule: rand_time in [0, 1) -> mask_ratio in (0, 1].
    How much of the maskable positions to mask; used as num_mask = round(total_unknown * mask_ratio).

    Args:
        rand_time: (B,) uniform in [0, 1)
        method: "cosine" or "linear"
    Returns:
        mask_ratio: (B,) in (0, 1]
    """
    if method == "cosine":
        return torch.clamp(1.0 - torch.cos(torch.pi * 0.5 * rand_time), 1e-6, 1.0)
    elif method == "linear":
        return torch.clamp(rand_time, 1e-6, 1.0)
    else:
        raise ValueError(f"Unknown train mask schedule: {method}")


def decode_mask_schedule(ratio: torch.Tensor, method: str = "cosine") -> torch.Tensor:
    """
    Decode-time schedule: ratio in [0, 1] -> mask_ratio (fraction of unknown to keep masked).
    As ratio increases (later in decode), mask_ratio decreases (more positions unmasked).

    Args:
        ratio: (B,) or scalar, progress in [0, 1]
        method: "cosine" or "linear"
    Returns:
        mask_ratio: (B,) or scalar
    """
    if method == "cosine":
        return 0.5 * (1.0 + torch.cos(torch.pi * ratio))
    elif method == "linear":
        return 1.0 - ratio
    else:
        raise ValueError(f"Unknown decode schedule: {method}")


def mask_by_random_topk(
    selected_probs: torch.Tensor,
    mask_len: torch.Tensor,
    temperature: float = 1.0,
    generator=None,
) -> torch.Tensor:
    """
    Which positions stay masked: Gumbel + top-k (ref: ref_dd_mode.py).
    selected_probs [B, L]; mask_len [B].
    Returns action_mask [B, L] with True = stay masked (low confidence / low prob).
    Score = -log(probs)/temp + gumbel; temperature adds stochasticity to ranking.
    """
    B, L = selected_probs.shape
    device = selected_probs.device

    gumbel = -torch.log(-torch.log(torch.rand(B, L, device=device, generator=generator) + 1e-10) + 1e-10)
    score = -torch.log(selected_probs + 1e-8) / max(temperature, 1e-8) + gumbel
    perm = torch.argsort(score, dim=1)
    ranks = torch.argsort(perm, dim=1)
    action_mask = ranks < mask_len.unsqueeze(1)
    return action_mask


def mask_by_deterministic_lowest(
    selected_probs: torch.Tensor,
    mask_len: torch.Tensor,
) -> torch.Tensor:
    """
    Which positions stay masked: deterministic = mask the mask_len lowest-confidence positions.
    selected_probs [B, L]; mask_len [B].
    Returns action_mask [B, L] with True = stay masked.
    """
    perm = torch.argsort(selected_probs, dim=1)  # ascending: smallest prob first
    ranks = torch.argsort(perm, dim=1)
    return ranks < mask_len.unsqueeze(1)
