"""
Uniform action binning for discrete diffusion.
Aligns with ref_dd_mode: floor-based encode, bin centers for decode.
Maps continuous actions in [low, high] to discrete bins in [0, num_bins-1].

Representation modes:
- "bin": model outputs num_bins logits per position; decode via argmax.
- "bit": model outputs 8 logits (P(bit_i=1) via sigmoid); 8 bits simulate 256-bin
  distribution. decode via threshold (0.5) or sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def continuous_to_bins(actions: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Map continuous actions in [low, high] to bin indices in [0, num_bins-1]. Ref: floor((x+1)/2 * num_bins)."""
    clamped = actions.clamp(low + 1e-6, high - 1e-6)
    x = (clamped - low) / (high - low)  # [0, 1)
    bins = (x * num_bins).floor().long().clamp(0, num_bins - 1)
    return bins


def bins_to_continuous(bin_indices: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Map bin indices to continuous actions (bin centers). Ref: (i+0.5)/num_bins * 2 - 1 for [-1,1]."""
    scale = (high - low) / num_bins
    centers = (bin_indices.float() + 0.5) * scale + low
    return centers


class ActionBinning(nn.Module):
    """
    Discretize continuous actions via uniform binning.
    Supports two representations:
    - "bin": model outputs num_bins logits per position; argmax to get bin index.
    - "bit": model outputs 8 logits (P(bit_i=1)); bits form bin index in [0,255].
    """

    def __init__(
        self,
        num_bins: int,
        action_dim: int,
        low: float = -1.0,
        high: float = 1.0,
        representation: str = "bin",
        num_bits: int = 8,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.low = low
        self.high = high
        self.representation = representation
        self.num_bits = num_bits

        if representation == "bit":
            expected_bins = 1 << num_bits
            assert num_bins == expected_bins, f"bit representation requires num_bins={expected_bins}, got {num_bins}"

        # Bin centers for decoding
        scale = (high - low) / num_bins
        centers = (torch.arange(num_bins, dtype=torch.float32) + 0.5) * scale + low
        self.register_buffer("bin_centers", centers)

        # Powers of 2 for bit->bin conversion
        if representation == "bit":
            self.register_buffer("bit_powers", torch.arange(num_bits, dtype=torch.long))

    @property
    def logits_dim(self) -> int:
        """Per-position model output dimension (last dim of logits)."""
        return self.num_bits if self.representation == "bit" else self.num_bins

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode continuous actions to discrete bin indices. (B, T, action_dim) -> (B, T, action_dim) long."""
        return continuous_to_bins(actions, self.num_bins, self.low, self.high)

    def decode(self, indices: torch.Tensor, logging: bool = False) -> torch.Tensor:
        """Decode discrete bin indices to continuous actions (bin centers)."""
        flat = indices.reshape(-1)
        decoded = self.bin_centers[flat].to(indices.device)
        B = indices.shape[0]
        output = decoded.reshape(B, -1, self.action_dim)
        if logging:
            print(f"indices output shape: {indices.shape}")
            print(f"indices output first batch first action: {indices[0, :]}")
        return output

    def _logits_to_indices_bin(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (indices, selected_probs)."""
        safe_temp = max(temperature, 1e-8)
        probs = F.softmax(logits / safe_temp, dim=-1)
        if deterministic:
            indices = logits.argmax(dim=-1)
            selected_probs = probs.max(dim=-1).values
        else:
            flat = probs.reshape(-1, self.num_bins)
            sampled_flat = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
            indices = sampled_flat.reshape(*logits.shape[:-1])
            selected_probs = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return indices, selected_probs

    def _logits_to_indices_bit(
        self,
        logits: torch.Tensor,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert 8 bit logits to bin indices. Returns (indices, selected_probs)."""
        probs = logits.sigmoid()
        if deterministic:
            bits = (probs > 0.5).long()
        else:
            bits = (torch.rand_like(probs, generator=generator) < probs).long()
        powers = 2 ** self.bit_powers.to(logits.device)
        indices = (bits * powers).sum(dim=-1)
        # P(bin) under factorized model = prod_i (p_i if b_i else (1-p_i))
        chosen = probs * bits + (1 - probs) * (1 - bits)
        selected_probs = chosen.prod(dim=-1)
        return indices, selected_probs

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode model logits to continuous actions. Same interface for bin and bit."""
        if self.representation == "bin":
            indices, _ = self._logits_to_indices_bin(logits, temperature=1.0, deterministic=True)
        else:
            indices, _ = self._logits_to_indices_bit(logits, deterministic=True)
        return self.decode(indices)

    def sample_indices_from_logits(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample bin indices from logits (for iterative decode).
        Returns (indices, selected_probs) where selected_probs is used for mask scheduling.
        """
        if self.representation == "bin":
            return self._logits_to_indices_bin(
                logits, temperature=temperature, deterministic=deterministic, generator=generator
            )
        return self._logits_to_indices_bit(logits, deterministic=deterministic, generator=generator)

    def indices_to_bit_targets(self, indices: torch.Tensor) -> torch.Tensor | None:
        """
        Convert bin indices to 8-bit targets for BCE loss. Returns None for bin representation.
        (B, T, D) long -> (B, T, D, 8) float in {0,1}.
        """
        if self.representation != "bit":
            return None
        # bit_i = (indices >> i) & 1
        shifts = self.bit_powers.to(indices.device)
        bits = ((indices.unsqueeze(-1) >> shifts) & 1).float()
        return bits
