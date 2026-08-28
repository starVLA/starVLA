"""VQ-VLA-style residual VQ-VAE action tokenizer.

This module is intentionally separate from ``VARActionTokenizer`` so that the
VQ-VLA ablation compares a distinct tokenizer design rather than another mode
inside the existing VAR/Product-VQ implementation.

The implementation follows the tokenizer described in VQ-VLA as closely as the
paper specifies: an action-only convolutional encoder maps a fixed action chunk
to one latent vector, residual vector quantization represents that latent with
``Nq`` codebook indices, and a convolutional decoder reconstructs the full
action chunk from the quantized latent.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    def __init__(self, dim: int, *, dilation: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VQVLARVQActionTokenizer(nn.Module):
    """Convolutional residual VQ-VAE for fixed-horizon action chunks."""

    def __init__(
        self,
        *,
        action_dim: int,
        seq_len: int,
        codebook_size: int = 512,
        embed_dim: int = 32,
        residual_vq_layers: int = 4,
        commitment_cost: float = 0.25,
        codebook_loss_weight: float = 1.0,
        normalize_codebook_for_lookup: bool = False,
        use_dilated: bool = True,
        dim_groups: dict[str, list[int]] | None = None,
        use_time_embedding: bool = True,
        use_action_type_embedding: bool = True,
        input_embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_type = "vqvla_rvq_action_tokenizer"
        self.action_dim = int(action_dim)
        self.seq_len = int(seq_len)
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.residual_vq_layers = int(residual_vq_layers)
        self.commitment_cost = float(commitment_cost)
        self.codebook_loss_weight = float(codebook_loss_weight)
        self.normalize_codebook_for_lookup = bool(normalize_codebook_for_lookup)
        self.use_dilated = bool(use_dilated)
        self.dim_groups = {name: list(dims) for name, dims in (dim_groups or {}).items()}
        self.use_time_embedding = bool(use_time_embedding)
        self.use_action_type_embedding = bool(use_action_type_embedding)
        self.input_embedding_scale = float(input_embedding_scale)
        self.scales = [self.residual_vq_layers]

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if self.residual_vq_layers <= 0:
            raise ValueError(f"residual_vq_layers must be positive, got {self.residual_vq_layers}.")

        self.input_proj = nn.Linear(self.action_dim, self.embed_dim)
        if self.use_time_embedding:
            self.register_buffer("encoder_time_embedding", self._build_sinusoidal_time_embedding(), persistent=False)
        else:
            self.encoder_time_embedding = None
        if self.use_action_type_embedding:
            self.register_buffer("action_type_ids", self._build_action_type_ids(), persistent=False)
            self.encoder_action_type_embedding = nn.Embedding(4, self.embed_dim)
            nn.init.normal_(self.encoder_action_type_embedding.weight, mean=0.0, std=self.embed_dim**-0.5)
        else:
            self.register_buffer("action_type_ids", torch.zeros(self.action_dim, dtype=torch.long), persistent=False)
            self.encoder_action_type_embedding = None

        if self.use_dilated:
            self.encoder_res = nn.Sequential(
                ResBlock1D(self.embed_dim, dilation=1),
                ResBlock1D(self.embed_dim, dilation=2),
                ResBlock1D(self.embed_dim, dilation=4),
                ResBlock1D(self.embed_dim, dilation=8),
            )
        else:
            self.encoder_res = nn.Sequential(*[ResBlock1D(self.embed_dim) for _ in range(4)])
        self.encoder_norm = nn.GroupNorm(1, self.embed_dim)
        self.encoder_out = nn.Linear(self.embed_dim, self.embed_dim)

        self.codebooks = nn.ModuleList(
            [nn.Embedding(self.codebook_size, self.embed_dim) for _ in range(self.residual_vq_layers)]
        )
        for codebook in self.codebooks:
            nn.init.normal_(codebook.weight, mean=0.0, std=self.embed_dim**-0.5)

        self.decoder_seed = nn.Linear(self.embed_dim, self.seq_len * self.embed_dim)
        self.decoder_in = nn.ConvTranspose1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)
        self.decoder_res = nn.Sequential(*[ResBlock1D(self.embed_dim) for _ in range(3)])
        self.decoder_out = nn.Sequential(
            nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, self.embed_dim),
            nn.GELU(),
            nn.Conv1d(self.embed_dim, self.action_dim, kernel_size=3, padding=1),
        )

    @property
    def token_dim(self) -> int:
        return self.residual_vq_layers

    def _build_sinusoidal_time_embedding(self) -> torch.Tensor:
        positions = torch.arange(self.seq_len, dtype=torch.float32).unsqueeze(1)
        half_dim = max(1, self.embed_dim // 2)
        div_term = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000.0) / max(1, half_dim - 1)))
        emb = torch.zeros(self.seq_len, self.embed_dim, dtype=torch.float32)
        emb[:, 0 : 2 * half_dim : 2] = torch.sin(positions * div_term)[:, : emb[:, 0::2].shape[1]]
        emb[:, 1 : 2 * half_dim : 2] = torch.cos(positions * div_term)[:, : emb[:, 1::2].shape[1]]
        return emb

    def _build_action_type_ids(self) -> torch.LongTensor:
        type_ids = torch.zeros(self.action_dim, dtype=torch.long)
        for dim in self.dim_groups.get("position", []):
            type_ids[int(dim)] = 1
        for dim in self.dim_groups.get("rotation", []):
            type_ids[int(dim)] = 2
        for dim in self.dim_groups.get("gripper", []):
            type_ids[int(dim)] = 3
        return type_ids

    def get_config(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
            "codebook_size": self.codebook_size,
            "embed_dim": self.embed_dim,
            "residual_vq_layers": self.residual_vq_layers,
            "commitment_cost": self.commitment_cost,
            "codebook_loss_weight": self.codebook_loss_weight,
            "normalize_codebook_for_lookup": self.normalize_codebook_for_lookup,
            "use_dilated": self.use_dilated,
            "dim_groups": self.dim_groups,
            "use_time_embedding": self.use_time_embedding,
            "use_action_type_embedding": self.use_action_type_embedding,
            "input_embedding_scale": self.input_embedding_scale,
        }

    def encode_features(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[1] != self.seq_len or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected actions with shape [B, {self.seq_len}, {self.action_dim}], got {tuple(actions.shape)}."
            )
        h_t = self.input_proj(actions)
        if self.encoder_time_embedding is not None:
            h_t = h_t + self.input_embedding_scale * self.encoder_time_embedding.to(actions.device).unsqueeze(0)
        if self.encoder_action_type_embedding is not None:
            type_embeddings = self.encoder_action_type_embedding(self.action_type_ids.to(actions.device))
            type_features = torch.einsum("btd,de->bte", actions, type_embeddings)
            h_t = h_t + self.input_embedding_scale * type_features / math.sqrt(float(self.action_dim))
        h = h_t.transpose(1, 2).contiguous()
        h = self.encoder_norm(self.encoder_res(h))
        pooled = h.mean(dim=-1)
        return self.encoder_out(pooled)

    def _nearest(self, residual: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalize_codebook_for_lookup:
            residual_lookup = F.normalize(residual, dim=-1, eps=1e-6)
            codebook_lookup = F.normalize(codebook, dim=-1, eps=1e-6)
            indices = torch.argmax(torch.matmul(residual_lookup, codebook_lookup.t()), dim=-1)
        else:
            r_sq = torch.sum(residual**2, dim=-1, keepdim=True)
            c_sq = torch.sum(codebook**2, dim=-1)
            rc = torch.matmul(residual, codebook.t())
            indices = torch.argmin(r_sq + c_sq - 2.0 * rc, dim=-1)
        return indices, codebook.index_select(0, indices)

    def quantize_features(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2:
            raise ValueError(f"Expected chunk latent with shape [B, {self.embed_dim}], got {tuple(features.shape)}.")
        residual = features
        quantized_sum = torch.zeros_like(features)
        indices_layers = []
        total_vq_loss = torch.zeros((), dtype=features.dtype, device=features.device)

        for codebook_module in self.codebooks:
            indices, z_q = self._nearest(residual, codebook_module.weight)
            codebook_loss = F.mse_loss(z_q, residual.detach())
            commitment_loss = F.mse_loss(z_q.detach(), residual)
            total_vq_loss = total_vq_loss + self.codebook_loss_weight * codebook_loss + self.commitment_cost * commitment_loss
            quantized_sum = quantized_sum + z_q
            residual = residual - z_q
            indices_layers.append(indices)

        quantized_ste = features + (quantized_sum - features).detach()
        token_ids = torch.stack(indices_layers, dim=-1)
        return quantized_ste, token_ids, total_vq_loss

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected chunk latent with shape [B, {self.embed_dim}], got {tuple(features.shape)}.")
        d = self.decoder_seed(features).view(features.shape[0], self.seq_len, self.embed_dim).transpose(1, 2).contiguous()
        d = self.decoder_in(d)
        d = self.decoder_res(d)
        return self.decoder_out(d).transpose(1, 2).contiguous()

    def flatten_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.reshape(token_ids.shape[0], -1)

    def unflatten_tokens(self, flat_tokens: torch.Tensor) -> torch.Tensor:
        if flat_tokens.ndim == 1:
            flat_tokens = flat_tokens.unsqueeze(0)
        if flat_tokens.ndim != 2 or flat_tokens.shape[1] != self.token_dim:
            raise ValueError(f"Expected flat tokens with shape [B, {self.token_dim}], got {tuple(flat_tokens.shape)}.")
        return flat_tokens.long().reshape(flat_tokens.shape[0], self.residual_vq_layers)

    @torch.no_grad()
    def encode(self, actions: torch.Tensor) -> torch.LongTensor:
        features = self.encode_features(actions)
        _, token_ids, _ = self.quantize_features(features)
        return self.flatten_tokens(token_ids)

    @torch.no_grad()
    def decode(self, token_ids: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(token_ids, list):
            token_ids = token_ids[0]
        token_ids = self.unflatten_tokens(token_ids)
        layer_features = []
        for layer_idx, codebook in enumerate(self.codebooks):
            layer_features.append(codebook(token_ids[:, layer_idx].long()))
        features = torch.stack(layer_features, dim=0).sum(dim=0)
        return self.decode_features(features)

    def forward(self, actions: torch.Tensor, *, return_latent: bool = False) -> dict[str, Any]:
        features = self.encode_features(actions)
        quantized, token_ids, vq_loss = self.quantize_features(features)
        recon = self.decode_features(quantized)
        output: dict[str, Any] = {
            "recon": recon,
            "token_ids": [token_ids],
            "flat_token_ids": self.flatten_tokens(token_ids),
            "vq_loss": vq_loss,
            "aux": {
                "token_dim": self.token_dim,
                "scales": list(self.scales),
            },
        }
        if return_latent:
            output["latent"] = features
        return output
