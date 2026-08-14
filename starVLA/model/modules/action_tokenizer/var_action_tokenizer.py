"""VAR-style multi-scale residual VQ tokenizer for action chunks."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_scales(seq_len: int) -> list[int]:
    """Return power-of-two scales ending at ``seq_len``.

    Examples:
        seq_len=8  -> [1, 2, 4, 8]
        seq_len=50 -> [1, 2, 4, 8, 16, 32, 50]
    """

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    scales = []
    scale = 1
    while scale < seq_len:
        scales.append(scale)
        scale *= 2
    if not scales or scales[-1] != seq_len:
        scales.append(seq_len)
    return scales


class Phi1D(nn.Module):
    def __init__(self, embed_dim: int, quant_resi: float = 0.5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.resi_ratio = float(quant_resi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - self.resi_ratio) + self.conv(x) * self.resi_ratio


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


class VARActionTokenizer(nn.Module):
    """Action tokenizer with shared-codebook multi-scale residual VQ.

    The model consumes normalized action chunks shaped ``[B, T, D]`` and returns
    reconstructed normalized actions plus scale-major token ids.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        seq_len: int,
        codebook_size: int = 512,
        embed_dim: int = 128,
        scales: list[int] | tuple[int, ...] | None = None,
        use_dilated: bool = True,
        quant_resi: float = 0.5,
        commitment_cost: float = 0.25,
        normalize_codebook_for_lookup: bool = True,
        decoder_head_type: str = "plain",
        quantization_mode: str = "vq",
        product_codebook_groups: int = 1,
        dim_groups: dict[str, list[int]] | None = None,
        use_time_embedding: bool = False,
        use_action_type_embedding: bool = False,
        input_embedding_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.seq_len = int(seq_len)
        self.codebook_size = int(codebook_size)
        self.embed_dim = int(embed_dim)
        self.scales = list(scales) if scales is not None else default_scales(self.seq_len)
        self.use_dilated = bool(use_dilated)
        self.quant_resi = float(quant_resi)
        self.commitment_cost = float(commitment_cost)
        self.normalize_codebook_for_lookup = bool(normalize_codebook_for_lookup)
        self.decoder_head_type = str(decoder_head_type)
        self.quantization_mode = str(quantization_mode)
        self.product_codebook_groups = int(product_codebook_groups)
        self.dim_groups = {name: list(dims) for name, dims in (dim_groups or {}).items()}
        self.use_time_embedding = bool(use_time_embedding)
        self.use_action_type_embedding = bool(use_action_type_embedding)
        self.input_embedding_scale = float(input_embedding_scale)

        if self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {self.action_dim}.")
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}.")
        if not self.scales:
            raise ValueError("At least one scale is required.")
        if max(self.scales) > self.seq_len or min(self.scales) <= 0:
            raise ValueError(f"Scales must be in [1, seq_len]. Got scales={self.scales}, seq_len={self.seq_len}.")
        if self.decoder_head_type not in {"plain", "grouped"}:
            raise ValueError(f"Unsupported decoder_head_type={self.decoder_head_type!r}.")
        if self.quantization_mode not in {"vq", "none", "product_vq"}:
            raise ValueError(f"Unsupported quantization_mode={self.quantization_mode!r}.")
        if self.quantization_mode == "product_vq":
            if self.product_codebook_groups <= 0:
                raise ValueError(f"product_codebook_groups must be positive, got {self.product_codebook_groups}.")
            if self.embed_dim % self.product_codebook_groups != 0:
                raise ValueError(
                    f"embed_dim={self.embed_dim} must be divisible by "
                    f"product_codebook_groups={self.product_codebook_groups}."
                )
        if self.decoder_head_type == "grouped":
            self._validate_dim_groups()

        self.encoder_in = nn.Conv1d(self.action_dim, self.embed_dim, kernel_size=3, padding=1)
        if self.use_time_embedding:
            self.encoder_time_embedding = nn.Embedding(self.seq_len, self.embed_dim)
            nn.init.normal_(self.encoder_time_embedding.weight, mean=0.0, std=self.embed_dim**-0.5)
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
            self.encoder_res = nn.Sequential(*[ResBlock1D(self.embed_dim) for _ in range(6)])
        self.encoder_norm = nn.GroupNorm(1, self.embed_dim)

        self.shared_codebook = nn.Embedding(self.codebook_size, self.embed_dim)
        nn.init.normal_(self.shared_codebook.weight, mean=0.0, std=self.embed_dim**-0.5)
        product_code_dim = self.embed_dim // max(1, self.product_codebook_groups)
        self.product_codebooks = nn.ModuleList(
            [nn.Embedding(self.codebook_size, product_code_dim) for _ in range(self.product_codebook_groups)]
        )
        for codebook in self.product_codebooks:
            nn.init.normal_(codebook.weight, mean=0.0, std=product_code_dim**-0.5)
        self.phi_layers = nn.ModuleList([Phi1D(self.embed_dim, quant_resi=self.quant_resi) for _ in self.scales])

        self.decoder_in = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1)
        self.decoder_res = nn.Sequential(*[ResBlock1D(self.embed_dim) for _ in range(3)])
        if self.decoder_head_type == "plain":
            self.decoder_out = nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.embed_dim),
                nn.GELU(),
                nn.Conv1d(self.embed_dim, self.action_dim, kernel_size=3, padding=1),
            )
        else:
            self.decoder_shared_out = nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
                nn.GroupNorm(1, self.embed_dim),
                nn.GELU(),
            )
            self.decoder_group_heads = nn.ModuleDict(
                {
                    name: nn.Conv1d(self.embed_dim, len(dims), kernel_size=3, padding=1)
                    for name, dims in self.dim_groups.items()
                }
            )

    @property
    def token_dim(self) -> int:
        groups = self.product_codebook_groups if self.quantization_mode == "product_vq" else 1
        return int(sum(self.scales) * groups)

    def _validate_dim_groups(self) -> None:
        seen: list[int] = []
        for name, dims in self.dim_groups.items():
            if not dims:
                raise ValueError(f"Grouped decoder head {name!r} has no dims.")
            for dim in dims:
                if dim < 0 or dim >= self.action_dim:
                    raise ValueError(f"Grouped decoder dim {dim} is outside action_dim={self.action_dim}.")
                seen.append(dim)
        if sorted(seen) != list(range(self.action_dim)):
            raise ValueError(
                "Grouped decoder dim_groups must cover every action dim exactly once. "
                f"Got {self.dim_groups} for action_dim={self.action_dim}."
            )

    def get_config(self) -> dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "seq_len": self.seq_len,
            "codebook_size": self.codebook_size,
            "embed_dim": self.embed_dim,
            "scales": self.scales,
            "use_dilated": self.use_dilated,
            "quant_resi": self.quant_resi,
            "commitment_cost": self.commitment_cost,
            "normalize_codebook_for_lookup": self.normalize_codebook_for_lookup,
            "decoder_head_type": self.decoder_head_type,
            "quantization_mode": self.quantization_mode,
            "product_codebook_groups": self.product_codebook_groups,
            "dim_groups": self.dim_groups,
            "use_time_embedding": self.use_time_embedding,
            "use_action_type_embedding": self.use_action_type_embedding,
            "input_embedding_scale": self.input_embedding_scale,
        }

    def _build_action_type_ids(self) -> torch.LongTensor:
        """Map action dimensions to coarse physical types.

        Type ids are: 0=unknown, 1=position, 2=rotation, 3=gripper.
        """

        type_ids = torch.zeros(self.action_dim, dtype=torch.long)
        for dim in self.dim_groups.get("position", []):
            type_ids[int(dim)] = 1
        for dim in self.dim_groups.get("rotation", []):
            type_ids[int(dim)] = 2
        for dim in self.dim_groups.get("gripper", []):
            type_ids[int(dim)] = 3
        return type_ids

    def encode_features(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3 or actions.shape[1] != self.seq_len or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected actions with shape [B, {self.seq_len}, {self.action_dim}], "
                f"got {tuple(actions.shape)}."
            )
        h = self.encoder_in(actions.transpose(1, 2))
        if self.encoder_time_embedding is not None:
            positions = torch.arange(self.seq_len, device=actions.device)
            time_features = self.encoder_time_embedding(positions).transpose(0, 1).unsqueeze(0)
            h = h + self.input_embedding_scale * time_features
        if self.encoder_action_type_embedding is not None:
            type_embeddings = self.encoder_action_type_embedding(self.action_type_ids.to(actions.device))
            type_features = torch.einsum("btd,de->bet", actions, type_embeddings)
            type_features = type_features / math.sqrt(float(self.action_dim))
            h = h + self.input_embedding_scale * type_features
        h = self.encoder_res(h)
        return self.encoder_norm(h)

    def quantize(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, length = z.shape
        z_flat = z.permute(0, 2, 1).reshape(-1, channels)
        codebook = self.shared_codebook.weight

        if self.normalize_codebook_for_lookup:
            z_lookup = F.normalize(z_flat, dim=-1, eps=1e-6)
            codebook_lookup = F.normalize(codebook, dim=-1, eps=1e-6)
            indices = torch.argmax(torch.matmul(z_lookup, codebook_lookup.t()), dim=-1)
        else:
            z_sq = torch.sum(z_flat**2, dim=-1, keepdim=True)
            c_sq = torch.sum(codebook**2, dim=-1)
            zc = torch.matmul(z_flat, codebook.t())
            indices = torch.argmin(z_sq + c_sq - 2.0 * zc, dim=-1)

        z_q = codebook.index_select(0, indices)
        codebook_loss = F.mse_loss(z_q, z_flat.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z_flat)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        z_q_ste = z_flat + (z_q - z_flat).detach()
        z_q_ste = z_q_ste.view(batch_size, length, channels).permute(0, 2, 1).contiguous()
        return z_q_ste, indices.view(batch_size, length), vq_loss

    def quantize_product(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, length = z.shape
        group_dim = channels // self.product_codebook_groups
        z_groups = z.permute(0, 2, 1).reshape(batch_size * length, self.product_codebook_groups, group_dim)
        z_q_groups = []
        indices_groups = []
        total_vq_loss = torch.zeros((), device=z.device, dtype=z.dtype)

        for group_idx, codebook_module in enumerate(self.product_codebooks):
            z_group = z_groups[:, group_idx, :]
            codebook = codebook_module.weight
            if self.normalize_codebook_for_lookup:
                z_lookup = F.normalize(z_group, dim=-1, eps=1e-6)
                codebook_lookup = F.normalize(codebook, dim=-1, eps=1e-6)
                indices = torch.argmax(torch.matmul(z_lookup, codebook_lookup.t()), dim=-1)
            else:
                z_sq = torch.sum(z_group**2, dim=-1, keepdim=True)
                c_sq = torch.sum(codebook**2, dim=-1)
                zc = torch.matmul(z_group, codebook.t())
                indices = torch.argmin(z_sq + c_sq - 2.0 * zc, dim=-1)

            z_q = codebook.index_select(0, indices)
            codebook_loss = F.mse_loss(z_q, z_group.detach())
            commitment_loss = F.mse_loss(z_q.detach(), z_group)
            total_vq_loss = total_vq_loss + codebook_loss + self.commitment_cost * commitment_loss
            z_q_groups.append(z_group + (z_q - z_group).detach())
            indices_groups.append(indices.view(batch_size, length))

        z_q_ste = torch.stack(z_q_groups, dim=1).reshape(batch_size, length, channels)
        z_q_ste = z_q_ste.permute(0, 2, 1).contiguous()
        indices_tensor = torch.stack(indices_groups, dim=-1)
        return z_q_ste, indices_tensor, total_vq_loss

    def quantize_latent(self, latent_full: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        f_rest = latent_full.clone()
        f_hat = torch.zeros_like(latent_full)
        total_vq_loss = torch.zeros((), device=latent_full.device, dtype=latent_full.dtype)
        indices_list: list[torch.Tensor] = []

        for scale_idx, scale in enumerate(self.scales):
            z_scale = F.interpolate(f_rest, size=scale, mode="linear", align_corners=False)
            if self.quantization_mode == "product_vq":
                z_q_scale, indices, vq_loss = self.quantize_product(z_scale)
            else:
                z_q_scale, indices, vq_loss = self.quantize(z_scale)
            upsampled = F.interpolate(z_q_scale, size=self.seq_len, mode="linear", align_corners=False)
            refined = self.phi_layers[scale_idx](upsampled)
            f_hat = f_hat + refined
            f_rest = f_rest - refined
            total_vq_loss = total_vq_loss + vq_loss
            indices_list.append(indices)

        return f_hat, indices_list, total_vq_loss

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        d = self.decoder_in(features)
        d = self.decoder_res(d)
        if self.decoder_head_type == "plain":
            return self.decoder_out(d).transpose(1, 2).contiguous()

        shared = self.decoder_shared_out(d)
        output = shared.new_zeros(shared.shape[0], self.action_dim, shared.shape[-1])
        for name, dims in self.dim_groups.items():
            output[:, dims, :] = self.decoder_group_heads[name](shared)
        return output.transpose(1, 2).contiguous()

    def flatten_tokens(self, indices_list: list[torch.Tensor]) -> torch.Tensor:
        if len(indices_list) != len(self.scales):
            raise ValueError(f"Expected {len(self.scales)} token groups, got {len(indices_list)}.")
        return torch.cat([indices.reshape(indices.shape[0], -1) for indices in indices_list], dim=1)

    def unflatten_tokens(self, flat_tokens: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(flat_tokens, list):
            return flat_tokens
        if flat_tokens.ndim == 1:
            flat_tokens = flat_tokens.unsqueeze(0)
        if flat_tokens.ndim != 2 or flat_tokens.shape[1] != self.token_dim:
            raise ValueError(f"Expected flat tokens with shape [B, {self.token_dim}], got {tuple(flat_tokens.shape)}.")

        groups = []
        offset = 0
        for scale in self.scales:
            width = scale * (self.product_codebook_groups if self.quantization_mode == "product_vq" else 1)
            chunk = flat_tokens[:, offset : offset + width].long()
            if self.quantization_mode == "product_vq":
                chunk = chunk.reshape(chunk.shape[0], scale, self.product_codebook_groups)
            groups.append(chunk)
            offset += width
        return groups

    @torch.no_grad()
    def encode(self, actions: torch.Tensor) -> torch.LongTensor:
        latent_full = self.encode_features(actions)
        _, indices_list, _ = self.quantize_latent(latent_full)
        return self.flatten_tokens(indices_list)

    @torch.no_grad()
    def decode(self, token_ids: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        token_groups = self.unflatten_tokens(token_ids)
        batch_size = token_groups[0].shape[0]
        device = token_groups[0].device
        dtype = self.shared_codebook.weight.dtype
        f_hat = torch.zeros(batch_size, self.embed_dim, self.seq_len, device=device, dtype=dtype)

        for scale_idx, (scale, indices) in enumerate(zip(self.scales, token_groups, strict=True)):
            if indices.shape[1] != scale:
                raise ValueError(f"Token group for scale {scale} has length {indices.shape[1]}.")
            if self.quantization_mode == "product_vq":
                group_features = []
                for group_idx, codebook in enumerate(self.product_codebooks):
                    group_features.append(codebook(indices[:, :, group_idx].long()))
                z_q_scale = torch.cat(group_features, dim=-1).transpose(1, 2).contiguous()
            else:
                z_q_scale = self.shared_codebook(indices.long()).transpose(1, 2).contiguous()
            upsampled = F.interpolate(z_q_scale, size=self.seq_len, mode="linear", align_corners=False)
            f_hat = f_hat + self.phi_layers[scale_idx](upsampled)

        return self.decode_features(f_hat)

    @torch.no_grad()
    def reinitialize_underused_codes(
        self,
        latent_samples: torch.Tensor,
        usage_counts: torch.Tensor,
        *,
        min_count: int = 0,
        max_codes: int = 64,
        noise_scale: float = 1e-3,
    ) -> int:
        if max_codes <= 0 or latent_samples.numel() == 0:
            return 0
        if latent_samples.ndim != 2 or latent_samples.shape[1] != self.embed_dim:
            raise ValueError(f"Expected latent_samples with shape [N, {self.embed_dim}], got {tuple(latent_samples.shape)}.")

        device = self.shared_codebook.weight.device
        usage_counts = torch.as_tensor(usage_counts, device=device)
        dead_indices = torch.nonzero(usage_counts <= min_count, as_tuple=False).flatten()
        if dead_indices.numel() == 0:
            return 0
        if dead_indices.numel() > max_codes:
            perm = torch.randperm(dead_indices.numel(), device=device)[:max_codes]
            dead_indices = dead_indices.index_select(0, perm)

        samples = latent_samples.to(device=device, dtype=self.shared_codebook.weight.dtype)
        if samples.shape[0] < dead_indices.numel():
            repeat_factor = math.ceil(dead_indices.numel() / samples.shape[0])
            samples = samples.repeat(repeat_factor, 1)
        sample_perm = torch.randperm(samples.shape[0], device=device)[: dead_indices.numel()]
        replacements = samples.index_select(0, sample_perm)
        if noise_scale > 0:
            replacements = replacements + noise_scale * torch.randn_like(replacements)
        if self.normalize_codebook_for_lookup:
            replacements = F.normalize(replacements, dim=-1, eps=1e-6)
        self.shared_codebook.weight.index_copy_(0, dead_indices, replacements)
        return int(dead_indices.numel())

    def forward(self, actions: torch.Tensor, *, return_latent: bool = False) -> dict[str, Any]:
        latent_full = self.encode_features(actions)
        if self.quantization_mode == "none":
            reconstructed_features = latent_full
            token_ids: list[torch.Tensor] = []
            vq_loss = torch.zeros((), device=latent_full.device, dtype=latent_full.dtype)
        else:
            reconstructed_features, token_ids, vq_loss = self.quantize_latent(latent_full)
        recon = self.decode_features(reconstructed_features)
        flat_token_ids = self.flatten_tokens(token_ids) if token_ids else torch.empty(
            actions.shape[0],
            0,
            device=actions.device,
            dtype=torch.long,
        )
        output: dict[str, Any] = {
            "recon": recon,
            "token_ids": token_ids,
            "flat_token_ids": flat_token_ids,
            "vq_loss": vq_loss,
            "aux": {
                "token_dim": self.token_dim,
                "scales": list(self.scales),
            },
        }
        if return_latent:
            output["latent"] = latent_full
        return output
