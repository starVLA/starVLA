# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused self/cross-attention DiT used by :mod:`QwenPI_v4`.

This module deliberately has its own implementation instead of extending the
legacy DiT.  A QwenPI_v4 transformer block uses action/state tokens as the
query and concatenates them with the VLM tokens for the key/value sequence:

    query = action/state tokens
    key/value = [action/state tokens, VLM tokens]

This fuses action-token self-attention and VLM cross-attention into one
attention operation per block.  The action-token portion of the key/value
mask is all valid, while the VLM portion uses ``encoder_attention_mask``.
There is no interleaving/mode flag in this implementation and the VLM model
itself is not changed.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


class QwenPIv4TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)


class QwenPIv4AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.norm = nn.LayerNorm(
            embedding_dim,
            eps=norm_eps,
            elementwise_affine=norm_elementwise_affine,
        )

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1 + scale[:, None]) + shift[:, None]


class QwenPIv4TransformerBlock(nn.Module):
    """One fused action/VLM attention block followed by an FFN."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "ada_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        if cross_attention_dim is None:
            cross_attention_dim = dim
        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError(
                "num_positional_embeddings is required when positional_embeddings is set."
            )

        self.norm_type = norm_type
        self.pos_embed = (
            SinusoidalPositionalEmbedding(
                dim,
                max_seq_length=num_positional_embeddings,
            )
            if positional_embeddings == "sinusoidal"
            else None
        )

        if norm_type == "ada_norm":
            self.norm1 = QwenPIv4AdaLayerNorm(
                dim,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
            )
        else:
            self.norm1 = nn.LayerNorm(
                dim,
                eps=norm_eps,
                elementwise_affine=norm_elementwise_affine,
            )

        # A single cross-attention module receives action/state tokens and VLM
        # tokens as its KV sequence.  If a GR00T-style configuration keeps a
        # different cross-attention input dimension, project action/state KV
        # tokens to that dimension before concatenation.
        self.action_kv_proj = (
            nn.Identity() if cross_attention_dim == dim else nn.Linear(dim, cross_attention_dim)
        )
        self.fused_cross_attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(
            dim,
            eps=norm_eps,
            elementwise_affine=norm_elementwise_affine,
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenPI_v4 requires VLM encoder states in every transformer block.")

        # 1. Fused attention: action/state queries attend to both action/state
        # keys and valid VLM keys.  The action-token mask is deliberately all
        # ones; only VLM padding positions are masked out.
        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)
        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        action_kv = self.action_kv_proj(norm_hidden_states)
        fused_encoder_hidden_states = torch.cat(
            [action_kv, encoder_hidden_states],
            dim=1,
        )
        fused_attention_mask = self._build_fused_attention_mask(
            encoder_attention_mask,
            batch_size=hidden_states.shape[0],
            action_length=hidden_states.shape[1],
            device=hidden_states.device,
        )
        attn_output = self.fused_cross_attn(
            norm_hidden_states,
            encoder_hidden_states=fused_encoder_hidden_states,
            attention_mask=fused_attention_mask,
        )
        if self.final_dropout is not None:
            attn_output = self.final_dropout(attn_output)
        hidden_states = hidden_states + attn_output

        # 2. Feed-forward network.
        ff_output = self.ff(self.norm3(hidden_states))
        hidden_states = hidden_states + ff_output
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states

    @staticmethod
    def _build_fused_attention_mask(
        encoder_attention_mask: Optional[torch.Tensor],
        batch_size: int,
        action_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Prefix the VLM padding mask with an all-valid action-token mask."""
        if encoder_attention_mask is None:
            return None
        if encoder_attention_mask.ndim != 2:
            raise ValueError(
                "QwenPI_v4 fused attention expects a 2D VLM padding mask "
                "with shape (batch, vlm_tokens)."
            )
        if encoder_attention_mask.shape[0] != batch_size:
            raise ValueError(
                "QwenPI_v4 fused attention mask batch size does not match "
                f"hidden_states: {encoder_attention_mask.shape[0]} vs {batch_size}."
            )
        action_attention_mask = torch.ones(
            batch_size,
            action_length,
            dtype=encoder_attention_mask.dtype,
            device=device,
        )
        return torch.cat(
            [action_attention_mask, encoder_attention_mask.to(device=device)],
            dim=1,
        )


class QwenPIv4DiT(ModelMixin, ConfigMixin):
    """Independent fused self/cross-attention DiT for QwenPI_v4."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        input_embedding_dim: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        if input_embedding_dim is not None and int(input_embedding_dim) != self.inner_dim:
            raise ValueError(
                "QwenPI_v4 diffusion_model_cfg.input_embedding_dim must equal "
                "num_attention_heads * attention_head_dim."
            )
        if cross_attention_dim is None:
            cross_attention_dim = self.inner_dim

        self.gradient_checkpointing = False
        self.timestep_encoder = QwenPIv4TimestepEncoder(
            embedding_dim=self.inner_dim,
            compute_dtype=compute_dtype,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                QwenPIv4TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)
        print(
            "Total number of QwenPIv4DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        return_all_hidden_states: bool = False,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_pre_output: bool = False,
    ):
        """Run every block with fused action/VLM attention.

        ``encoder_hidden_states`` may be one tensor shared by all blocks or a
        list/tuple with one VLM hidden state tensor per block.  In both cases,
        ``encoder_attention_mask`` masks only the VLM segment of every block's
        concatenated key/value sequence.  Action/state keys are always valid.
        """
        if timestep is None:
            raise ValueError("QwenPI_v4 requires a timestep tensor.")

        temb = self.timestep_encoder(timestep)
        hidden_states = hidden_states.contiguous()
        is_layerwise_encoder = isinstance(encoder_hidden_states, (list, tuple))
        if is_layerwise_encoder:
            if len(encoder_hidden_states) != len(self.transformer_blocks):
                raise ValueError(
                    "QwenPI_v4 expects one layer-wise VLM encoder state per DiT block: "
                    f"got {len(encoder_hidden_states)} states for "
                    f"{len(self.transformer_blocks)} blocks."
                )
            encoder_hidden_states = [state.contiguous() for state in encoder_hidden_states]
        else:
            encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]
        for idx, block in enumerate(self.transformer_blocks):
            block_encoder_hidden_states = (
                encoder_hidden_states[idx] if is_layerwise_encoder else encoder_hidden_states
            )
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=block_encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                temb=temb,
            )
            all_hidden_states.append(hidden_states)

        if return_pre_output:
            if return_all_hidden_states:
                return hidden_states, all_hidden_states
            return hidden_states

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        output = self.proj_out_2(hidden_states)
        if return_all_hidden_states:
            return output, all_hidden_states
        return output
