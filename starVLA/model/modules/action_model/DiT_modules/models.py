# Modified from facebookresearch's DiT repos
# DiT: https://github.com/facebookresearch/DiT/blob/main/models.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


#################################################################################
#               Embedding Layers for Timesteps and conditions                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds conditions into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_size, hidden_size, dropout_prob=0.1, conditions_shape=(1, 1, 4096)):
        super().__init__()
        self.linear = nn.Linear(in_size, hidden_size)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.uncondition = nn.Parameter(torch.empty(conditions_shape[1:]))

    def token_drop(self, conditions, force_drop_ids=None):
        """
        Drops conditions to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(conditions.shape[0], device=conditions.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        conditions = torch.where(
            drop_ids.unsqueeze(1).unsqueeze(1).expand(conditions.shape[0], *self.uncondition.shape),
            self.uncondition,
            conditions,
        )
        return conditions

    def forward(self, conditions, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            conditions = self.token_drop(conditions, force_drop_ids)
        embeddings = self.linear(conditions)
        return embeddings


#################################################################################
#                      Embedding Layers for Actions and                         #
#################################################################################
class ActionEmbedder(nn.Module):
    def __init__(self, action_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(action_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        return x


# Action_History is not used now
class HistoryEmbedder(nn.Module):
    def __init__(self, action_size, hidden_size):
        super().__init__()
        self.linear = nn.Linear(action_size, hidden_size)

    def forward(self, x):
        x = self.linear(x)
        return x


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with self-attention conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels=7,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        token_size=4096,
        future_action_window_size=1,
        past_action_window_size=0,
        learn_sigma=False,
        n_conditon_token=64,
    ):
        super().__init__()

        assert past_action_window_size == 0, "Error: action_history is not used now"
        self.num_cond_tokens = n_conditon_token
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_heads = num_heads
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size

        # Action history is not used now.
        self.history_embedder = HistoryEmbedder(action_size=in_channels, hidden_size=token_size)

        self.x_embedder = ActionEmbedder(action_size=in_channels, hidden_size=token_size)
        self.t_embedder = TimestepEmbedder(token_size)
        conditions_shape = (1, n_conditon_token, token_size)

        self.z_embedder = LabelEmbedder(
            in_size=token_size,
            hidden_size=token_size,
            dropout_prob=class_dropout_prob,
            conditions_shape=conditions_shape,
        )
        scale = token_size**-0.5

        # Learnable positional embeddings
        # 1+64, one for the conditional token, and one for the current action prediction
        self.positional_embedding = nn.Parameter(
            scale
            * torch.randn(self.num_cond_tokens + future_action_window_size + past_action_window_size + 1, token_size)
        )

        self.blocks = nn.ModuleList([DiTBlock(token_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(token_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize token_embed like nn.Linear
        nn.init.normal_(self.x_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.x_embedder.linear.bias, 0)

        nn.init.normal_(self.history_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.history_embedder.linear.bias, 0)

        # Initialize label embedding table:
        if self.class_dropout_prob > 0:
            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
        nn.init.normal_(self.z_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.z_embedder.linear.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, z):
        """
        Forward pass of DiT.
        history: (B, H, D) tensor of action history # not used now
        x: (B, T, D) tensor of predicting action inputs
        t: (B,) tensor of diffusion timesteps
        z: [B, num_cond_tokens, D] -- condition token
        """
        x = self.x_embedder(x)  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)
        z = self.z_embedder(z, self.training)  # [N, num_cond_tokens, D]
        c = t.unsqueeze(1) + z  # (N, 64, D)
        x = torch.cat((c, x), dim=1)  # (N, T+64, D)
        x = x + self.positional_embedding  # (N, T+64, D)
        for block in self.blocks:
            x = block(x)  # (N, T+64, D)
        x = self.final_layer(x)  # (N, T+64, out_channels)
        return x[:, self.num_cond_tokens :, :]  # (N, T, C)

    def forward_with_cfg(self, x, t, z, cfg_scale):
        """
        Forward pass of Diffusion, but also batches the unconditional forward pass for classifier-free guidance.
        """

        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0).to(next(self.x_embedder.parameters()).dtype)
        model_out = self.forward(combined, t, z)
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        # return torch.cat([eps, rest], dim=1)
        return torch.cat([eps, rest], dim=2)


# Cross-Attention DiT Implementation


class CrossAttention(nn.Module):
    """
    Cross-attention module that supports both self-attention and cross-attention.
    """

    def __init__(self, hidden_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        """
        Args:
            x: query tensor [B, N, C]
            context: key/value tensor [B, M, C]. If None, performs self-attention
        """
        B, N, C = x.shape

        # Query from x
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Key and Value from context (or x if self-attention)
        if context is None:
            context = x
        M = context.shape[1]
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlockCrossAttn(nn.Module):
    """
    A DiT block with only cross-attention + MLP.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        # Cross-attention components
        self.norm_attn = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(hidden_size, num_heads=num_heads, **block_kwargs)

        # MLP components
        self.norm_mlp = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, encoder_features=None):
        """
        Args:
            x: input tensor [B, N, C] (action-related tokens)
            encoder_features: encoder features [B, M, C] (e.g., vision-language features)
        """
        # Cross-attention (if encoder features provided) or self-attention (if None)
        if encoder_features is not None:
            # Cross-attention: Query from x, Key/Value from encoder_features
            x = x + self.cross_attn(self.norm_attn(x), context=encoder_features)
        else:
            # Self-attention: Query, Key, Value all from x (for backward compatibility)
            x = x + self.cross_attn(self.norm_attn(x), context=None)

        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DiTBlockSelfAttn(nn.Module):
    """
    A DiT block with only self-attention + MLP.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        # Self-attention components (same as original DiTBlock)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        # MLP components (same as original DiTBlock)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, encoder_features=None):
        """
        Args:
            x: input tensor [B, N, C] (action-related tokens)
            encoder_features: encoder features [B, M, C] (not used in self-attention, for interface compatibility)
        """
        # Self-attention (identical to original DiTBlock)
        x = x + self.attn(self.norm1(x))

        # MLP (identical to original DiTBlock)
        x = x + self.mlp(self.norm2(x))
        return x


class DiTCrossAttn(nn.Module):
    """
    Diffusion model with a Transformer backbone supporting cross-attention.
    """

    def __init__(
        self,
        in_channels=7,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        token_size=4096,
        future_action_window_size=1,
        past_action_window_size=0,
        learn_sigma=False,
        n_conditon_token=64,
    ):
        super().__init__()

        assert past_action_window_size == 0, "Error: action_history is not used now"
        self.num_cond_tokens = n_conditon_token
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.class_dropout_prob = class_dropout_prob
        self.num_heads = num_heads
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size

        # Action history is not used now.
        self.history_embedder = HistoryEmbedder(action_size=in_channels, hidden_size=token_size)

        self.x_embedder = ActionEmbedder(action_size=in_channels, hidden_size=token_size)
        self.t_embedder = TimestepEmbedder(token_size)
        conditions_shape = (1, n_conditon_token, token_size)

        self.z_embedder = LabelEmbedder(
            in_size=token_size,
            hidden_size=token_size,
            dropout_prob=class_dropout_prob,
            conditions_shape=conditions_shape,
        )
        scale = token_size**-0.5

        # Learnable positional embeddings
        actual_action_length = future_action_window_size + past_action_window_size + 1
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_cond_tokens + actual_action_length, token_size)
        )

        # Alternating cross-attention and self-attention blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(depth):
            if layer_idx % 2 == 0:  # Even layers (0, 2, 4, ...): Cross-Attention
                block = DiTBlockCrossAttn(token_size, num_heads, mlp_ratio=mlp_ratio)
            else:  # Odd layers (1, 3, 5, ...): Self-Attention
                block = DiTBlockSelfAttn(token_size, num_heads, mlp_ratio=mlp_ratio)
            self.blocks.append(block)
        self.final_layer = FinalLayer(token_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize embedders
        nn.init.normal_(self.x_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.x_embedder.linear.bias, 0)

        nn.init.normal_(self.history_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.history_embedder.linear.bias, 0)

        # Initialize label embedding table:
        if self.class_dropout_prob > 0:
            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
        nn.init.normal_(self.z_embedder.linear.weight, std=0.02)
        nn.init.constant_(self.z_embedder.linear.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, z, encoder_features=None):
        """
        Forward pass of DiT with cross-attention.
        Args:
            x: (B, T, D) tensor of predicting action inputs
            t: (B,) tensor of diffusion timesteps
            z: [B, num_cond_tokens, D] -- condition token
            encoder_features: [B, M, D] -- encoder features for cross-attention (e.g., vision-language features)
        """
        x = self.x_embedder(x)  # (N, T, D)
        t = self.t_embedder(t)  # (N, D)
        z = self.z_embedder(z, self.training)  # [N, num_cond_tokens, D]
        c = t.unsqueeze(1) + z  # (N, 64, D)
        x = torch.cat((c, x), dim=1)  # (N, T+64, D)
        x = x + self.positional_embedding  # (N, T+64, D)

        # Pass through cross-attention blocks
        for block in self.blocks:
            x = block(x, encoder_features=encoder_features)  # (N, T+64, D)

        x = self.final_layer(x)  # (N, T+64, out_channels)
        return x[:, self.num_cond_tokens :, :]  # (N, T, C)

    def forward_with_cfg(self, x, t, z, cfg_scale, encoder_features=None):
        """
        Forward pass with classifier-free guidance for cross-attention DiT.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0).to(next(self.x_embedder.parameters()).dtype)

        # Handle encoder features for CFG: conditional + unconditional
        if encoder_features is not None:
            # First half: conditional (with encoder features)
            # Second half: unconditional (without encoder features, set to None)
            encoder_features_combined = torch.cat([encoder_features, encoder_features], dim=0)
            # Note: For true CFG, you might want to pass None for the second half:
            # But this would require modifying the forward pass to handle mixed batches
        else:
            encoder_features_combined = None

        model_out = self.forward(combined, t, z, encoder_features=encoder_features_combined)
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=2)
