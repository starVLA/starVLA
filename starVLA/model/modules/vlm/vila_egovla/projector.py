# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Vendored from EgoVLA / VILA `multimodal_projector/base_projector.py`
# (original: Apache-2.0, NVIDIA). Trimmed to the `mlp_downsample` variant
# actually used by the EgoVLA checkpoint (mm_projector_type=mlp_downsample).
"""EgoVLA vision->language projector (2x2 spatial downsample + MLP).

Maps SigLIP patch features (mm_hidden_size=1152) into the LLM embedding space
(hidden_size=1536). The ``layers.{1,2,4}`` parameter naming is preserved so the
EgoVLA checkpoint's ``mm_projector/model.safetensors`` loads with strict=True.

For SigLIP-384 (27x27 patches) the DownSampleBlock pads to 28x28 and folds 2x2
neighbourhoods -> 14x14 = 196 tokens of width mm_hidden_size*4, so each image
contributes 196 tokens to the LLM sequence.
"""

import torch
import torch.nn as nn


class DownSampleBlock(nn.Module):
    """Fold each 2x2 patch neighbourhood into the channel dim (VILA-exact)."""

    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        return x


class EgoVLAMMProjector(nn.Module):
    """`mlp_downsample` projector. state_dict keys: layers.{1,2,4}.{weight,bias}."""

    def __init__(self, mm_hidden_size: int = 1152, hidden_size: int = 1536):
        super().__init__()
        self.layers = nn.Sequential(
            DownSampleBlock(),                                  # layers.0 (no params)
            nn.LayerNorm(mm_hidden_size * 4),                   # layers.1
            nn.Linear(mm_hidden_size * 4, hidden_size),         # layers.2
            nn.GELU(),                                          # layers.3 (no params)
            nn.Linear(hidden_size, hidden_size),                # layers.4
        )

    def forward(self, x, *args, **kwargs):
        return self.layers(x)
