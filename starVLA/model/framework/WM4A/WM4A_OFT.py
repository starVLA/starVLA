# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WM4A-OFT Framework — World Model + MLP Regression for Action Prediction.

A lightweight WM4A variant inspired by OpenVLA-OFT: instead of a heavy
flow-matching diffusion action head, it uses a simple MLP with L1 regression
over pooled DiT hidden states.

Supports any world-model backend registered in ``get_world_model()``
(Cosmos-Predict2, Wan2.2-TI2V, etc.). The world model hidden_dim is
auto-detected at runtime.

Architecture:
  Text encoder + Image encoder → DiT Transformer
    → hidden_states [B, N_tokens, hidden_dim]
    → Adaptive pool → [B, chunk_len, hidden_dim]
    → MLP (L1 regression) → action predictions [B, chunk_len, action_dim]

Key differences from WM4A-GR00T variants:
  - Action head: MLP L1 regression (not flow-matching diffusion)
  - No repeated_diffusion_steps (single forward pass)
  - Faster training & inference, suitable for real-time deployment
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WM4AOFTDefaultConfig:
    """WM4A-OFT default parameters."""

    name: str = "WM4A_OFT"

    # === World Model backbone (any supported WM) ===
    world_model: dict = field(default_factory=lambda: {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B",
        "extract_layers": [-1],
    })

    # Legacy compat: vlm/__init__.py and world_model/__init__.py fall back to
    # config.framework.qwenvl.base_vlm when world_model.base_wm is absent.
    # Some action heads (VLA_AdapterHeader, LayerwiseFM) also read
    # qwenvl.vl_hidden_dim / num_vl_layers.  WM4A_OFT uses neither of those
    # action heads, so we only mirror base_vlm here for the factory fallback.
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B",
    })

    # === Action head (MLP L1 regression) ===
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP",
        "action_dim": 7,
        # Will be auto-set at runtime from world model hidden_size
        "action_hidden_dim": 3072,
        "future_action_window_size": 8,
        "past_action_window_size": 0,
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("WM4A_OFT")
class WM4A_OFT(baseframework):
    """
    World-Model-for-Action framework with MLP regression (OFT variant).

    Components:
      - Any DiT world model backbone for spatiotemporal feature extraction
      - Adaptive pooling to compress spatial tokens into action queries
      - MLP regression head (L1 loss) for continuous action prediction

    This is a lightweight alternative to the GR00T variants that use
    flow-matching diffusion heads. Faster training and inference,
    suitable for real-time deployment.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WM4AOFTDefaultConfig, config)

        # Load world model backbone
        self.backbone = get_world_model(config=self.config)

        # Auto-align action_hidden_dim to world model hidden size
        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.action_hidden_dim = wm_hidden

        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Learnable projection: pool N spatial tokens → chunk_len action queries
        self.action_query_proj = nn.Linear(wm_hidden, self.chunk_len * wm_hidden)

        self.l1_loss = nn.L1Loss()

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool spatiotemporal tokens into fixed-size action queries.

        Args:
            hidden_states: [B, N_tokens, hidden_dim] from DiT

        Returns:
            action_queries: [B, chunk_len, hidden_dim]
        """
        B, N, H = hidden_states.shape
        # Global average pool over spatial tokens → [B, H]
        pooled = hidden_states.mean(dim=1)  # [B, H]
        # Project to chunk_len separate queries
        queries = self.action_query_proj(pooled)  # [B, chunk_len * H]
        action_queries = queries.view(B, self.chunk_len, H)  # [B, chunk_len, H]
        return action_queries

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        # Step 1: World model input encoding
        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        # Step 2: DiT forward to extract spatiotemporal features
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = wm_outputs.hidden_states[-1]  # [B, N_tokens, hidden_dim]

        # Step 3: Pool to action queries and predict
        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(last_hidden)  # [B, chunk_len, H]
            pred_actions = self.action_model.predict_action(action_queries)  # [B, chunk_len, action_dim]

            actions = torch.tensor(
                np.array(actions), device=pred_actions.device, dtype=pred_actions.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(last_hidden)
            pred_actions = self.action_model.predict_action(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
