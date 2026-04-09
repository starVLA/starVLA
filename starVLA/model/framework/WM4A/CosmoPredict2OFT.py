# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
CosmoPredict2-OFT Framework — World Model + MLP Regression for Action Prediction.

Uses Cosmos-Predict2 DiT as the perception backbone with a lightweight MLP
action head (L1 regression), inspired by OpenVLA-OFT.

Architecture:
  T5 (text) + VAE (image) → DiT Transformer → hidden_states [B, N, 2048]
    → Global avg pool → [B, 2048]
    → Linear projection → [B, chunk_len, 2048]
    → MLP (L1 regression) → action predictions [B, chunk_len, action_dim]

Key differences from CosmoPredict2GR00T:
  - Action head: MLP L1 regression (not flow-matching diffusion)
  - No repeated_diffusion_steps (single forward pass)
  - Faster training & inference
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
class CosmoPredict2OFTDefaultConfig:
    """CosmoPredict2-OFT default parameters."""

    name: str = "CosmoPredict2OFT"

    # === World Model backbone (Cosmos-Predict2) ===
    world_model: dict = field(default_factory=lambda: {
        "base_wm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
        "extract_layers": [-1],
    })

    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
    })

    # === Action head (MLP L1 regression) ===
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "MLP",
        "action_dim": 7,
        "action_hidden_dim": 2048,
        "future_action_window_size": 8,
        "past_action_window_size": 0,
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("CosmoPredict2OFT")
class CosmoPredict2_OFT(baseframework):
    """
    World-Model-for-Action framework using Cosmos-Predict2 + MLP regression.

    Components:
      - Cosmos-Predict2 DiT (T5 + VAE + Transformer) for spatiotemporal features
      - Adaptive pooling + MLP regression head (L1 loss)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmoPredict2OFTDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.action_hidden_dim = wm_hidden

        self.action_model = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        self.action_query_proj = nn.Linear(wm_hidden, self.chunk_len * wm_hidden) # Project into a two-layer MLP

        self.l1_loss = nn.L1Loss()

    def _pool_to_action_queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, H = hidden_states.shape
        pooled = hidden_states.mean(dim=1)
        queries = self.action_query_proj(pooled)
        action_queries = queries.view(B, self.chunk_len, H)
        return action_queries

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = wm_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            action_queries = self._pool_to_action_queries(last_hidden) # B, chunk_len, hidden_dim
            pred_actions = self.action_model.predict_action(action_queries)

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


if __name__ == "__main__":
    import argparse
    import os

    from PIL import Image
    from omegaconf import OmegaConf

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "CosmoPredict2OFT"
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World",
        "extract_layers": [-1],
    }

    model: CosmoPredict2_OFT = CosmoPredict2_OFT(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "Pick up the red block and place it on the table.",
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Pick up the red block and place it on the table.",
    }

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Predicted Action: {normalized_actions}")
    print("Finished")
