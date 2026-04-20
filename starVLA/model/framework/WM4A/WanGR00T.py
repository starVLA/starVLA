# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanGR00T Framework — Wan2.2-TI2V World Model for Action Prediction.

Uses Wan2.2-TI2V-5B (DiT-based Text+Image-to-Video model) as the perception
backbone. The DiT's intermediate representations encode spatiotemporal
dynamics learned from large-scale video generation pretraining,
which are passed directly to the action head as cross-attention condition
for continuous action prediction.

Architecture:
  UMT5 (text) + VAE (image→latent) → WanTransformer3D
    → hidden_states [B, N, 3072]
    → FlowmatchingActionHead (cross-attention on 3072-dim tokens)
    → action predictions

Key differences from CosmoPredict2GR00T:
  - Text encoder: UMT5-XXL (dim=4096) vs T5 (dim=1024)
  - VAE latent channels: 48 vs 16
  - DiT hidden dim: 3072 (24×128) vs 2048 (16×128)
  - 30 transformer blocks vs 28
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

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WanGR00TDefaultConfig:
    """WanGR00T default parameters."""

    name: str = "WanGR00T"

    # === World Model backbone (Wan2.2-TI2V-5B-Diffusers) ===
    world_model: dict = field(default_factory=lambda: {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    })

    # Legacy compat: factory functions (vlm/__init__, world_model/__init__)
    # fall back to qwenvl.base_vlm when world_model.base_wm is absent.
    # vl_hidden_dim is read by some action heads (VLA_AdapterHeader, LayerwiseFM).
    # TODO next version should refactor to remove this redundant config section and update all shared utilities to read from world_model.base_wm instead of qwenvl.base_vlm.
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vl_hidden_dim": 3072,
    })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 7,
        "action_horizon": 8,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 8,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "diffusion_model_cfg": {
            # Decoupled from world model hidden_size; wm_projector bridges the gap
            "cross_attention_dim": 512,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("WanGR00T")
class Wan_GR00T(baseframework):
    """
    World-Model-for-Action framework using Wan2.2-TI2V-5B backbone.

    Components:
      - Wan2.2-TI2V DiT (UMT5 + VAE + WanTransformer3D) for features
      - Flow-matching (DiT) diffusion head for continuous action prediction

    The Wan world model provides spatiotemporal representations learned
    from large-scale video generation pretraining with expand_timesteps
    image conditioning (per-token timestep expansion).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanGR00TDefaultConfig, config)

        # Load world model backbone
        self.backbone = get_world_model(config=self.config)

        # Project world model features to action model's cross-attention dim
        wm_hidden = self.backbone.model.config.hidden_size
        cross_attn_dim = self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim
        self.wm_projector = torch.nn.Linear(wm_hidden, cross_attn_dim)

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Step 1: World model input encoding (UMT5 + VAE)
        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        # Step 2: DiT forward to extract spatiotemporal features
        with torch.autocast("cuda", dtype=torch.bfloat16):
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states[-1]: [B, N_tokens, hidden_dim=3072]
            last_hidden = wm_outputs.hidden_states[-1]
            last_hidden = self.wm_projector(last_hidden)

        # Step 3: Action head forward and loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

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
            last_hidden = self.wm_projector(last_hidden)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)

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
        default="./starVLA/config/training/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    # Point to the diffusers-format model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    }

    model: Wan_GR00T = Wan_GR00T(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
