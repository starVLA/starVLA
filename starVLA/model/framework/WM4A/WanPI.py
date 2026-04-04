# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
WanPI Framework — Wan2.2-TI2V World Model + Layer-wise Cross-DiT Flow-Matching.

Uses Wan2.2-TI2V-5B (DiT-based Text+Image-to-Video model) as the perception
backbone with a layer-wise cross-DiT flow-matching action head, inspired by π₀.

Architecture:
  UMT5 (text) + VAE (image→latent) → WanTransformer3D (30 blocks)
    → Multi-layer hidden_states [30 × (B, N, 3072)]
    → LayerwiseFlowmatchingActionHead (cross-DiT)
    → action predictions [B, chunk_len, action_dim]

Key differences from WanGR00T:
  - Action head: Layer-wise cross-DiT (not single-layer flow-matching)
  - Uses ALL transformer layers (not just last hidden state)
  - More expressive multi-scale feature fusion
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
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class WanPIDefaultConfig:
    """WanPI default parameters."""

    name: str = "WanPI"

    # === World Model backbone (Wan2.2-TI2V-5B-Diffusers) ===
    world_model: dict = field(default_factory=lambda: {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    })

    # LayerwiseFM reads qwenvl.vl_hidden_dim and qwenvl.num_vl_layers
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vl_hidden_dim": 3072,
        "num_vl_layers": 30,
    })

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "LayerwiseFM",
        "action_dim": 7,
        "state_dim": 7,
        "future_action_window_size": 15,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 2,
        "num_inference_timesteps": 4,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "num_target_vision_tokens": 32,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "diffusion_model_cfg": {},
    })

    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("WanPI")
class Wan_PI(baseframework):
    """
    World-Model-for-Action framework using Wan2.2-TI2V + Layer-wise cross-DiT.

    Components:
      - Wan2.2-TI2V DiT (UMT5 + VAE + WanTransformer3D) for features
      - Layer-wise cross-DiT flow-matching head fed by all transformer layers
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(WanPIDefaultConfig, config)

        self.backbone = get_world_model(config=self.config)

        # Auto-detect hidden size and num layers from world model
        wm_hidden = self.backbone.model.config.hidden_size
        # Cosmos uses transformer_blocks, Wan uses blocks
        if hasattr(self.backbone.transformer, 'transformer_blocks'):
            num_blocks = len(self.backbone.transformer.transformer_blocks)
        else:
            num_blocks = len(self.backbone.transformer.blocks)

        self.config.framework.qwenvl.vl_hidden_dim = wm_hidden
        self.config.framework.qwenvl.num_vl_layers = num_blocks

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Register hooks for ALL transformer blocks
        self._all_hidden_states = []
        self._all_hooks = []
        self._register_all_hooks()

    def _register_all_hooks(self):
        """Register forward hooks on ALL transformer blocks for layerwise features."""
        for hook in self._all_hooks:
            hook.remove()
        self._all_hooks.clear()

        if hasattr(self.backbone.transformer, 'transformer_blocks'):
            blocks = self.backbone.transformer.transformer_blocks
        else:
            blocks = self.backbone.transformer.blocks
        for block in blocks:
            hook = block.register_forward_hook(self._capture_all_hook)
            self._all_hooks.append(hook)

    def _capture_all_hook(self, module, input, output):
        if isinstance(output, tuple):
            self._all_hidden_states.append(output[0])
        else:
            self._all_hidden_states.append(output)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        wm_inputs = self.backbone.build_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._all_hidden_states.clear()
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(self._all_hidden_states)
            base_hidden = vl_embs_list[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 2)
                if self.config and hasattr(self.config, "framework")
                else 2
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(
                vl_embs_list_repeated, actions_target_repeated, state_repeated
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
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
            self._all_hidden_states.clear()
            wm_outputs = self.backbone(
                **wm_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            vl_embs_list = list(self._all_hidden_states)

        state = (
            torch.from_numpy(np.array(state)).to(vl_embs_list[-1].device, dtype=vl_embs_list[-1].dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(vl_embs_list, state)

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

    cfg.framework.name = "WanPI"
    cfg.framework.qwenvl = {
        "base_vlm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vl_hidden_dim": 3072,
        "num_vl_layers": 30,
    }
    cfg.framework.world_model = {
        "base_wm": "./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "extract_layers": [-1],
    }

    model: Wan_PI = Wan_PI(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "Pick up the red block and place it on the table.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "Pick up the red block and place it on the table.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
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
