# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
Qwen-Dual Framework
A lightweight implementation that Qwen2.5-vl + dinov2 + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenDual
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenDualDefaultConfig:
    """QwenDual framework default parameters.

    Qwen-VL + DINOv2 + Flow-matching (DiT) action head.
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "QwenDual"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
    })

    # === DINO encoder (multi-view spatial tokens) ===
    dino: dict = field(default_factory=lambda: {
        # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
        "dino_backbone": "dinov2_vits14",
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
        # Layer index to connect VLM hidden states to action head (-1 = last layer)
        "connect_layer_index": -1,
        # Inference denoising steps
        "num_inference_timesteps": 4,
        "diffusion_model_cfg": {
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
    })

    # === Observation image size (resize before encoding) ===
    obs_image_size: Optional[list] = field(default_factory=lambda: [224, 224])


@FRAMEWORK_REGISTRY.register("QwenDual")
class Qwen_Dual(baseframework):
    """
    Multimodal vision-language-action model (Dual variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - DINOv2 encoder for dense multi-view spatial tokens
      - Flow-matching (DiT) diffusion head for continuous action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(QwenDualDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align cross_attention_dim to VLM hidden_size at runtime
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.dino_encoder = get_dino_model(
            backone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels, out_features=self.qwen_vl_interface.model.config.hidden_size
        )

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly regress future actions (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
        last_hidden, state = self.get_action_condition(batch_images, instructions, wrist_views, state)

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # get action labels
            actions = [example["action"] for example in examples]  # List of [T_full, action_dim]
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T, action_dim]
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]  # (B, chunk_len, action_dim)

            # repeate for efficient training
            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = None
            if state is not None:
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to directly regress future actions (no diffusion sampling).

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images, wrist_views, instructions, state = self.align_model_input(examples)
        last_hidden, state = self.get_action_condition(batch_images, instructions, wrist_views, state)
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)
        normalized_actions = pred_actions.detach().cpu().numpy()

        return {"normalized_actions": normalized_actions}

    def align_model_input(self, examples: List[dict]):

        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        wrist_views = (
            [to_pil_preserve(example["wrist_views"]) for example in examples] if "wrist_views" in examples[0] else None
        )  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", [224, 224])
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        if train_obs_image_size and wrist_views is not None:
            wrist_views = resize_images(wrist_views, target_size=train_obs_image_size)

        return batch_images, wrist_views, instructions, state

    def get_action_condition(self, batch_images, instructions, wrist_views=None, state=None):
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            connect_layer_index = self.config.framework.action_model.get("connect_layer_index", -1)
            last_hidden = qwenvl_outputs.hidden_states[connect_layer_index]  # [B, L, H]

            # Step 2: DINO Forward
            if wrist_views == None:
                wrist_views = batch_images
            image_tensors = self.dino_encoder.prepare_dino_input(wrist_views)  #
            B = len(batch_images)
            dino_features = self.dino_encoder(image_tensors)  # DINO output is [B*num_view, token, dim]
            dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])  # [B, num_view * token, dim]
            dino_encoded_features = self.dino_pro(dino_encoded_features)  # [B, num_view * token, hidden_size]

            # Step 3: Feature Concatenation
            last_hidden = torch.cat([last_hidden, dino_encoded_features], dim=1)
        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        return last_hidden, state


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    try:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    except (ImportError, RuntimeError):
        pass

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    # # cfg.framework.action_model.connect_layer_index = 16
    # cfg.framework.action_model.state_dim = 44
    # cfg.datasets.vla_data.include_state = True

    cfg.framework.action_model.action_hidden_dim = 2048
    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"

    model: Qwen_Dual = Qwen_Dual(cfg)
    print(model)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),  # action_chunk, action_dim
        "image": [image],  # three views
        # "wrist_views": [image, image],
        "lang": (
            "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room."
        ),
        # "state" : np.random.uniform(-1, 1, size=(1, 44)).astype(np.float16), # chunk, state_dim
    }

    sample2 = sample.copy()
    sample2["lang"] = "Move the red cup from the table to the kitchen counter next to the sink."
    batch = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])  # , state=[batch[0]["state"]]
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    # vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.task_id = 40
    # vla_dataset_cfg.video_backend = "torchvision_av"
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    # from torch.utils.data import DataLoader
    # train_dataloader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    # count = 0
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     count += 1
    #     if count > 1:
    #         break
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)
    # action = model.predict_action(examples=[sample])
    print("Finished")
