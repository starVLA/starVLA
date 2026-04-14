# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
QwenIDM Framework — Inverse Dynamics Model with Visual Prediction

Architecture:
  - Learnable query tokens appended after image+language tokens inside the VLM backbone
  - Query tokens predict future visual state z_hat_{t+h} via MSE loss against ViT-encoded future image
  - Action expert (flow-matching DiT) acts as an inverse dynamics model:
    noise attends to concatenated [z_t; z_{t+h}] to denoise action trajectories

Training requires `future_image` in examples (set `future_obs_horizon` in data config).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("QwenIDM")
class Qwen_IDM(baseframework):
    """
    Inverse Dynamics Model framework with visual prediction.

    Components:
      - Qwen VL backbone (image + language encoding)
      - Learnable query tokens for future visual state prediction
      - Flow-matching DiT as inverse dynamics action expert

    The VLM backbone processes [image_tokens | language_tokens | query_tokens]
    with standard causal masking — query tokens at the end can attend to all
    preceding tokens, but preceding tokens cannot attend to query tokens.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # 1. VLM backbone
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # 2. Derive hidden size from VLM config
        llm_hidden_size = self.qwen_vl_interface.model.config.hidden_size

        # 3. Learnable query tokens for visual prediction
        idm_cfg = config.framework.idm
        num_query_tokens = idm_cfg.num_query_tokens
        self.query_tokens = nn.Embedding(num_query_tokens, llm_hidden_size)
        nn.init.normal_(self.query_tokens.weight, mean=0.0, std=0.02)
        self.num_query_tokens = num_query_tokens

        # 4. Action model (non-layerwise flow-matching DiT)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = llm_hidden_size
        self.action_model = FlowmatchingActionHead(full_config=self.config)

        # 5. Loss weight for visual prediction
        self.visual_pred_loss_weight = idm_cfg.get("visual_pred_loss_weight", 1.0)

        # 6. Action window sizes
        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # 7. Internal reference to inner Qwen VL model (Qwen2_5_VLModel / Qwen3VLModel)
        self._qwen_vl_model = self.qwen_vl_interface.model.model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_fused_embeds_and_vit_features(
        self, input_ids, attention_mask, pixel_values, image_grid_thw
    ):
        """
        Replicate the embedding fusion logic from Qwen2_5_VLModel.forward():
        text embedding + ViT encoding + masked_scatter to merge vision tokens.

        Returns:
            inputs_embeds: (B, T, D) — fused text + vision embeddings
            image_embeds: tuple of tensors — raw ViT features per image
        """
        inputs_embeds = self._qwen_vl_model.get_input_embeddings()(input_ids)

        image_embeds = None
        if pixel_values is not None:
            image_embeds = self._qwen_vl_model.get_image_features(pixel_values, image_grid_thw)
            image_embeds_cat = torch.cat(image_embeds, dim=0).to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            image_mask, _ = self._qwen_vl_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds_cat
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_cat)

        return inputs_embeds, image_embeds

    def _extend_with_query_tokens(self, inputs_embeds, attention_mask, input_ids, image_grid_thw):
        """
        Append learnable query tokens to the fused embedding sequence and
        extend attention_mask / position_ids accordingly.

        Returns:
            inputs_embeds_ext: (B, T+Q, D)
            attention_mask_ext: (B, T+Q)
            position_ids_ext: (3, B, T+Q)
        """
        B = inputs_embeds.shape[0]
        device = inputs_embeds.device

        # Append query token embeddings
        query_embeds = self.query_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        inputs_embeds_ext = torch.cat([inputs_embeds, query_embeds], dim=1)

        # Extend attention mask
        query_mask = torch.ones(
            B, self.num_query_tokens, device=device, dtype=attention_mask.dtype
        )
        attention_mask_ext = torch.cat([attention_mask, query_mask], dim=1)

        # Compute position_ids on original sequence, then extend for query tokens
        position_ids, _ = self._qwen_vl_model.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw=None, attention_mask=attention_mask
        )
        last_pos = position_ids.max(dim=-1, keepdim=True).values  # (3, B, 1)
        query_pos_offsets = torch.arange(
            1, self.num_query_tokens + 1, device=device, dtype=position_ids.dtype
        )
        query_pos = query_pos_offsets.view(1, 1, -1).expand(3, B, -1) + last_pos
        position_ids_ext = torch.cat([position_ids, query_pos], dim=-1)

        return inputs_embeds_ext, attention_mask_ext, position_ids_ext

    def _pool_image_embeds_by_batch(self, image_embeds_tuple, batch_images):
        """
        Average-pool ViT features to (B, D) by grouping per sample and
        averaging across tokens and views.

        Args:
            image_embeds_tuple: tuple of tensors from get_image_features(),
                one tensor per image across all batch items.
            batch_images: List[List[PIL.Image]] — tells us how many images per sample.

        Returns:
            Tensor: (B, D)
        """
        pooled = []
        idx = 0
        for imgs in batch_images:
            num_views = len(imgs)
            view_features = []
            for _ in range(num_views):
                view_features.append(image_embeds_tuple[idx].mean(dim=0))
                idx += 1
            pooled.append(torch.stack(view_features).mean(dim=0))
        return torch.stack(pooled)

    @torch.no_grad()
    def _encode_future_images(self, future_images_batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Encode future images through ViT only (no LLM layers).
        Returns average-pooled features (B, D).

        Gradient is detached — the visual prediction loss does NOT update
        the ViT's representation of the ground-truth future image.
        """
        future_qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=future_images_batch,
            instructions=[""] * len(future_images_batch),
        )
        pixel_values = future_qwen_inputs["pixel_values"]
        image_grid_thw = future_qwen_inputs["image_grid_thw"]

        image_embeds = self._qwen_vl_model.get_image_features(pixel_values, image_grid_thw)
        return self._pool_image_embeds_by_batch(image_embeds, future_images_batch)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        """
        Training forward pass.

        Args:
            examples: List[dict], each with:
                - image: List[PIL.Image]
                - future_image: List[PIL.Image]
                - lang: str
                - action: np.ndarray [T, action_dim]
                - state: np.ndarray [1, state_dim] (optional)

        Returns:
            dict with action_loss (total), raw_action_loss, visual_pred_loss
        """
        batch_images = [ex["image"] for ex in examples]
        future_images = [ex["future_image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        # ---- Step 1: Build VLM inputs for current observation ----
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        pixel_values = qwen_inputs["pixel_values"]
        image_grid_thw = qwen_inputs["image_grid_thw"]
        B = input_ids.shape[0]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # ---- Step 2: Embedding fusion (text + ViT merge) ----
            inputs_embeds, image_embeds = self._get_fused_embeds_and_vit_features(
                input_ids, attention_mask, pixel_values, image_grid_thw
            )

            # z_t: pooled current ViT features for action conditioning
            z_t_pooled = self._pool_image_embeds_by_batch(image_embeds, batch_images)

            # ---- Step 3: Append query tokens + extend masks ----
            inputs_embeds_ext, attention_mask_ext, position_ids_ext = (
                self._extend_with_query_tokens(
                    inputs_embeds, attention_mask, input_ids, image_grid_thw
                )
            )

            # ---- Step 4: Forward through language model ----
            outputs = self._qwen_vl_model.language_model(
                input_ids=None,
                position_ids=position_ids_ext,
                attention_mask=attention_mask_ext,
                inputs_embeds=inputs_embeds_ext,
                output_hidden_states=True,
                return_dict=True,
            )

            # ---- Step 5: Extract query token hidden states ----
            last_hidden = outputs.last_hidden_state  # (B, T+Q, D)
            z_hat_future = last_hidden[:, -self.num_query_tokens:, :]  # (B, Q, D)
            z_hat_future_pooled = z_hat_future.mean(dim=1)  # (B, D)

            # ---- Step 6: Encode future image through ViT only ----
            z_future_pooled = self._encode_future_images(future_images)  # (B, D)

        # ---- Step 7: Visual prediction loss ----
        with torch.autocast("cuda", dtype=torch.float32):
            visual_pred_loss = F.mse_loss(
                z_hat_future_pooled.float(),
                z_future_pooled.detach().float(),
            )

        # ---- Step 8: Action expert (inverse dynamics) ----
        with torch.autocast("cuda", dtype=torch.float32):
            z_t_token = z_t_pooled.unsqueeze(1)  # (B, 1, D)
            z_future_token = z_future_pooled.unsqueeze(1)  # (B, 1, D)
            action_conditioning = torch.cat([z_t_token, z_future_token], dim=1)  # (B, 2, D)

            actions_tensor = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=torch.float32
            )
            actions_target = actions_tensor[:, -(self.future_action_window_size + 1):, :]

            repeated_diffusion_steps = self.config.framework.action_model.get(
                "repeated_diffusion_steps", 2
            )
            actions_target_rep = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            conditioning_rep = action_conditioning.repeat(repeated_diffusion_steps, 1, 1)

            state_rep = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=torch.float32
                )
                state_rep = state_tensor.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(conditioning_rep, actions_target_rep, state_rep)

        # ---- Step 9: Total loss ----
        total_loss = action_loss + self.visual_pred_loss_weight * visual_pred_loss

        return {
            "action_loss": total_loss,
            "action_dit_loss": action_loss.item(),
            "visual_pred_loss": visual_pred_loss.item(),
        }

    # ------------------------------------------------------------------
    # Predict action (inference)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """
        Inference: predict actions using predicted future visual state.

        At inference time there is no future image — the VLM's query tokens
        produce z_hat_{t+h} which substitutes for z_{t+h}.

        Returns:
            dict with normalized_actions: np.ndarray [B, T, action_dim]
        """
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        B = len(examples)

        # ---- Build VLM inputs ----
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        pixel_values = qwen_inputs["pixel_values"]
        image_grid_thw = qwen_inputs["image_grid_thw"]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Embedding fusion
            inputs_embeds, image_embeds = self._get_fused_embeds_and_vit_features(
                input_ids, attention_mask, pixel_values, image_grid_thw
            )
            z_t_pooled = self._pool_image_embeds_by_batch(image_embeds, batch_images)

            # Append query tokens + forward
            inputs_embeds_ext, attention_mask_ext, position_ids_ext = (
                self._extend_with_query_tokens(
                    inputs_embeds, attention_mask, input_ids, image_grid_thw
                )
            )

            outputs = self._qwen_vl_model.language_model(
                input_ids=None,
                position_ids=position_ids_ext,
                attention_mask=attention_mask_ext,
                inputs_embeds=inputs_embeds_ext,
                output_hidden_states=False,
                return_dict=True,
            )

            # Predicted future visual state
            last_hidden = outputs.last_hidden_state
            z_hat_future = last_hidden[:, -self.num_query_tokens:, :]
            z_hat_future_pooled = z_hat_future.mean(dim=1)

        # ---- Action expert ----
        with torch.autocast("cuda", dtype=torch.float32):
            z_t_token = z_t_pooled.unsqueeze(1)
            z_hat_token = z_hat_future_pooled.unsqueeze(1)
            action_conditioning = torch.cat([z_t_token, z_hat_token], dim=1)

            if state is not None:
                state = torch.from_numpy(np.array(state)).to(
                    device=action_conditioning.device, dtype=action_conditioning.dtype
                )

            pred_actions = self.action_model.predict_action(action_conditioning, state)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_idm.yaml",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    model = Qwen_IDM(cfg)
    print(model)

    # --- Fake data ---
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    future_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "future_image": [future_image],
        "lang": "Pick up the red cube and place it on the plate.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }

    batch = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Forward
    output = model(batch)
    print(f"Total Loss: {output['action_loss'].item():.4f}")
    print(f"Action Loss: {output['action_dit_loss']:.4f}")
    print(f"Visual Pred Loss: {output['visual_pred_loss']:.4f}")

    # Predict
    pred = model.predict_action([sample])
    print(f"Predicted actions shape: {pred['normalized_actions'].shape}")

    # Gradient check
    output["action_loss"].backward()
    assert model.query_tokens.weight.grad is not None, "query_tokens has no gradient!"
    assert next(model.action_model.parameters()).grad is not None, "action_model has no gradient!"
    print("Gradient check passed.")
