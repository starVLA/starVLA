# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoVLA (VILA) vision-language backbone, vendored natively into starVLA in 2025.
"""EgoVLA / VILA VLM interface.

Assembles the EgoVLA vision-language backbone from standard parts so the whole
model runs inside starVLA's single (transformers 4.57) environment — no
dependency on the EgoVLA_Release / `llava` package:

  * ``llm``          : Qwen2-1.5B  -> native ``AutoModelForCausalLM``
  * ``vision_tower`` : SigLIP-384  -> native ``SiglipVisionModel``
  * ``mm_projector`` : mlp_downsample -> vendored ``EgoVLAMMProjector``

Each part loads directly from an EgoVLA checkpoint laid out VILA-style
(``<ckpt>/{llm,vision_tower,mm_projector}``); the trajectory-decoder action head
lives in ``action_model/EgoVLA_ActionHeader.py`` and is owned by the framework.

Verified against ``ego_vla_checkpoint/ckpt-6720``: native SigLIP loads the
vision_tower and the projector/decoder state-dicts match key-for-key.

``encode_images``: SigLIP ``hidden_states[-2]`` (``cls_patch`` keeps all 729
patch tokens for 384px) -> projector -> 196 tokens x 1536 per image.
"""

import os

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

from .vila_egovla import IGNORE_INDEX, IMAGE_TOKEN_INDEX, EgoVLAMMProjector
from .vila_egovla.arch import prepare_inputs_labels_for_multimodal

logger = initialize_overwatch(__name__)

# EgoVLA vision-tower feature selection (from checkpoint config.json).
_SELECT_LAYER = -2            # mm_vision_select_layer
_SELECT_FEATURE = "cls_patch"  # keep all tokens (SigLIP has no CLS)


class _EgoVLA_VILA_Interface(nn.Module):
    """VILA backbone wrapper exposing the pieces the EgoVLA framework needs."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        base = config.framework.qwenvl.base_vlm  # points at the VILA-style checkpoint dir
        attn = getattr(config.framework.qwenvl, "attn_implementation", "sdpa")
        dtype = torch.bfloat16 if getattr(config.framework.qwenvl, "use_bf16", True) else torch.float32
        self.model_dtype = dtype

        from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel

        llm_dir = os.path.join(base, "llm")
        vt_dir = os.path.join(base, "vision_tower")
        proj_dir = os.path.join(base, "mm_projector")

        # --- LLM (Qwen2) ---
        self.llm = AutoModelForCausalLM.from_pretrained(llm_dir, torch_dtype=dtype, attn_implementation=attn)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_dir)
        self.hidden_size = self.llm.config.hidden_size

        # --- Vision tower (SigLIP) ---
        self.vision_tower = SiglipVisionModel.from_pretrained(vt_dir, torch_dtype=dtype, attn_implementation=attn)
        self.image_processor = SiglipImageProcessor.from_pretrained(vt_dir)

        # --- Projector (vendored mlp_downsample) ---
        self.mm_projector = EgoVLAMMProjector(
            mm_hidden_size=self.vision_tower.config.hidden_size, hidden_size=self.hidden_size
        ).to(dtype)
        proj_weights = os.path.join(proj_dir, "model.safetensors")
        if os.path.exists(proj_weights):
            from safetensors.torch import load_file

            self.mm_projector.load_state_dict(load_file(proj_weights), strict=True)
            logger.info("EgoVLA mm_projector weights loaded from checkpoint.")

    # ------------------------------------------------------------------
    def encode_images(self, images):
        """SigLIP patch features -> projector. images: (N,3,384,384) -> (N,196,H)."""
        outs = self.vision_tower(images, output_hidden_states=True)
        feat = outs.hidden_states[_SELECT_LAYER]
        if _SELECT_FEATURE == "patch":
            feat = feat[:, 1:]
        # "cls_patch": keep all tokens (SigLIP-384 -> 729)
        feat = feat.to(self.mm_projector.layers[1].weight.dtype)
        return self.mm_projector(feat)

    def embed_tokens(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)

    def forward(self, input_ids, images, attention_mask=None, labels=None, position_ids=None):
        """Splice images into text, run the LLM, return last hidden states.

        Returns dict with ``hidden_states`` (B,L,H), ``attention_mask`` (B,L),
        ``labels`` (B,L) — the framework uses ``labels`` to locate action-query
        positions and ``hidden_states`` as both the query latent and decoder memory.
        """
        if images.ndim == 5:  # (B, n_img, C, H, W)
            images = images.flatten(0, 1)
        image_features = self.encode_images(images)

        inputs_embeds, attention_mask, position_ids, labels = prepare_inputs_labels_for_multimodal(
            self.llm,
            self.embed_tokens,
            image_features,
            input_ids,
            attention_mask,
            position_ids,
            labels,
            padding_side=getattr(self.tokenizer, "padding_side", "right"),
        )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        return {
            "hidden_states": outputs.hidden_states[-1],
            "attention_mask": attention_mask,
            "labels": labels,
        }
