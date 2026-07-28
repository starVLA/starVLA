# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoVLA framework — adds EgoVLA (VILA-based bimanual VLA) as a selectable
# starVLA backbone, alongside QwenOFT / QwenGR00T / PI0 / PI05, in 2025.
"""EgoVLA framework (``framework.name: EgoVLA``).

EgoVLA is a VILA-based bimanual VLA: SigLIP-384 vision tower + Qwen2-1.5B LLM +
an ``mlp_downsample`` projector + a transformer trajectory decoder that predicts
a 48-dim per-step MANO / camera-frame action (per hand: 3 wrist-translation +
6 rot6d + 15 MANO pose) over a chunk of future steps.

Native to starVLA — no dependency on the EgoVLA_Release / ``llava`` package:
  * vision tower  : ``SiglipVisionModel``           (transformers, native)
  * LLM           : ``Qwen2ForCausalLM``            (transformers, native)
  * projector     : ``EgoVLAMMProjector``           (vendored, modules/vlm/vila_egovla)
  * action head   : ``EgoVLATrajDecoder``           (vendored, action_model/EgoVLA_ActionHeader)

The whole pipeline (encode_images -> multimodal splice -> LLM -> action-query
latent -> traj decoder) was verified end-to-end with the public
``ego_vla_checkpoint`` weights.

Contract (starVLA framework):
  forward(examples)        -> {"action_loss": scalar}          # L1 on 48-dim action
  predict_action(examples) -> {"normalized_actions": (B,T,48)} # inference

``examples`` items: ``image`` (List[PIL]), ``lang`` (str), ``action`` ([T,48]),
and camera-frame proprio (``state`` / ``proprio*``). Action-query positions are
marked by placeholder token ids in ``[input_placeholder_end_token_idx,
input_placeholder_start_token_idx]`` (EgoVLA: 151195..151375), matched by the
decoder's ``output_mask`` — identical to EgoVLA's own forward.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.EgoVLA_ActionHeader import get_action_model
from starVLA.model.modules.vlm.VILA import _EgoVLA_VILA_Interface
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


@dataclass
class EgoVLADefaultConfig:
    """EgoVLA framework defaults (YAML ``framework:`` overrides these)."""

    name: str = "EgoVLA"

    # VILA backbone: base_vlm points at a VILA-style checkpoint dir with
    # {llm, vision_tower, mm_projector} sub-folders.
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/ego_vla_checkpoint/ckpt-6720",
            "attn_implementation": "sdpa",
            "use_bf16": True,
        }
    )

    # Trajectory-decoder action head (see EgoVLA_ActionHeader.get_action_model).
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "EgoVLATrajDecoder",
            "action_dim": 48,           # 2 hands x (3 trans + 6 rot6d + 15 MANO)
            "action_hidden_dim": 1536,  # Qwen2-1.5B hidden (aligned at runtime)
            "action_horizon": 30,       # future steps per chunk
            "proprio_size": 16,
            "use_proprio": True,
            "sep_proprio": True,
            "traj_decoder_type": "transformer_split_action_v2",
        }
    )

    # Action-query placeholder token id range (EgoVLA action_tokenizer).
    input_placeholder_end_token_idx: int = 151195
    input_placeholder_start_token_idx: int = 151375
    # Prompt scaffold; the action-query block is appended as `action_query_id` * (2*horizon).
    action_query_token_id: int = 151300


@FRAMEWORK_REGISTRY.register("EgoVLA")
class EgoVLA(baseframework):
    """EgoVLA VILA backbone + trajectory-decoder head as a starVLA framework."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(EgoVLADefaultConfig, config)

        # VLM backbone (native SigLIP + Qwen2 + vendored projector).
        self.vlm = _EgoVLA_VILA_Interface(self.config)
        # Align the decoder hidden dim to the actual LLM hidden size.
        self.config.framework.action_model.action_hidden_dim = self.vlm.hidden_size

        # Trajectory-decoder action head (vendored, checkpoint-compatible).
        # Match the backbone dtype so bf16 hidden states / proprio flow through it.
        self.action_model = get_action_model(config=self.config).to(self.vlm.model_dtype)

        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        fw = self.config.framework
        self._q_lo = int(getattr(fw, "input_placeholder_end_token_idx", 151195))
        self._q_hi = int(getattr(fw, "input_placeholder_start_token_idx", 151375))
        self._q_id = int(getattr(fw, "action_query_token_id", 151300))
        self.tokenizer = self.vlm.tokenizer
        self.l1_loss = nn.L1Loss()

    # ------------------------------------------------------------------
    # Input construction (StarVLA examples -> EgoVLA model inputs)
    # ------------------------------------------------------------------
    def _build_inputs(self, examples):
        """Tokenize instruction + <image> + action-query block; assemble proprio.

        Produces (input_ids, attention_mask, labels, images, proprio_dict). The
        action-query block is ``2*action_horizon`` placeholder tokens (left/right
        interleaved) whose label ids sit in the query range so the decoder's
        ``output_mask`` selects exactly those hidden states.
        """
        device = next(self.parameters()).device
        n_query = 2 * self.action_horizon
        ids_list, labels_list, imgs_list = [], [], []

        for ex in examples:
            instr = ex["lang"]
            tok = self.tokenizer(instr, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            # [text tokens] <image> [text?] then N_QUERY action-query tokens.
            seq = torch.cat([
                tok,
                torch.tensor([IMAGE_TOKEN_INDEX], dtype=torch.long),
                torch.full((n_query,), self._q_id, dtype=torch.long),
            ])
            lab = torch.full_like(seq, IGNORE_INDEX)
            lab[-n_query:] = self._q_id  # query positions carry their id
            ids_list.append(seq)
            labels_list.append(lab)
            imgs_list.append(ex["image"])

        max_len = max(len(s) for s in ids_list)
        B = len(ids_list)
        input_ids = torch.full((B, max_len), self.tokenizer.pad_token_id or 0, dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.bool)
        for i, (s, l) in enumerate(zip(ids_list, labels_list)):
            input_ids[i, : len(s)] = s
            labels[i, : len(l)] = l
            attn[i, : len(s)] = True

        # Images: List[List[PIL]] -> (B, n_img, 3, 384, 384) via SigLIP processor.
        flat = [im for sub in imgs_list for im in (sub if isinstance(sub, (list, tuple)) else [sub])]
        px = self.vlm.image_processor(images=flat, return_tensors="pt")["pixel_values"]
        n_img = len(flat) // B
        images = px.view(B, n_img, *px.shape[1:])

        proprio = self._build_proprio(examples, device)
        return (input_ids.to(device), attn.to(device), labels.to(device),
                images.to(device, next(self.vlm.parameters()).dtype), proprio)

    def _build_proprio(self, examples, device):
        """Assemble the sep-proprio dict (camera-frame) from the example.

        Reads explicit ``proprio_3d`` / ``proprio_rot`` / ``proprio_hand_finger_tip``
        if the DataConfig provides them; otherwise zeros (model still runs).
        """
        dt = next(self.vlm.parameters()).dtype
        B = len(examples)

        def stack(key, dim):
            if key in examples[0] and examples[0][key] is not None:
                return torch.tensor(np.array([np.asarray(e[key]).reshape(-1) for e in examples]),
                                    device=device, dtype=dt)
            return torch.zeros((B, dim), device=device, dtype=dt)

        return {
            "proprio": stack("proprio", 16),
            "proprio_3d": stack("proprio_3d", 6),
            "proprio_rot": stack("proprio_rot", 6),
            "proprio_hand_finger_tip": stack("proprio_hand_finger_tip", 30),
        }

    # ------------------------------------------------------------------
    def _run(self, examples):
        """Shared pipeline: backbone -> action-query latent -> traj decoder pred."""
        input_ids, attn, labels, images, proprio = self._build_inputs(examples)
        out = self.vlm(input_ids, images, attention_mask=attn, labels=labels)
        hidden, attn2, labels2 = out["hidden_states"], out["attention_mask"], out["labels"]

        output_mask = (labels2 >= self._q_lo) & (labels2 <= self._q_hi)
        latent = hidden[output_mask]
        memory_mask = torch.where(output_mask, torch.zeros_like(attn2), attn2)
        result = self.action_model(latent, proprio, memory=hidden, memory_mask=memory_mask)
        return result["pred"]  # (B*T, 48)

    def forward(self, examples: List[dict] = None, **kwargs):
        pred = self._run(examples)  # (B*T, 48)
        B = len(examples)
        pred = pred.reshape(B, self.action_horizon, -1)
        actions = torch.tensor(np.array([np.asarray(e["action"]) for e in examples]),
                               device=pred.device, dtype=pred.dtype)
        target = actions[:, -self.action_horizon :, :]
        return {"action_loss": self.l1_loss(pred, target)}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        train_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_size:
            for e in examples:
                if isinstance(e.get("image"), list):
                    e["image"] = resize_images(e["image"], target_size=train_size)
        pred = self._run(examples)
        B = len(examples)
        pred = pred.reshape(B, self.action_horizon, -1)
        return {"normalized_actions": pred.detach().float().cpu().numpy()}
