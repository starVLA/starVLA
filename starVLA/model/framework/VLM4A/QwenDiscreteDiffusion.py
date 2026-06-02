# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
"""
QwenDiscreteDiffusion Framework
Qwen2.5-VL backbone + MaskGIT-style discrete-diffusion action head.

Layer-wise cross-attention over multi-layer VLM hidden states, same shape
as QwenPI, but the continuous flow-matching head is replaced by a
discrete-diffusion head that predicts logits over a uniform action binning.

The same wrapper exposes:
  - forward(...) for training
  - predict_action(...) for single-shot MaskGIT decode
  - predict_action_realtime(...) for real-time chunking (RTC): pin the
    un-executed tail of the previous chunk as a known prefix and only
    decode the trailing execution_horizon positions.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import (
    merge_framework_config,
    populate_layerwise_dit_cfg,
)
from starVLA.model.modules.action_model.LayerwiseDiscreteDiffusion_ActionHeader import (
    LayerwiseDiscreteDiffusionActionHead,
    get_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenDiscreteDiffusion
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenDiscreteDiffusionDefaultConfig:
    """QwenDiscreteDiffusion framework default parameters.

    Layer-wise cross-DiT discrete-diffusion (MaskGIT-style) action prediction
    conditioned on multi-layer VLM hidden states. All fields can be overridden
    by the corresponding key in the YAML ``framework:`` section.
    """

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "QwenDiscreteDiffusion"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct",
            "attn_implementation": "flash_attention_2",
            # Auto-set at runtime from VLM HF config.
            "vl_hidden_dim": 2048,
            "num_vl_layers": 36,
        }
    )

    # === Action head (Layer-wise cross-DiT discrete diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseDD",
            # Action / state dims and chunk length.
            "action_dim": 7,
            "state_dim": 7,
            # Canonical chunk length. Legacy YAMLs may use
            # future_action_window_size = action_horizon - 1; apply_config_compat
            # normalises both directions.
            "action_horizon": 16,
            # Repeat factor used during training (matches QwenPI).
            "repeated_diffusion_steps": 2,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            # --- Discrete-diffusion specific ---
            # "bin": num_bins logits per (T, D) position, CE loss.
            # "bit": num_bits=8 sigmoid bits per position, BCE loss; 8 bits
            #        simulate 256 bins. Requires num_bins == 256.
            "representation": "bin",
            "num_bins": 256,
            "action_low": -1.0,
            "action_high": 1.0,
            # MaskGIT decode rounds (used by predict_action and as the basis
            # for predict_action_realtime's scaled-step heuristic).
            "num_inference_steps": 8,
            "train_mask_schedule": "cosine",  # "cosine" | "linear"
            "decode_schedule": "cosine",
            "no_mask_token_prob": 0.0,
            # Auxiliary L1 on the decoded continuous prediction, keeps bin
            # centers calibrated.
            "l1_loss_weight": 0.1,
            # If True, replace iterative decode with a single forward + argmax.
            "use_simple_max": False,
            # DiT architecture settings — shape fields (num_layers,
            # input_embedding_dim, cross_attention_dim, num_attention_heads)
            # are auto-populated by populate_layerwise_dit_cfg at runtime.
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": "sinusoidal",
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenDiscreteDiffusion")
class Qwen_DiscreteDiffusion(baseframework):
    """
    VLA with MaskGIT-style discrete-diffusion action head.
    Layer-wise cross-attention over multi-layer VLM hidden states (QwenPI shape).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(QwenDiscreteDiffusionDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts
        # it on the top-level config — getattr handles both.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # DiT runs at the LLM hidden size (no compression).
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=llm_hidden_size,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseDiscreteDiffusionActionHead = get_action_model(config=self.config)

        # action_horizon is the single source of truth for chunk length.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _encode_vl_hidden_states(self, batch_images: List, instructions: List[str]) -> List[torch.Tensor]:
        """Run QwenVL and return the last-N layer-wise hidden states for the Action DiT."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(qwenvl_outputs.hidden_states[-expected_layers:])
        return vl_embs_list

    def forward(
        self,
        examples: Optional[List[dict]] = None,
        **kwargs,
    ) -> dict:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
                - state: optional np.ndarray [state_dim]
        Returns:
            dict: action_loss (torch.Tensor)
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Step 1: encode through QwenVL
        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        # Step 2: Action head forward + loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 2))
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            pred, target, extra = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
            )
            action_loss = self.action_model.loss(pred, target, **extra)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: Optional[List[dict]] = None,
        **kwargs,
    ) -> dict:
        """
        Inference: MaskGIT-style iterative decode to future actions.

        Returns:
            dict: normalized_actions (np.ndarray) [B, action_horizon, action_dim]
        """
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )

        decode_temperature = kwargs.get("decode_temperature", 0.1)
        choice_temperature = kwargs.get("choice_temperature", 0.1)
        use_simple_max = kwargs.get(
            "use_simple_max",
            bool(self.config.framework.action_model.get("use_simple_max", False)),
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state,
                choice_temperature=choice_temperature,
                decode_temperature=decode_temperature,
                use_simple_max=use_simple_max,
            )

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_action_realtime(
        self,
        examples: Optional[List[dict]] = None,
        prev_action_chunk_normalized: np.ndarray = None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        """
        RTC-aware inference: the previous chunk's un-executed tail is encoded
        into bins and pinned as a known prefix; only the trailing
        ``execution_horizon`` positions are masked and resampled.

        Args:
            prev_action_chunk_normalized: (B, action_horizon, action_dim)
                *normalized* continuous actions from the previous prediction.
            inference_delay: fallback for ``execution_horizon`` when not given;
                also used as prefix length under ``hard_mask=True``.
            execution_horizon (kw): number of trailing positions to regenerate.
                Defaults to ``inference_delay``.
            hard_mask (kw): if True, prefix length = ``inference_delay``
                (legacy). If False (default), prefix length =
                ``action_horizon - execution_horizon``.
            fixed_steps (kw): always run ``num_inference_steps``; otherwise
                scale step count by mask fraction.
            choice_temperature / decode_temperature: MaskGIT sampling.
            early_stop (kw): exit early once all non-prefix slots are unmasked.

        Returns:
            dict: normalized_actions (np.ndarray) [B, action_horizon, action_dim]
        """
        if prev_action_chunk_normalized is None or inference_delay <= 0:
            return self.predict_action(examples, **kwargs)

        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        vl_embs_list = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]

        state_t = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )

        prev_chunk_t = torch.from_numpy(np.array(prev_action_chunk_normalized)).to(
            base_hidden.device, dtype=torch.float32
        )

        decode_temperature = kwargs.get("decode_temperature", 0.1)
        choice_temperature = kwargs.get("choice_temperature", 0.1)
        fixed_steps = kwargs.get("fixed_steps", False)
        execution_horizon = kwargs.get("execution_horizon", None)
        hard_mask = kwargs.get("hard_mask", False)
        early_stop = kwargs.get("early_stop", False)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action_realtime(
                vl_embs_list,
                state_t,
                prev_action_chunk=prev_chunk_t,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                choice_temperature=choice_temperature,
                decode_temperature=decode_temperature,
                fixed_steps=fixed_steps,
                hard_mask=hard_mask,
                early_stop=early_stop,
            )

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
