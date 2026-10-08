# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Ported from GR00T N1.7 (``Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py``) by the starVLA community.
"""
Gr00tN1d7 Framework
A faithful port of NVIDIA's GR00T N1.7 VLA: Qwen3-VL (Cosmos-Reason2) backbone +
flow-matching (DiT / AlternateVLDiT) action head with multi-embodiment conditioning,
per-dim action masking (padded max_action_dim / max_state_dim), vlln + vl_self_attention
on backbone features, state dropout, and RTC (reactive temporal control) inpainting.

Differences vs the N1.5-based ``QwenGR00T``:
  - Multi-embodiment conditioned state encoder / action encoder / action decoder.
  - ``AlternateVLDiT`` that alternates cross-attention between image and text backbone
    tokens (derived from the VLM ``image_token_id``).
  - ``vlln`` (LayerNorm) + ``vl_self_attention`` on backbone features.
  - State dropout (training only).
  - RTC inpainting at inference (pass ``options`` + ``rtc_actions`` via predict_action).
  - N1.7 flow-matching noise schedule (``t = (1 - sample) * noise_s``).
  - Per-dim ``action_mask`` loss with padded ``max_action_dim`` / ``max_state_dim``.

StarVLA raw-examples contract → N1.7 tensor contract (handled in forward/predict_action):
  - ``embodiment_id``: absent in current dataloaders → defaults to ``0`` per sample.
    Read ``example.get("embodiment_id", 0)`` so multi-embodiment dataloaders can supply it.
  - ``action_mask``: derived from the real ``action_dim`` vs padded ``max_action_dim``
    (ones for real dims, zeros for padding). Overridable via ``example["action_mask"]``.
  - ``state``: StarVLA gives ``[B, state_dim]`` or ``[B, T, state_dim]``; zero-padded to
    ``max_state_dim`` and tiled to ``state_history_length``.
  - ``image_mask``: ``(input_ids == image_token_id)``; only used when ``use_alternate_vl_dit``.
"""

import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.action_model.GR00T_N1d7_ActionHeader import get_action_model_n1d7
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


# ──────────────────────────────────────────────────────────────────────
#  Default Config for Gr00tN1d7
#  Mirrors ``Gr00tN1d7Config`` (Isaac-GR00T/gr00t/configs/model/gr00t_n1d7.py).
#  YAML values override these defaults; extra YAML keys are preserved.
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Gr00tN1d7DefaultConfig:
    """Gr00tN1d7 framework default parameters.

    All fields can be overridden by the corresponding key in the YAML ``framework:``
    section. Extra YAML keys not listed here are kept as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "Gr00tN1d7"

    # === VLM backbone (Qwen3-VL / Qwen2.5-VL / Cosmos-Reason2) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
        }
    )

    # === Action head (N1.7 flow-matching / DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT latent shape (inner_dim = num_attention_heads * attention_head_dim = 1536).
            "input_embedding_dim": 1536,
            "hidden_size": 1024,
            # Padded multi-embodiment dimensions (faithful N1.7: 132). The real robot
            # action_dim/state_dim occupy the leading columns; trailing columns are
            # zero-padded and masked out of the loss.
            "max_action_dim": 132,
            "max_state_dim": 132,
            "state_history_length": 1,
            # Real robot dimensions (LIBERO 7-DoF). Override per benchmark.
            "action_dim": 7,
            "state_dim": 7,
            # Canonical chunk length (single source of truth; legacy aliases are
            # normalised upstream by share_tools.apply_config_compat).
            "action_horizon": 16,
            # Backbone-feature post-processing.
            "use_vlln": True,
            "vl_self_attention_cfg": {"num_layers": 0},  # off by default; set num_layers>0 for full N1.7 parity
            # AlternateVLDiT (image/text alternating cross-attention).
            "use_alternate_vl_dit": True,
            "attend_text_every_n_blocks": 2,
            # Positional embedding.
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # State dropout (training-only augmentation). 0.0 = off (default for
            # StarVLA single-frame states); N1.7 uses 0.8.
            "state_dropout_prob": 0.0,
            # Flow-matching noise schedule (N1.7).
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            # Multi-embodiment table size.
            "max_num_embodiments": 32,
            # Trainable-parameter toggles.
            "tune_projector": True,
            "tune_diffusion_model": True,
            "tune_vlln": True,
            # StarVLA efficiency knob: repeat the batch R times with independent noise
            # samples per step (more flow-matching samples per batch). 1 = faithful N1.7.
            "repeated_diffusion_steps": 1,
            # === DiT Transformer sub-config (N1.7 defaults) ===
            "diffusion_model_cfg": {
                "positional_embeddings": None,
                "num_layers": 16,
                "num_attention_heads": 32,
                "attention_head_dim": 48,
                "norm_type": "ada_norm",
                "dropout": 0.2,
                "final_dropout": True,
                "output_dim": 1024,
                "interleave_self_attention": True,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("Gr00tN1d7")
class Gr00tN1d7(baseframework):
    """GR00T N1.7 VLA: Qwen3-VL backbone + flow-matching action head.

    Components:
      - Qwen3-VL / Qwen2.5-VL backbone for fused language/vision token embeddings
      - N1.7 flow-matching (DiT / AlternateVLDiT) action head with multi-embodiment
        conditioning, per-dim action masking, vlln + vl_self_attention, state dropout,
        and RTC inpainting.

    Focus: Predict future continuous actions conditioned on images + instruction,
    faithful to NVIDIA's GR00T N1.7.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(Gr00tN1d7DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Align cross-attention dim to the VLM hidden size BEFORE building the head,
        # mirroring QwenGR00T. The head reads backbone_embedding_dim / cross_attention_dim.
        vl_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        am = self.config.framework.action_model
        am.backbone_embedding_dim = vl_hidden_size
        am.diffusion_model_cfg["cross_attention_dim"] = vl_hidden_size

        self.action_model = get_action_model_n1d7(config=self.config)

        # Cache key dims.
        self.action_horizon = int(am.action_horizon)
        self.max_action_dim = int(am.max_action_dim)
        self.max_state_dim = int(am.max_state_dim)
        self.real_action_dim = int(am.action_dim)
        self.real_state_dim = int(am.state_dim)
        self.state_history_length = int(am.state_history_length)
        self.use_alternate_vl_dit = bool(am.use_alternate_vl_dit)

        # image_token_id for deriving image_mask from input_ids (Qwen3-VL / Qwen2.5-VL).
        self.image_token_id = int(getattr(self.qwen_vl_interface.model.config, "image_token_id", 151655))

    # ── helpers: raw examples → N1.7 padded tensors ─────────────────────
    def _pad_actions(self, actions_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zero-pad [B, H, real_action_dim] → [B, H, max_action_dim]; build action_mask."""
        B, H, D = actions_target.shape
        if D >= self.max_action_dim:
            return actions_target, torch.ones_like(actions_target)
        padded = torch.zeros(B, H, self.max_action_dim, device=actions_target.device, dtype=actions_target.dtype)
        padded[..., :D] = actions_target
        mask = torch.zeros(B, H, self.max_action_dim, device=actions_target.device, dtype=actions_target.dtype)
        mask[..., :D] = 1.0
        return padded, mask

    def _prepare_state(self, state_list: List[np.ndarray], device, dtype) -> Optional[torch.Tensor]:
        """Normalise heterogeneous state arrays to [B, state_history_length, max_state_dim].

        Accepts per-example 1D ``[state_dim]`` or 2D ``[H_s, state_dim]`` arrays; zero-pads
        to ``max_state_dim`` and tiles/truncates the time axis to ``state_history_length``.
        """
        if state_list is None:
            return None
        arr = np.array(state_list)  # [B, ...]
        if arr.ndim == 1:  # single shared vector → broadcast
            arr = arr[None, None, :]
        if arr.ndim == 2:  # [B, state_dim] → [B, 1, state_dim]
            arr = arr[:, None, :]
        # Now arr is [B, H_s, state_dim].
        B, H_s, D = arr.shape
        # Tile/truncate the time axis to state_history_length.
        if H_s < self.state_history_length:
            reps = int(np.ceil(self.state_history_length / H_s))
            arr = np.repeat(arr, reps, axis=1)
        arr = arr[:, -self.state_history_length :, :]
        # Zero-pad state_dim → max_state_dim.
        if D < self.max_state_dim:
            padded = np.zeros((B, self.state_history_length, self.max_state_dim), dtype=arr.dtype)
            padded[..., :D] = arr
            arr = padded
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)

    def _embodiment_ids(self, examples: List[dict], device) -> torch.Tensor:
        return torch.tensor(
            [int(ex.get("embodiment_id", 0)) for ex in examples],
            device=device,
            dtype=torch.long,
        )

    # ── backbone forward (shared by train + infer) ─────────────────────
    def _encode_backbone(self, batch_images, instructions):
        """Run the VLM and return (last_hidden, backbone_attention_mask, image_mask)."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        input_ids = qwen_inputs.get("input_ids", None)
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

        # image_mask: which backbone tokens are image tokens (needed by AlternateVLDiT).
        image_mask = None
        if self.use_alternate_vl_dit and input_ids is not None:
            image_mask = (input_ids == self.image_token_id).to(device=last_hidden.device, dtype=torch.bool)

        return last_hidden, backbone_attention_mask, image_mask

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]  # [B, len, real_action_dim]
        state_list = [example["state"] for example in examples] if "state" in examples[0] else None

        last_hidden, backbone_attention_mask, image_mask = self._encode_backbone(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            device = last_hidden.device
            dtype = last_hidden.dtype

            actions = torch.tensor(np.array(actions), device=device, dtype=dtype)  # [B, T_full, real_D]
            actions_target = actions[:, -self.action_horizon :, :]  # [B, H, real_D]
            actions_padded, action_mask = self._pad_actions(actions_target)  # [B, H, max_D]

            state = self._prepare_state(state_list, device=device, dtype=dtype)  # [B, H_s, max_state] or None
            embodiment_id = self._embodiment_ids(examples, device=device)

            repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 1))
            if repeated_diffusion_steps > 1:
                actions_padded = actions_padded.repeat(repeated_diffusion_steps, 1, 1)
                action_mask = action_mask.repeat(repeated_diffusion_steps, 1, 1)
                last_hidden = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                if backbone_attention_mask is not None:
                    backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1)
                if image_mask is not None:
                    image_mask = image_mask.repeat(repeated_diffusion_steps, 1)
                if state is not None:
                    state = state.repeat(repeated_diffusion_steps, 1, 1)
                embodiment_id = embodiment_id.repeat(repeated_diffusion_steps)

            action_loss = self.action_model(
                last_hidden,
                actions_padded,
                state,
                embodiment_id,
                action_mask,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )["action_loss"]

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: Any,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state_list = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        last_hidden, backbone_attention_mask, image_mask = self._encode_backbone(batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            device = last_hidden.device
            dtype = last_hidden.dtype
            state = self._prepare_state(state_list, device=device, dtype=dtype)
            embodiment_id = self._embodiment_ids(examples, device=device)

            options = kwargs.get("options", None)
            rtc_actions = kwargs.get("rtc_actions", None)
            if rtc_actions is not None and not torch.is_tensor(rtc_actions):
                rtc_actions = torch.tensor(np.array(rtc_actions), device=device, dtype=dtype)

            pred = self.action_model.predict_action(
                last_hidden,
                state,
                embodiment_id,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                options=options,
                rtc_actions=rtc_actions,
            )  # [B, action_horizon, max_action_dim]

        # Drop the multi-embodiment padding → real action_dim.
        pred = pred[:, :, : self.real_action_dim]
        normalized_actions = pred.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/LIBERO/train_files/starvla_gr00t_n1d7_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Gr00tN1d7 = Gr00tN1d7(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
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
    print(f"Predicted normalized action shape: {normalized_actions.shape}")
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
