# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""

import sys
from pathlib import Path
import re

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenGR00T
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenGR00TDefaultConfig:
    """QwenGR00T framework default parameters.

    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.  Extra YAML keys not listed here are kept
    as-is (Config-as-API flexibility).
    """

    # --- Registry identifier ---
    name: str = "QwenGR00T"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            # Path to base VLM checkpoint (local or HF hub id)
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            # Attention implementation: "flash_attention_2" | "eager" | "sdpa"
            "attn_implementation": "flash_attention_2",
            # VLM hidden dimension (used for cross-attention alignment)
            "vl_hidden_dim": 2048,
        }
    )

    # # === DINO encoder (optional multi-view spatial tokens) === Dino is not used in this QwenGR00T version, we can add it later when we want to use it
    # dino: dict = field(default_factory=lambda: {
    #     # DINO backbone variant: "dinov2_vits14" | "dinov2_vitb14" | ...
    #     "dino_backbone": "dinov2_vits14",
    # })

    # === Action head (Flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            # DiT model size: "DiT-B" | "DiT-L" | "DiT-XL"
            "action_model_type": "DiT-B",
            # Hidden dim for action model (auto-aligned at runtime)
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            # Whether to add positional embeddings in the action head
            "add_pos_embed": True,
            "max_seq_len": 1024,
            # Dimensionality of each action vector (e.g., 7 for 6-DoF + gripper)
            "action_dim": 7,
            # State dimension (proprioception input)
            "state_dim": 7,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 8,
            # Repeat factor for flow-matching loss (more noise samples per batch)
            "repeated_diffusion_steps": 8,
            # Beta distribution params for noise schedule
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Inference denoising steps
            "num_inference_timesteps": 4,
            # Number of vision tokens fed to action head
            "num_target_vision_tokens": 32,
            # === DiT Transformer sub-config ===
            "diffusion_model_cfg": {
                # Cross-attention dim (aligned to VLM hidden_size at runtime)
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # === Lightweight VLM-to-action connector ===
    vl_connector: dict = field(
        default_factory=lambda: {
            # Disabled by default so old configs/checkpoints remain strict-load compatible.
            "enabled": False,
            # "residual_mlp" keeps the original Qwen hidden states and adds a small adapter delta.
            "type": "residual_mlp",
            # Bottleneck width for the MLP adapter.
            "hidden_dim": 512,
            "num_layers": 2,
            "dropout": 0.0,
            "residual_scale": 1.0,
            # With residual_mlp this starts exactly as identity and learns the adapter gradually.
            "zero_init": True,
        }
    )

    # # === Training precision flag === This is unnecessary, unused parameter
# reduce_in_full_precision: bool = True


class VLMTokenConnector(nn.Module):
    """Small trainable adapter between Qwen hidden states and the action head."""

    def __init__(self, input_dim: int, config: Optional[dict] = None) -> None:
        super().__init__()
        config = config or {}
        connector_type = str(config.get("type", "residual_mlp")).lower()
        self.enabled = bool(config.get("enabled", False)) and connector_type not in {"identity", "none"}
        self.residual = connector_type in {"residual_mlp", "residual"}
        self.residual_scale = float(config.get("residual_scale", 1.0))

        if not self.enabled:
            self.net = nn.Identity()
            return

        if connector_type not in {"mlp", "residual_mlp", "residual"}:
            raise ValueError(f"Unsupported vl_connector.type={connector_type!r}")

        hidden_dim = int(config.get("hidden_dim", input_dim))
        num_layers = max(1, int(config.get("num_layers", 2)))
        dropout = float(config.get("dropout", 0.0))

        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, input_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.extend([nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)])
            layers.extend([nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim)])

        self.net = nn.Sequential(*layers)
        if self.residual and bool(config.get("zero_init", True)):
            final_linear = next((layer for layer in reversed(layers) if isinstance(layer, nn.Linear)), None)
            if final_linear is not None:
                nn.init.zeros_(final_linear.weight)
                if final_linear.bias is not None:
                    nn.init.zeros_(final_linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return hidden_states
        delta = self.net(hidden_states)
        if self.residual:
            return hidden_states + self.residual_scale * delta
        return delta


class LoRALinear(nn.Module):
    """Dependency-free LoRA wrapper for exploratory Qwen adaptation."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = False

        device = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))

        self.lora_A._starvla_trainable_after_freeze = True
        self.lora_B._starvla_trainable_after_freeze = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_x = self.dropout(x).to(dtype=self.lora_A.dtype)
        update = torch.matmul(torch.matmul(lora_x, self.lora_A.t()), self.lora_B.t()) * self.scaling
        return base_out + update.to(dtype=base_out.dtype)


def _as_list(value, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _get_nested_module(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def apply_qwen_lora(qwen_model: nn.Module, config) -> list[str]:
    """Attach LoRA to selected Qwen Linear modules before optimizer creation."""

    lora_cfg = config.framework.get("qwen_lora", {}) if config is not None else {}
    if not bool(lora_cfg.get("enabled", False)):
        return []

    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.05))
    target_modules = set(_as_list(lora_cfg.get("target_modules", None), ["q_proj", "v_proj"]))
    last_n_layers = int(lora_cfg.get("last_n_layers", 0))
    max_modules = int(lora_cfg.get("max_modules", 0))

    candidates: list[tuple[str, nn.Linear, Optional[int]]] = []
    layer_indices: list[int] = []
    layer_re = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")

    for name, module in qwen_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in target_modules:
            continue
        match = layer_re.search(name)
        layer_idx = int(match.group(1)) if match else None
        if layer_idx is not None:
            layer_indices.append(layer_idx)
        candidates.append((name, module, layer_idx))

    if last_n_layers > 0 and layer_indices:
        min_layer = max(layer_indices) - last_n_layers + 1
        candidates = [(name, module, idx) for name, module, idx in candidates if idx is not None and idx >= min_layer]

    if max_modules > 0:
        candidates = candidates[:max_modules]

    applied: list[str] = []
    for name, module, _ in candidates:
        parent_path, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = _get_nested_module(qwen_model, parent_path) if parent_path else qwen_model
        if isinstance(getattr(parent, child_name), LoRALinear):
            continue
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        applied.append(name)

    if not applied:
        raise RuntimeError(
            "qwen_lora.enabled=true but no target Linear modules were matched. "
            f"target_modules={sorted(target_modules)}, last_n_layers={last_n_layers}"
        )

    print(
        f"[qwen_lora] enabled rank={rank} alpha={alpha} dropout={dropout} "
        f"targets={sorted(target_modules)} applied={len(applied)}"
    )
    for name in applied[:20]:
        print(f"[qwen_lora]   {name}")
    if len(applied) > 20:
        print(f"[qwen_lora]   ... {len(applied) - 20} more")

    return applied


@FRAMEWORK_REGISTRY.register("QwenGR00T")
class Qwen_GR00T(baseframework):
    """
    Multimodal vision-language-action model (GR00T variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
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
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.qwen_lora_modules = apply_qwen_lora(self.qwen_vl_interface.model, self.config)
        # align dims --> we should put them to config or no?
        vl_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = vl_hidden_size
        self.vl_connector = VLMTokenConnector(
            input_dim=vl_hidden_size,
            config=self.config.framework.get("vl_connector", {}),
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _apply_vl_connector(self, hidden_states: torch.Tensor) -> torch.Tensor:
        connector = getattr(self, "vl_connector", None)
        if connector is None or not getattr(connector, "enabled", False):
            return hidden_states
        with torch.autocast(hidden_states.device.type, dtype=torch.bfloat16, enabled=hidden_states.is_cuda):
            return connector(hidden_states)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """ """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

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
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
        last_hidden = self._apply_vl_connector(last_hidden)

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -self.action_horizon :, :]  # (B, action_horizon, action_dim)

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

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
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
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

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
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]
        last_hidden = self._apply_vl_connector(last_hidden)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model: Qwen_GR00T = Qwen_GR00T(cfg)
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
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
