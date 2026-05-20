"""Qwen + Cosmos-Predict2 fusion frameworks for action prediction.

The fusion path keeps Qwen as the main VLM backbone and appends pooled
Cosmos-Predict2 latent tokens to the action head conditioning sequence.
Cosmos acts like an extra physics-aware KV context for the action DiT.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import (
    merge_framework_config,
    populate_layerwise_dit_cfg,
)
from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model as get_gr00t_action_model,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    get_action_model as get_pi_action_model,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model import get_world_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class QwenCosmosGR00TDefaultConfig:
    name: str = "QwenCosmosGR00T"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/qwen35-2B",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
        }
    )

    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Cosmos-Predict2-2B-Video2World",
            "extract_layers": [-1],
            "multiview_mode": "horizontal_concat",
        }
    )

    fusion: dict = field(
        default_factory=lambda: {
            # Pool Cosmos spatial-temporal tokens before concat to control memory.
            "num_cosmos_tokens": 64,
            # Keep Cosmos as a fixed feature extractor by default; Qwen remains trainable.
            "train_cosmos_backbone": False,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "hidden_size": 1024,
            "action_hidden_dim": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 0,
            "action_horizon": 8,
            "repeated_diffusion_steps": 4,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
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


@dataclass
class QwenCosmosPIDefaultConfig:
    name: str = "QwenCosmosPI"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/qwen35-2B",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
            "num_vl_layers": 36,
        }
    )

    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "./playground/Pretrained_models/Cosmos-Predict2-2B-Video2World",
            "extract_layers": [-1],
            "multiview_mode": "horizontal_concat",
        }
    )

    fusion: dict = field(
        default_factory=lambda: {
            "num_cosmos_tokens": 64,
            "train_cosmos_backbone": False,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 0,
            "action_horizon": 8,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "diffusion_model_cfg": {
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


class _CosmosFusionMixin:
    def _init_cosmos_fusion(self):
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.cosmos_backbone = get_world_model(config=self.config)

        qwen_hidden = int(self.qwen_vl_interface.model.config.hidden_size)
        cosmos_hidden = int(self.cosmos_backbone.model.config.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = qwen_hidden

        self.cosmos_norm = nn.LayerNorm(cosmos_hidden)
        self.cosmos_to_qwen = nn.Linear(cosmos_hidden, qwen_hidden, bias=False)
        if cosmos_hidden == qwen_hidden:
            nn.init.eye_(self.cosmos_to_qwen.weight)

        self._train_cosmos_backbone = bool(
            self.config.framework.get("fusion", {}).get("train_cosmos_backbone", False)
        )
        if not self._train_cosmos_backbone:
            self.cosmos_backbone.requires_grad_(False)

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        if target_tokens is None or int(target_tokens) <= 0:
            return tokens
        target_tokens = int(target_tokens)
        if tokens.shape[1] <= target_tokens:
            return tokens
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            tokens.transpose(1, 2).float(),
            target_tokens,
        ).transpose(1, 2)
        return pooled.to(dtype=tokens.dtype)

    def _encode_cosmos_tokens(self, batch_images: List, instructions: List[str], device: torch.device) -> torch.Tensor:
        ctx = torch.enable_grad() if self._train_cosmos_backbone else torch.no_grad()
        with ctx:
            wm_inputs = self.cosmos_backbone.build_inputs(images=batch_images, instructions=instructions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                wm_outputs = self.cosmos_backbone(
                    **wm_inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
                cosmos_tokens = wm_outputs.hidden_states[-1]

        num_cosmos_tokens = self.config.framework.get("fusion", {}).get("num_cosmos_tokens", 64)
        cosmos_tokens = self._pool_tokens(cosmos_tokens, int(num_cosmos_tokens))
        cosmos_tokens = cosmos_tokens.to(device=device, dtype=self.cosmos_to_qwen.weight.dtype)
        cosmos_tokens = self.cosmos_to_qwen(self.cosmos_norm(cosmos_tokens))
        return cosmos_tokens

    def _fuse_last_hidden(self, qwen_hidden: torch.Tensor, batch_images: List, instructions: List[str]) -> torch.Tensor:
        cosmos_tokens = self._encode_cosmos_tokens(batch_images, instructions, qwen_hidden.device)
        cosmos_tokens = cosmos_tokens.to(dtype=qwen_hidden.dtype)
        return torch.cat([qwen_hidden, cosmos_tokens], dim=1)

    def _encode_qwen_last_hidden(self, batch_images: List, instructions: List[str]) -> torch.Tensor:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return qwenvl_outputs.hidden_states[-1]

    def _encode_qwen_layerwise_hidden(self, batch_images: List, instructions: List[str]) -> List[torch.Tensor]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            expected_layers = len(self.action_model.model.transformer_blocks)
            return list(qwenvl_outputs.hidden_states[-expected_layers:])


@FRAMEWORK_REGISTRY.register("QwenCosmosGR00T")
class Qwen_Cosmos_GR00T(_CosmosFusionMixin, baseframework):
    """Qwen-VL + pooled Cosmos-Predict2 latent tokens + GR00T action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenCosmosGR00TDefaultConfig, config)
        self._init_cosmos_fusion()

        qwen_hidden = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = qwen_hidden
        self.action_model: FlowmatchingActionHead = get_gr00t_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_hidden = self._encode_qwen_last_hidden(batch_images, instructions)
        fused_hidden = self._fuse_last_hidden(qwen_hidden, batch_images, instructions)

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=fused_hidden.device, dtype=fused_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = self.config.framework.action_model.get("repeated_diffusion_steps", 4)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            fused_hidden_repeated = fused_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=fused_hidden.device, dtype=fused_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(fused_hidden_repeated, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_hidden = self._encode_qwen_last_hidden(batch_images, instructions)
        fused_hidden = self._fuse_last_hidden(qwen_hidden, batch_images, instructions)

        state = (
            torch.from_numpy(np.array(state)).to(fused_hidden.device, dtype=fused_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(fused_hidden, state)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}


@FRAMEWORK_REGISTRY.register("QwenCosmosPI")
class Qwen_Cosmos_PI(_CosmosFusionMixin, baseframework):
    """Qwen-VL + pooled Cosmos-Predict2 latent tokens + PI layerwise action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenCosmosPIDefaultConfig, config)
        self._init_cosmos_fusion()

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        qwen_hidden = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = qwen_hidden
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=qwen_hidden,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_pi_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

    def _fuse_layerwise_hidden(self, qwen_layers: List[torch.Tensor], batch_images: List, instructions: List[str]):
        cosmos_tokens = self._encode_cosmos_tokens(batch_images, instructions, qwen_layers[-1].device)
        return [torch.cat([hidden, cosmos_tokens.to(dtype=hidden.dtype)], dim=1) for hidden in qwen_layers]

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        qwen_layers = self._encode_qwen_layerwise_hidden(batch_images, instructions)
        fused_layers = self._fuse_layerwise_hidden(qwen_layers, batch_images, instructions)
        base_hidden = fused_layers[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype)
            actions_target = actions[:, -self.action_horizon :, :]

            repeated_diffusion_steps = self.config.framework.action_model.get("repeated_diffusion_steps", 2)
            repeated_diffusion_steps = 2
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            fused_layers_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in fused_layers]

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(fused_layers_repeated, actions_target_repeated, state_repeated)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        qwen_layers = self._encode_qwen_layerwise_hidden(batch_images, instructions)
        fused_layers = self._fuse_layerwise_hidden(qwen_layers, batch_images, instructions)
        base_hidden = fused_layers[-1]

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(fused_layers, state)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}
