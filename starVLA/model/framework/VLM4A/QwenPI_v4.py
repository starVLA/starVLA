"""QwenPI_v4: Qwen VLM with a fused self/cross-attention layer-wise DiT.

QwenPI_v4 keeps the QwenPI_v3 VLM input, layer selection, projectors, flow
matching loss, sampler, and VLM-token masking.  Its action head is separate:
every DiT block uses action/state tokens as queries and the concatenation of
action/state and VLM tokens as keys/values, followed by an FFN.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.framework.VLM4A.QwenPI_v3 import (
    QwenPI_v3DefaultConfig,
    Qwen_PI_v3,
)
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader_v4 import (
    LayerwiseFlowmatchingActionHeadV4,
    get_action_model_v4,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenPI_v4DefaultConfig(QwenPI_v3DefaultConfig):
    """Defaults for the fixed-topology QwenPI_v4 action head."""

    name: str = "QwenPI_v4"
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM_v4",
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 16,
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
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v4")
class Qwen_PI_v4(Qwen_PI_v3):
    """Qwen VLM + independent fused self/cross-attention DiT action head."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        # QwenPI_v3's methods are reused for the VLM-side data flow, but its
        # constructor is intentionally not called because it would build the
        # legacy action head before we can install the v4 head.
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenPI_v4DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHeadV4 = get_action_model_v4(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
