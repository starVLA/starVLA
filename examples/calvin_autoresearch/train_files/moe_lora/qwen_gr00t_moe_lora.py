"""QwenGR00T_MoE_LoRA for CALVIN ABC-only exploration.

This local framework combines GTY's MoE action decoder with the dependency-free
Qwen LoRA hooks implemented in this WMH checkout.  It is intentionally kept
state-compatible with the GTY MoE checkpoints: include_state=false/state_dim=7.
"""

from starVLA.model.framework.VLM4A.QwenGR00T import (
    Qwen_GR00T,
    QwenGR00TDefaultConfig,
    VLMTokenConnector,
    apply_qwen_lora,
    get_vlm_model,
)
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY

from moe.moe_action_head import get_action_model


@FRAMEWORK_REGISTRY.register("QwenGR00T_MoE_LoRA")
class QwenGR00T_MoE_LoRA(Qwen_GR00T):
    """QwenGR00T + GTY MoE action head + trainable Qwen LoRA adapters."""

    def __init__(self, config=None, **kwargs):
        baseframework.__init__(self)
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.qwen_lora_modules = apply_qwen_lora(self.qwen_vl_interface.model, self.config)

        vl_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = vl_hidden_size

        # Keep the parent forward()/predict_action() path compatible with WMH's
        # connector-aware Qwen_GR00T.  The MoE95k-compatible configs leave this
        # disabled, so it introduces no additional trainable parameters.
        self.vl_connector = VLMTokenConnector(
            input_dim=vl_hidden_size,
            config=self.config.framework.get("vl_connector", {}),
        )

        self.action_model = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
