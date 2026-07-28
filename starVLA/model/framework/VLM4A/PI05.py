from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from starVLA.model.framework.VLM4A.PI0 import PI0, PI0DefaultConfig, IMAGE_KEYS
from starVLA.model.modules.action_model.OpenPI_ActionHead import OpenPI05ActionHead
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class PI05DefaultConfig(PI0DefaultConfig):
    name: str = "PI05"
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    action_dim: int = 32
    action_horizon: int = 50
    max_state_dim: int = 32
    discrete_state_input: bool = True
    max_token_len: int = 200
    num_inference_steps: int = 10

    image_resolution: list[int] = field(default_factory=lambda: [224, 224])
    image_keys: list[str] = field(default_factory=lambda: list(IMAGE_KEYS))

    tokenizer: dict = field(default_factory=lambda: {"model_path": "", "pad_state_value": -2.0})
    paligemma: dict = field(default_factory=lambda: {"model_name_or_path": ""})
    paligemma_config: dict = field(
        default_factory=lambda: {
            "width": 2048,
            "depth": 18,
            "mlp_dim": 16384,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )
    action_expert_config: dict = field(
        default_factory=lambda: {
            "width": 1024,
            "depth": 18,
            "mlp_dim": 4096,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )


@FRAMEWORK_REGISTRY.register("PI05")
@FRAMEWORK_REGISTRY.register("Pi05")
class PI05(PI0):
    """OpenPI PI05 model exposed through the StarVLA framework API."""

    default_config_cls = PI05DefaultConfig
    action_head_cls = OpenPI05ActionHead
    model_name = "PI05"
    discrete_state_input = True
