from __future__ import annotations

import inspect

import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING

from starVLA.model.modules.vlm.openpi_transformers.gemma.configuration_gemma import GemmaConfig
from starVLA.model.modules.vlm.openpi_transformers.gemma.modeling_gemma import GemmaModel
from starVLA.model.modules.vlm.openpi_transformers.siglip.modeling_siglip import SiglipVisionModel


class OpenPIMultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.linear(image_features)


class OpenPIPaliGemmaCore(nn.Module):
    def __init__(self, hf_config, text_config: GemmaConfig):
        super().__init__()
        self.config = hf_config
        self.vision_tower = SiglipVisionModel(hf_config.vision_config)
        self.multi_modal_projector = OpenPIMultiModalProjector(hf_config)
        self.language_model = GemmaModel(text_config)
        self.vocab_size = text_config.vocab_size
        self.pad_token_id = hf_config.pad_token_id if hf_config.pad_token_id is not None else -1

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_outputs = self.vision_tower(pixel_values)
        return self.multi_modal_projector(image_outputs.last_hidden_state)


class OpenPIPaliGemmaContainer(nn.Module):
    def __init__(self, hf_config, text_config: GemmaConfig):
        super().__init__()
        self.config = hf_config
        self.model = OpenPIPaliGemmaCore(hf_config, text_config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        self.model.language_model.embed_tokens.weight = self.lm_head.weight

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_image_features(pixel_values)


class OpenPIPaliGemma(nn.Module):
    """PaliGemma VLM component used by OpenPI PI0/PI05."""

    def __init__(self, config, *, use_adarms: bool = False, model_name_or_path: str = ""):
        super().__init__()
        hf_config = CONFIG_MAPPING["paligemma"]()
        hf_config._vocab_size = 257152
        hf_config.image_token_index = 257152
        hf_config.text_config.hidden_size = config.width
        hf_config.text_config.intermediate_size = config.mlp_dim
        hf_config.text_config.num_attention_heads = config.num_heads
        hf_config.text_config.head_dim = config.head_dim
        hf_config.text_config.num_hidden_layers = config.depth
        hf_config.text_config.num_key_value_heads = config.num_kv_heads
        hf_config.text_config.hidden_activation = "gelu_pytorch_tanh"
        hf_config.text_config.torch_dtype = "float32"
        hf_config.text_config.vocab_size = 257152
        hf_config.text_config.use_adarms = bool(use_adarms)
        hf_config.text_config.adarms_cond_dim = config.width if use_adarms else None
        hf_config.vision_config.intermediate_size = 4304
        hf_config.vision_config.projection_dim = config.width
        hf_config.vision_config.projector_hidden_act = "gelu_fast"
        hf_config.vision_config.torch_dtype = "float32"

        if model_name_or_path:
            raise ValueError(
                "OpenPIPaliGemma uses StarVLA vendored OpenPI modules. "
                "Load weights through PI0/PI05.load_pretrained_checkpoint instead of model_name_or_path."
            )
        self.model = OpenPIPaliGemmaContainer(
            hf_config,
            self._build_openpi_text_config(config, use_adarms=use_adarms),
        )

    @staticmethod
    def _build_openpi_text_config(config, *, use_adarms: bool):
        return GemmaConfig(
            head_dim=config.head_dim,
            hidden_size=config.width,
            intermediate_size=config.mlp_dim,
            num_attention_heads=config.num_heads,
            num_hidden_layers=config.depth,
            num_key_value_heads=config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=bool(use_adarms),
            adarms_cond_dim=config.width if use_adarms else None,
        )

    @property
    def config(self):
        return self.model.config

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    def embed_image(self, image: torch.Tensor):
        vision_model = self.vision_tower.vision_model
        if hasattr(vision_model, "config"):
            vision_model.config._attn_implementation = "eager"
        try:
            return self.model.get_image_features(image)
        except RuntimeError as exc:
            if image.device.type != "cuda" or "expected scalar type Float but found BFloat16" not in str(exc):
                raise
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return self.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.language_model.embed_tokens(tokens)

    def forward_prefix(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values,
        use_cache: bool | None,
    ):
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if "adarms_cond" in inspect.signature(self.language_model.forward).parameters:
            kwargs["adarms_cond"] = None
        return self.language_model.forward(**kwargs)
