from __future__ import annotations

from dataclasses import dataclass
import inspect

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from starVLA.model.modules.vlm.openpi_transformers.gemma.configuration_gemma import GemmaConfig
from starVLA.model.modules.vlm.openpi_transformers.gemma.modeling_gemma import GemmaForCausalLM


@dataclass
class GemmaDims:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device: torch.device,
) -> Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch_size]")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * torch.pi
    sin_input = scaling_factor[None, :] * time.to(torch.float32)[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha: float, beta: float, bsize: int, device: torch.device) -> torch.Tensor:
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    return torch.distributions.Beta(alpha_t, beta_t).sample((bsize,))


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError(f"Expected 2D masks, got {pad_masks.ndim=} {att_masks.ndim=}")
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def make_att_4d_masks(att_2d_masks: torch.Tensor) -> torch.Tensor:
    return torch.where(att_2d_masks[:, None, :, :], 0.0, -2.3819763e38)



class OpenPIGemma(nn.Module):
    """Gemma transformer wrapper used by OpenPI action heads."""

    def __init__(self, config: GemmaDims, *, use_adarms: bool = False):
        super().__init__()
        hf_config = GemmaConfig(
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
        gemma = GemmaForCausalLM(config=hf_config)
        gemma.model.embed_tokens = None
        self.model = gemma.model
        self.lm_head = gemma.lm_head
        self.validate_transformers_api(use_adarms=use_adarms)
        if use_adarms:
            self._ensure_adarms_norms(config.width)

    def _ensure_adarms_norms(self, cond_dim: int) -> None:
        for layer in self.model.layers:
            self._ensure_adarms_norm(layer.input_layernorm, cond_dim)
            self._ensure_adarms_norm(layer.post_attention_layernorm, cond_dim)
        self._ensure_adarms_norm(self.model.norm, cond_dim)

    @staticmethod
    def _ensure_adarms_norm(norm: nn.Module, cond_dim: int) -> None:
        if getattr(norm, "dense", None) is None:
            width = int(norm.weight.shape[0])
            norm.cond_dim = int(cond_dim)
            norm.dense = nn.Linear(cond_dim, width * 3)
            nn.init.zeros_(norm.dense.weight)
        if hasattr(norm, "weight"):
            norm.weight.data.zero_()
            norm.weight.requires_grad_(False)

    def validate_transformers_api(self, *, use_adarms: bool) -> None:
        if not use_adarms:
            return
        norm_sig = inspect.signature(self.model.layers[0].input_layernorm.forward)
        model_sig = inspect.signature(self.model.forward)
        if "cond" not in norm_sig.parameters or "adarms_cond" not in model_sig.parameters:
            raise RuntimeError(
                "PI05 requires StarVLA vendored OpenPI-compatible Gemma adaRMS support. "
                "The imported Gemma does not expose GemmaRMSNorm.forward(cond=...) and "
                "GemmaModel.forward(adarms_cond=...). "
                f"Current signatures: GemmaRMSNorm.forward{norm_sig}, GemmaModel.forward{model_sig}"
            )


class OpenPIActionHeadBase(nn.Module):
    """OpenPI action expert + action head.

    This module owns the action-side Gemma expert, suffix embedding, action
    projections, and the suffix-only denoise forward used during inference.
    """

    use_adarms: bool = False

    def __init__(self, config: GemmaDims, *, action_dim: int, action_horizon: int, max_state_dim: int):
        super().__init__()
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.max_state_dim = int(max_state_dim)
        self.width = int(config.width)
        self.gemma_expert = OpenPIGemma(config, use_adarms=self.use_adarms)
        self.action_in_proj = nn.Linear(self.action_dim, self.width)
        self.action_out_proj = nn.Linear(self.width, self.action_dim)

    @property
    def model(self):
        return self.gemma_expert.model

    @property
    def layers(self):
        return self.gemma_expert.model.layers

    @property
    def norm(self):
        return self.gemma_expert.model.norm

    def set_gemma_expert_eager_attention(self):
        self.gemma_expert.model.config._attn_implementation = "eager"

    def supports_cached_denoise(self) -> bool:
        return True

    def forward_gemma_expert(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.LongTensor | None,
        past_key_values,
        use_cache: bool | None,
        adarms_cond: torch.Tensor | None,
    ):
        kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }
        if "adarms_cond" in inspect.signature(self.gemma_expert.model.forward).parameters:
            kwargs["adarms_cond"] = adarms_cond
        elif adarms_cond is not None:
            raise RuntimeError(
                "PI05 requires StarVLA vendored OpenPI-compatible Gemma adaRMS support, "
                "but the imported action expert does not expose it."
            )
        return self.gemma_expert.model.forward(**kwargs)

    def sample_action_noise(self, shape, device: torch.device):
        return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device)

    def sample_flow_time(self, bsize: int, device: torch.device):
        return (sample_beta(1.5, 1.0, bsize, device) * 0.999 + 0.001).to(torch.float32)

    def prepare_flow_targets(self, actions: torch.Tensor, noise=None, time=None):
        actions = actions.to(device=self.action_in_proj.weight.device, dtype=torch.float32)
        noise = self.sample_action_noise(actions.shape, actions.device) if noise is None else noise.to(actions.device, torch.float32)
        time = self.sample_flow_time(actions.shape[0], actions.device) if time is None else time.to(actions.device, torch.float32)
        x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
        u_t = noise - actions
        return x_t, u_t, noise, time

    def embed_action_suffix(self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        raise NotImplementedError

    def decode_action_velocity(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        action_hidden_states = action_hidden_states[:, -self.action_horizon :].to(self.action_out_proj.weight.dtype)
        return self.action_out_proj(action_hidden_states).to(torch.float32)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_values,
        suffix_embs: torch.Tensor,
        adarms_cond: torch.Tensor | None,
        force_bfloat16: bool = False,
    ):
        if force_bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)
        self.set_gemma_expert_eager_attention()
        suffix_output = self.forward_gemma_expert(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=suffix_embs,
            use_cache=False,
            adarms_cond=adarms_cond,
        )
        suffix_out = suffix_output.last_hidden_state
        return self.decode_action_velocity(suffix_out)



class OpenPI0ActionHead(OpenPIActionHeadBase):
    """PI0 suffix path: state token plus action-time MLP, no adaRMS conditioning."""

    use_adarms = False

    def __init__(self, config: GemmaDims, *, action_dim: int, action_horizon: int, max_state_dim: int):
        super().__init__(config, action_dim=action_dim, action_horizon=action_horizon, max_state_dim=max_state_dim)
        self.state_proj = nn.Linear(self.max_state_dim, self.width)
        self.action_time_mlp_in = nn.Linear(2 * self.width, self.width)
        self.action_time_mlp_out = nn.Linear(self.width, self.width)

    def embed_action_suffix(self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        state_emb = self.state_proj(state.to(self.state_proj.weight.dtype))
        embs = [state_emb[:, None, :]]
        pad_masks = [torch.ones(state_emb.shape[0], 1, dtype=torch.bool, device=state_emb.device)]
        att_masks = [1]

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.width,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        ).to(self.action_in_proj.weight.dtype)
        action_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = self.action_time_mlp_out(
            F.silu(self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=2)))
        )
        embs.append(action_time_emb)
        pad_masks.append(torch.ones(action_time_emb.shape[:2], dtype=torch.bool, device=action_time_emb.device))
        att_masks += [1] + ([0] * (self.action_horizon - 1))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=embs.device)[None, :]
        return embs, pad_masks, att_masks.expand(embs.shape[0], -1), None


class OpenPI05ActionHead(OpenPIActionHeadBase):
    """PI05 suffix path: action tokens conditioned by adaRMS time embedding."""

    use_adarms = True

    def __init__(self, config: GemmaDims, *, action_dim: int, action_horizon: int, max_state_dim: int):
        super().__init__(config, action_dim=action_dim, action_horizon=action_horizon, max_state_dim=max_state_dim)
        self.time_mlp_in = nn.Linear(self.width, self.width)
        self.time_mlp_out = nn.Linear(self.width, self.width)

    def embed_action_suffix(self, state: torch.Tensor, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.width,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        ).to(self.time_mlp_in.weight.dtype)
        action_time_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))
        adarms_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb))))
        pad_masks = torch.ones(action_time_emb.shape[:2], dtype=torch.bool, device=action_time_emb.device)
        att_masks = torch.tensor([1] + ([0] * (self.action_horizon - 1)), dtype=torch.bool, device=action_time_emb.device)
        return action_time_emb, pad_masks, att_masks[None, :].expand(action_time_emb.shape[0], -1), adarms_cond
