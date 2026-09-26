# Copyright 2026 NVIDIA Corp. and affiliates. All rights reserved.
# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# Ported from GR00T N1.7 (``Isaac-GR00T/gr00t/model/gr00t_n1d7/gr00t_n1d7.py``) into
# StarVLA's action-head conventions. The architecture / flow-matching math is a
# faithful port of ``Gr00tN1d7ActionHead``; only the I/O surface is adapted to take
# positional tensors (matching StarVLA's existing ``FlowmatchingActionHead``) instead
# of HuggingFace ``BatchFeature`` objects, so the framework can glue raw examples to it.
#
# Key N1.7 additions vs the N1.5 ``FlowmatchingActionHead``:
#   - Multi-embodiment conditioned encoders (``MultiEmbodimentActionEncoder`` +
#     ``CategorySpecificMLP`` state encoder / action decoder).
#   - Optional ``AlternateVLDiT`` that alternates cross-attention between image and
#     text backbone tokens (needs ``image_mask``).
#   - ``vlln`` (LayerNorm) + ``vl_self_attention`` on backbone features.
#   - State dropout during training.
#   - RTC (reactive temporal control) inpainting during inference.
#   - N1.7 flow-matching noise schedule (``t = (1 - sample) * noise_s``).
#   - Per-dim ``action_mask`` loss with padded ``max_action_dim`` / ``max_state_dim``.

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import (
    AlternateVLDiT,
    DiT,
    SelfAttentionTransformer,
)


# ──────────────────────────────────────────────────────────────────────
#  Multi-embodiment building blocks (faithful to N1.7 reference)
# ──────────────────────────────────────────────────────────────────────
class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, hidden_dim] output tensor
        """
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, output_dim] output tensor
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with multi-embodiment support and sinusoidal positional encoding."""

    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        Args:
            actions: [B, T, action_dim] action tensor
            timesteps: [B,] timesteps - a single scalar per batch item
            cat_ids: [B,] category/embodiment IDs
        Returns:
            [B, T, hidden_size] encoded action features
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


# ──────────────────────────────────────────────────────────────────────
#  N1.7 Action Head
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Gr00tN1d7ActionHeadConfig:
    """N1.7 action-head configuration (mirrors ``Gr00tN1d7Config`` action-head fields)."""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion (DiT) model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})
    hidden_size: int = field(default=1024, metadata={"help": "Hidden dim for state/action MLPs."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maximum sequence length for positional embedding."})

    # Padded multi-embodiment dimensions (faithful N1.7: 132). Real robot dims live in
    # the leading columns; trailing columns are zero-padded and masked out of the loss.
    max_action_dim: int = field(default=132, metadata={"help": "Padded action dimension (multi-embodiment)."})
    max_state_dim: int = field(default=132, metadata={"help": "Padded state dimension (multi-embodiment)."})
    state_history_length: int = field(default=1, metadata={"help": "Number of consecutive state timesteps."})

    action_horizon: int = field(default=None, metadata={"help": "Action chunk length the head predicts."})
    num_inference_timesteps: int = field(default=4, metadata={"help": "Euler denoising steps at inference."})

    # Flow-matching noise schedule (N1.7)
    noise_beta_alpha: float = field(default=1.5)
    noise_beta_beta: float = field(default=1.0)
    noise_s: float = field(default=0.999, metadata={"help": "Upper-clip of the sampled timestep."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Discretisation buckets for timestep encoder."})

    # Backbone-feature post-processing
    use_vlln: bool = field(default=True, metadata={"help": "LayerNorm on backbone features."})
    vl_self_attention_cfg: dict = field(
        default=None, metadata={"help": "SelfAttentionTransformer cfg on backbone features."}
    )
    backbone_embedding_dim: int = field(
        default=2048, metadata={"help": "VLM hidden size (cross_attention_dim). Set by the framework."}
    )

    # AlternateVLDiT
    use_alternate_vl_dit: bool = field(
        default=True, metadata={"help": "Use AlternateVLDiT (image/text alternating cross-attn)."}
    )
    attend_text_every_n_blocks: int = field(default=2)

    # State dropout (training-only augmentation)
    state_dropout_prob: float = field(
        default=0.0, metadata={"help": "Probability of zeroing a sample's state features."}
    )

    # Multi-embodiment
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments (category table size)."})

    # Trainable-parameter toggles
    tune_projector: bool = field(default=True, metadata={"help": "Tune state/action encoders/decoders."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Tune the DiT diffusion model."})
    tune_vlln: bool = field(default=True, metadata={"help": "Tune vlln + vl_self_attention."})


class Gr00tN1d7ActionHead(nn.Module):
    """Action head for GR00T N1.7 flow-matching diffusion policy.

    Ported from ``gr00t_n1d7.py:Gr00tN1d7ActionHead``. Unlike the reference's
    ``BatchFeature``-based API, this head takes positional tensors so the StarVLA
    framework can feed it directly. The flow-matching math, multi-embodiment
    conditioning, AlternateVLDiT, vlln/vl_self_attention, state dropout and RTC
    inpainting are preserved verbatim.
    """

    supports_gradient_checkpointing = True

    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        self.full_config = full_config
        self.config = config

        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        # Cross-attention dim = VLM hidden size (set by the framework before building).
        cross_attention_dim = config.get("backbone_embedding_dim", None) or config.diffusion_model_cfg.get(
            "cross_attention_dim", 2048
        )

        # Materialise the DiT cfg and set cross_attention_dim in exactly one place so we
        # never pass it twice (framework may also inject it into diffusion_model_cfg).
        diffusion_model_cfg = dict(config.diffusion_model_cfg)
        diffusion_model_cfg["cross_attention_dim"] = cross_attention_dim

        # DiT backbone (plain DiT, or AlternateVLDiT with image/text alternating cross-attn).
        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **diffusion_model_cfg,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
        else:
            self.model = DiT(**diffusion_model_cfg)

        # Padded dims — encoders/decoders operate on the padded multi-embodiment space.
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.state_history_length = config.state_history_length

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = nn.LayerNorm(cross_attention_dim) if config.use_vlln else nn.Identity()

        vl_self_attention_cfg = config.get("vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly so the noise schedule is
        # identical across SDPA/FA2/meta-device construction contexts (faithful to N1.7).
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model, config.tune_vlln)

    # ── trainable-parameter toggles (faithful to N1.7) ──────────────────
    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self):
        """HuggingFace calls ``model.train()`` each step; keep frozen modules in eval so
        dropout/batchnorm behave deterministically (faithful to N1.7)."""
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        # N1.7 schedule: t = (1 - sample) * noise_s  ∈ [0, noise_s]
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, vl_embs: torch.Tensor) -> torch.Tensor:
        """Apply vlln + vl_self_attention to the backbone (VLM) features."""
        vl_embs = self.vlln(vl_embs)
        vl_embs = self.vl_self_attention(vl_embs)
        return vl_embs

    def _encode_state(self, state: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """Embed state [B, state_history_length, max_state_dim] -> [B, 1, input_embedding_dim]."""
        assert (
            state.shape[1] == self.state_history_length
        ), f"state history length {state.shape[1]} != config.state_history_length {self.state_history_length}"
        state = state.view(state.shape[0], 1, -1)  # [B, 1, state_history_length * max_state_dim]
        state_features = self.state_encoder(state, embodiment_id)  # [B, 1, input_embedding_dim]
        # State dropout (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)
        return state_features

    def _run_model(
        self,
        sa_embs: torch.Tensor,
        vl_embs: torch.Tensor,
        timesteps_tensor: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ) -> torch.Tensor:
        """Dispatch to AlternateVLDiT (needs image_mask) or plain DiT."""
        if self.config.use_alternate_vl_dit:
            return self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                return_all_hidden_states=return_all_hidden_states,
            )
        return self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
            encoder_attention_mask=backbone_attention_mask,
            return_all_hidden_states=return_all_hidden_states,
        )

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        action_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Training forward.

        Args:
            vl_embs: [B, S, backbone_embedding_dim] VLM last hidden states.
            actions: [B, action_horizon, max_action_dim] (zero-padded) target actions.
            state: [B, state_history_length, max_state_dim] (zero-padded) proprioception.
            embodiment_id: [B] long embodiment IDs.
            action_mask: [B, action_horizon, max_action_dim] (1 for real dims, 0 padding).
            image_mask: [B, S] bool (required for AlternateVLDiT).
            backbone_attention_mask: [B, S] bool VLM attention mask.

        Returns:
            ``{"action_loss": scalar}``.
        """
        self.set_frozen_modules_to_eval_mode()

        vl_embs = self.process_backbone_output(vl_embs)
        device = vl_embs.device

        # Embed state (optional — StarVLA examples may omit ``state``).
        state_features = self._encode_state(state, embodiment_id) if state is not None else None

        # Flow-matching noise schedule (N1.7).
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add positional embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join state and action embeddings along the sequence dimension.
        if state_features is not None:
            sa_embs = torch.cat((state_features, action_features), dim=1)
        else:
            sa_embs = action_features

        model_output = self._run_model(
            sa_embs,
            vl_embs,
            timesteps_tensor=t_discretized,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]  # [B, action_horizon, max_action_dim]

        # Per-dim masked MSE (faithful to N1.7).
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        return {"action_loss": loss}

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
        options: Optional[dict[str, Any]] = None,
        rtc_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Flow-matching inference with Euler integration + optional RTC inpainting.

        Args:
            vl_embs: [B, S, backbone_embedding_dim].
            state: [B, state_history_length, max_state_dim].
            embodiment_id: [B] long.
            image_mask / backbone_attention_mask: see :meth:`forward`.
            options: RTC options dict with ``action_horizon``, ``rtc_overlap_steps``,
                ``rtc_frozen_steps``, ``rtc_ramp_rate`` (only used when ``rtc_actions``
                is provided).
            rtc_actions: previous action chunk [B, action_horizon, max_action_dim] for
                reactive temporal control (inpainting). ``None`` disables RTC.

        Returns:
            [B, action_horizon, max_action_dim] predicted (still-normalized) actions.
        """
        self.set_frozen_modules_to_eval_mode()
        vl_embs = self.process_backbone_output(vl_embs)

        batch_size = vl_embs.shape[0]
        device = vl_embs.device

        # Embed state once and reuse across denoising steps (optional).
        state_features = self._encode_state(state, embodiment_id) if state is not None else None

        # Initial actions = sampled noise.
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if rtc_actions is not None:
            # Reactive temporal control: inpaint the previous action chunk instead of
            # starting from pure noise (faithful to N1.7 ``get_action_with_features``).
            assert options is not None, "RTC requires `options` with rtc_* keys."
            assert "action_horizon" in options, "options must contain `action_horizon`."
            assert "rtc_overlap_steps" in options, "options must contain `rtc_overlap_steps`."
            assert "rtc_frozen_steps" in options, "options must contain `rtc_frozen_steps`."
            assert "rtc_ramp_rate" in options, "options must contain `rtc_ramp_rate`."

            action_horizon_before_padding = options["action_horizon"]

            # Use the tail of the previous chunk to inpaint the leading RTC overlap steps.
            actions[:, : options["rtc_overlap_steps"], :] = rtc_actions[
                :,
                action_horizon_before_padding - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            # Freeze the latency-equivalent steps (no denoising update).
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # Exponential ramp strength over the intermediate (unfrozen) overlap steps.
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            t_ramp = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t_ramp)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalise to [0, 1]
            ramp = ramp[1:-1]  # drop the 0.0 and 1.0 endpoints
            vel_strength[:, options["rtc_frozen_steps"] : options["rtc_overlap_steps"], :] = ramp[None, :, None].to(
                device
            )

        # Euler denoising loop.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)

            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            if state_features is not None:
                sa_embs = torch.cat((state_features, action_features), dim=1)
            else:
                sa_embs = action_features

            model_output = self._run_model(
                sa_embs,
                vl_embs,
                timesteps_tensor=timesteps_tensor,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                return_all_hidden_states=False,
            )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration with (optional) RTC velocity strength.
            actions = actions + dt * pred_velocity * vel_strength

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model_n1d7(config=None):
    """Factory: build :class:`Gr00tN1d7ActionHead` from the global framework config.

    The framework must set
    ``config.framework.action_model.backbone_embedding_dim`` (and/or
    ``diffusion_model_cfg.cross_attention_dim``) to the VLM hidden size *before*
    calling this, mirroring :func:`FlowmatchingActionHead.get_action_model`.
    """
    return Gr00tN1d7ActionHead(full_config=config)
