# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# RoboTTT action head (arxiv 2607.15275, "RoboTTT: Context Scaling for Robot Policies").
#
# Extends the N1.7 flow-matching head (:class:`Gr00tN1d7ActionHead`) with:
#   - :class:`RoboTTTDiT` (a TTT layer after each DiT attention block) operating across
#     the trajectory time dimension.
#   - per-timestep register tokens that carry VL information across time.
#   - **sequence action forcing** (independent flow-matching noise per timestep).
#   - **TBPTT** (fast weights carried across segment boundaries, gradients detached).
#   - an optional ``loss_mask`` to make selected timesteps context-only (for in-context
#     video imitation / DAgger Distillation, RoboTTT §3.3).
#   - an inference path that rolls the fast weights over a context trajectory, then
#     denoises the current action chunk reusing the carried fast weights.
#
# The flow-matching math, multi-embodiment conditioning, vlln/vl_self_attention, state
# dropout and per-dim action masking are inherited unchanged from the N1.7 head.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import RoboTTTDiT
from starVLA.model.modules.action_model.GR00T_N1d7_ActionHeader import Gr00tN1d7ActionHead


class RoboTTTActionHead(Gr00tN1d7ActionHead):
    """RoboTTT action head: N1.7 flow-matching DiT + TTT layers across the trajectory.

    ``forward_sequence`` processes a full trajectory ``[B, T_time, ...]``: attention runs
    within each timestep, TTT compresses the history across timesteps into fast weights,
    and a flow-matching loss is computed per timestep (with sequence action forcing +
    TBPTT). ``predict_action`` rolls the fast weights over a context trajectory then
    denoises the current action chunk.
    """

    def __init__(self, full_config):
        super().__init__(full_config)  # builds the N1.7 encoders/decoders/vlln/beta_dist + a DiT
        config = full_config.framework.action_model

        # Replace the DiT with a TTT-augmented RoboTTTDiT (same block cfg + per-block TTT).
        cross_attention_dim = config.get("backbone_embedding_dim", None) or config.diffusion_model_cfg.get(
            "cross_attention_dim", 2048
        )
        diffusion_model_cfg = dict(config.diffusion_model_cfg)
        diffusion_model_cfg["cross_attention_dim"] = cross_attention_dim
        ttt_cfg = dict(config.get("ttt_cfg", {}) or {})
        self.model = RoboTTTDiT(
            **diffusion_model_cfg,
            ttt_cfg=ttt_cfg,
            use_alternate_vl_dit=config.use_alternate_vl_dit,
            attend_text_every_n_blocks=config.attend_text_every_n_blocks,
        )
        self.tbptt_segment_length = config.get("tbptt_segment_length", None)

        # Per-timestep register tokens (RoboTTT §3.1): prepended at each timestep, attend
        # to all other tokens; carry VL information across time through the TTT layers.
        self.num_registers = int(config.get("num_registers", 4))
        self.register_tokens = nn.Parameter(0.02 * torch.randn(self.num_registers, self.input_embedding_dim))

        # Re-apply the trainable-parameter toggles to the new self.model + registers.
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model, config.tune_vlln)

    # ── helpers ─────────────────────────────────────────────────────────
    def _process_backbone_traj(self, vl_embs: torch.Tensor) -> torch.Tensor:
        """Apply vlln + vl_self_attention to a trajectory ``[B, T, S, D]``."""
        vl_embs = self.vlln(vl_embs)  # LayerNorm works on any trailing-dim shape
        if not isinstance(self.vl_self_attention, nn.Identity):
            B, T, S, D = vl_embs.shape
            vl_embs = self.vl_self_attention(vl_embs.reshape(B * T, S, D)).reshape(B, T, S, D)
        else:
            vl_embs = self.vl_self_attention(vl_embs)
        return vl_embs

    def _encode_state_traj(self, state: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        """state [B, T, Hs, max_S], emb_id [B] -> state features [B, T, 1, input_emb]."""
        B, T, Hs, max_S = state.shape
        flat = state.reshape(B * T, Hs, max_S).view(B * T, 1, -1)  # [B*T, 1, Hs*max_S]
        emb_rep = embodiment_id.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]
        sf = self.state_encoder(flat, emb_rep)  # [B*T, 1, input_emb]
        # State dropout (training only).
        if self.training and self.state_dropout_prob > 0:
            do_drop = torch.rand(sf.shape[0], device=sf.device) < self.state_dropout_prob
            sf = sf * (1 - do_drop[:, None, None].to(sf.dtype))
        return sf.reshape(B, T, 1, -1)

    def _encode_action_traj(
        self, actions: torch.Tensor, t_disc: torch.Tensor, embodiment_id: torch.Tensor
    ) -> torch.Tensor:
        """actions [B, T, H, max_D], t_disc [B, T], emb_id [B] -> [B, T, H, input_emb]."""
        B, T, H, max_D = actions.shape
        flat = actions.reshape(B * T, H, max_D)
        t_flat = t_disc.reshape(-1)  # [B*T]
        emb_rep = embodiment_id.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]
        af = self.action_encoder(flat, t_flat, emb_rep)  # [B*T, H, input_emb]
        return af.reshape(B, T, H, -1)

    def _build_per_timestep_tokens(self, state_features: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """Concatenate [register, proprio, noised-action] per timestep -> [B, T, L, input_emb]."""
        B, T = state_features.shape[:2]
        reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # [B,T,R,input_emb]
        sa_embs = torch.cat([reg, state_features, action_features], dim=2)  # [B,T,R+1+H,input_emb]
        if self.config.add_pos_embed:
            pos_ids = torch.arange(sa_embs.shape[2], dtype=torch.long, device=sa_embs.device)
            sa_embs = sa_embs + self.position_embedding(pos_ids)[None, None]
        return sa_embs

    # ── sequence training forward ───────────────────────────────────────
    def forward_sequence(
        self,
        vl_embs: torch.Tensor,  # [B, T, S, Dvl]
        actions: torch.Tensor,  # [B, T, H, max_D]
        state: Optional[torch.Tensor],  # [B, T, Hs, max_S]
        embodiment_id: torch.Tensor,  # [B]
        action_mask: torch.Tensor,  # [B, T, H, max_D]
        image_mask: Optional[torch.Tensor] = None,  # [B, T, S]
        backbone_attention_mask: Optional[torch.Tensor] = None,  # [B, T, S]
        loss_mask: Optional[torch.Tensor] = None,  # [B, T] 1=imitation target, 0=context-only
    ) -> dict:
        """Trajectory forward with sequence action forcing + TBPTT.

        Returns ``{"action_loss"}`` (per-dim masked flow-matching MSE, averaged over
        timesteps; ``loss_mask`` zeroes context-only timesteps).
        """
        self.set_frozen_modules_to_eval_mode()
        vl_embs = self._process_backbone_traj(vl_embs)  # [B, T, S, Dvl]
        B, T, H, max_D = actions.shape
        device = vl_embs.device
        dtype = actions.dtype

        # Sequence action forcing: independent flow-matching noise per (sample, timestep).
        noise = torch.randn(actions.shape, device=device, dtype=dtype)
        t = self.sample_time(B * T, device=device, dtype=dtype).reshape(B, T, 1, 1)  # [B,T,1,1]
        noisy = (1 - t) * noise + t * actions
        velocity = actions - noise  # [B,T,H,max_D]
        t_disc = (t.squeeze(-1).squeeze(-1) * self.num_timestep_buckets).long()  # [B,T]

        # Per-timestep token embeddings.
        state_features = self._encode_state_traj(state, embodiment_id) if state is not None else None
        action_features = self._encode_action_traj(noisy, t_disc, embodiment_id)  # [B,T,H,input_emb]
        if state_features is None:
            # No proprioception: prepend registers + action only.
            reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
            sa_embs = torch.cat([reg, action_features], dim=2)
        else:
            sa_embs = self._build_per_timestep_tokens(state_features, action_features)

        # RoboTTTDiT over the trajectory (fast weights carried internally; TBPTT detach).
        model_output, _ = self.model(
            hidden_states=sa_embs,  # [B, T, L, input_emb]
            encoder_hidden_states=vl_embs,  # [B, T, S, Dvl]
            timestep=t_disc,  # [B, T]
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
            ttt_states=None,
            tbptt_segment_length=self.tbptt_segment_length,
            update_ttt=True,
        )  # [B, T, L, output_dim]

        # Decode and slice the action portion (last H tokens).
        emb_rep = embodiment_id.unsqueeze(1).expand(-1, T).reshape(-1)  # [B*T]
        L = model_output.shape[2]
        out_flat = model_output.reshape(B * T, L, -1)  # [B*T, L, output_dim==hidden_size]
        pred = self.action_decoder(out_flat, emb_rep)  # [B*T, L, max_D]
        pred = pred.reshape(B, T, L, max_D)
        pred_actions = pred[:, :, -H:, :]  # [B, T, H, max_D]

        # Per-dim masked flow-matching loss, optionally masked per timestep.
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask  # [B,T,H,max_D]
        eff_mask = action_mask
        if loss_mask is not None:
            action_loss = action_loss * loss_mask[:, :, None, None]
            eff_mask = action_mask * loss_mask[:, :, None, None]
        loss = action_loss.sum() / (eff_mask.sum() + 1e-6)
        return {"action_loss": loss}

    # ── inference ───────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_context: torch.Tensor,  # [B, Tc, S, Dvl]  (history; Tc>=1, last = current)
        state_context: Optional[torch.Tensor],  # [B, Tc, Hs, max_S]
        embodiment_id: torch.Tensor,  # [B]
        image_mask: Optional[torch.Tensor] = None,  # [B, Tc, S]
        backbone_attention_mask: Optional[torch.Tensor] = None,  # [B, Tc, S]
        num_inference_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        """Roll fast weights over the context trajectory, then denoise the current chunk.

        The context forward updates the TTT fast weights on each observation (RoboTTT
        inference: "updates the fast weights on the current observation, and propagates
        them to the next timestep"). The K denoising steps then *apply* the carried fast
        weights (``update_ttt=False``) to denoise the final timestep's action chunk.

        Returns ``[B, action_horizon, max_action_dim]`` (still normalized & padded).
        """
        self.set_frozen_modules_to_eval_mode()
        vl_embs_context = self._process_backbone_traj(vl_embs_context)
        B, Tc, S, _ = vl_embs_context.shape
        device = vl_embs_context.device
        dtype = vl_embs_context.dtype
        K = num_inference_timesteps or self.num_inference_timesteps
        H = self.action_horizon
        max_D = self.action_dim

        # 1) Roll fast weights over the context trajectory using a zero-action placeholder
        #    (the TTT update is driven by the observation tokens, not the action values).
        ctx_actions = torch.zeros(B, Tc, H, max_D, device=device, dtype=dtype)
        ctx_t_disc = torch.zeros(B, Tc, device=device, dtype=torch.long)
        ctx_action_features = self._encode_action_traj(ctx_actions, ctx_t_disc, embodiment_id)
        ctx_state_features = self._encode_state_traj(state_context, embodiment_id) if state_context is not None else None
        if ctx_state_features is None:
            reg = self.register_tokens.unsqueeze(0).unsqueeze(0).expand(B, Tc, -1, -1)
            ctx_sa = torch.cat([reg, ctx_action_features], dim=2)
        else:
            ctx_sa = self._build_per_timestep_tokens(ctx_state_features, ctx_action_features)
        _, ttt_states = self.model(
            hidden_states=ctx_sa,
            encoder_hidden_states=vl_embs_context,
            timestep=ctx_t_disc,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
            ttt_states=None,
            update_ttt=True,
        )  # rolls fast weights over Tc timesteps

        # 2) Denoise the current (last) timestep's action chunk reusing the fast weights.
        cur_state = state_context[:, -1:, :, :] if state_context is not None else None
        cur_image_mask = image_mask[:, -1:, :] if image_mask is not None else None
        cur_bam = backbone_attention_mask[:, -1:, :] if backbone_attention_mask is not None else None
        cur_vl = vl_embs_context[:, -1:, :, :]

        actions = torch.randn(B, H, max_D, device=device, dtype=dtype)  # initial noise
        dt = 1.0 / K
        for k in range(K):
            k_cont = k / float(K)
            k_disc = int(k_cont * self.num_timestep_buckets)
            t_disc = torch.full((B, 1), k_disc, device=device, dtype=torch.long)
            af = self.action_encoder(
                actions, torch.full((B,), k_disc, device=device, dtype=torch.long), embodiment_id
            )  # [B, H, input_emb]
            af = af.unsqueeze(1)  # [B, 1, H, input_emb]
            sf = self._encode_state_traj(cur_state, embodiment_id) if cur_state is not None else None  # [B,1,1,..]
            if sf is None:
                reg = self.register_tokens.unsqueeze(0).expand(B, -1, -1).unsqueeze(1)  # [B,1,R,..]
                sa = torch.cat([reg, af], dim=2)
            else:
                sa = self._build_per_timestep_tokens(sf, af)  # [B,1,L,..]
            out, ttt_states = self.model(
                hidden_states=sa,
                encoder_hidden_states=cur_vl,
                timestep=t_disc,
                image_mask=cur_image_mask,
                backbone_attention_mask=cur_bam,
                ttt_states=ttt_states,
                update_ttt=False,  # reuse fast weights during denoising
            )  # [B,1,L,output_dim]
            out_flat = out.reshape(B, out.shape[2], -1)
            pred = self.action_decoder(out_flat, embodiment_id)  # [B, L, max_D]
            pred_velocity = pred[:, -H:, :]  # [B, H, max_D]
            actions = actions + dt * pred_velocity

        return actions  # [B, H, max_D]


def get_action_model_robottt(config=None):
    """Factory: build :class:`RoboTTTActionHead` from the global framework config."""
    return RoboTTTActionHead(full_config=config)
