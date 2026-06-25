"""
Layerwise Discrete Diffusion Action Head (MaskGIT-style).
QwenPI-style: layer-wise cross-attention with vl_embs_list, same arch size as QwenPI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.discrete_diffusion import (
    IGNORE_TOKEN,
    ActionBinning,
    decode_mask_schedule,
    mask_by_deterministic_lowest,
    mask_by_random_topk,
    train_mask_schedule,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


def _bin_range(config, action_norm_stats):
    if action_norm_stats and ("min" in action_norm_stats or "q01" in action_norm_stats):
        return (-1.0, 1.0)
    return config.get("action_low", -1.0), config.get("action_high", 1.0)


def _to_embed_ids(tokens, mask_token_id, ignore_token=IGNORE_TOKEN):
    return torch.where(tokens == ignore_token, torch.full_like(tokens, mask_token_id), tokens)


class LayerwiseDiscreteDiffusionActionHead(nn.Module):
    """
    MaskGIT-style discrete diffusion with QwenPI layer-wise cross-attention.
    Same DiT architecture size as LayerwiseFlowmatchingActionHead (QwenPI).
    """

    def __init__(self, global_config, action_norm_stats=None):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # Match QwenPI: override DiTConfig from qwenvl
        DiTConfig["num_layers"] = global_config.framework.qwenvl.num_vl_layers
        DiTConfig["input_embedding_dim"] = global_config.framework.qwenvl.vl_hidden_dim
        DiTConfig["num_attention_heads"] = DiTConfig["input_embedding_dim"] // DiTConfig["attention_head_dim"]
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.seq_len = self.action_horizon * self.action_dim
        self.num_inference_steps = action_config.get("num_inference_steps", 8)
        self.num_bins = action_config.get("num_bins", 256)
        self.mask_token_id = self.num_bins
        self.train_mask_schedule = action_config.get("train_mask_schedule", "cosine")
        self.no_mask_token_prob = action_config.get("no_mask_token_prob", 0.0)
        self.decode_schedule = action_config.get("decode_schedule", "cosine")
        self.l1_loss_weight = action_config.get("l1_loss_weight", 0.1)
        self.config = action_config

        low, high = _bin_range(action_config, action_norm_stats)
        representation = action_config.get("representation", "bin")
        self.binning = ActionBinning(
            num_bins=self.num_bins,
            action_dim=self.action_dim,
            low=low,
            high=high,
            representation=representation,
        )

        diffusion_model_cfg.update(DiTConfig)
        diffusion_model_cfg.cross_attention_dim = DiTConfig["input_embedding_dim"]
        diffusion_model_cfg.output_dim = self.binning.logits_dim

        self.input_embedding_dim = global_config.framework.qwenvl.vl_hidden_dim
        self.model = DiT(**diffusion_model_cfg)

        self.inner_dim = DiTConfig["num_attention_heads"] * DiTConfig["attention_head_dim"]
        self.token_embedding = nn.Embedding(self.num_bins + 1, self.inner_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        if action_config.get("state_dim", 0) > 0:
            self.state_encoder = MLP(action_config.state_dim, self.inner_dim, self.inner_dim)
        else:
            self.state_encoder = None

        self.future_tokens = nn.Embedding(action_config.get("num_target_vision_tokens", 32), self.inner_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.get("add_pos_embed", True):
            self.position_embedding = nn.Embedding(action_config.get("max_seq_len", 1024), self.inner_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        else:
            self.position_embedding = None

    def apply_mask(self, input_tokens, generator=None):
        B, T, D = input_tokens.shape
        device = input_tokens.device
        L = T * D

        total_unknown = torch.full((B,), L, dtype=torch.float, device=device)
        mask_ratios = train_mask_schedule(torch.rand(B, device=device, generator=generator), self.train_mask_schedule)
        num_mask = (total_unknown * mask_ratios).round().long().clamp(1, L)
        num_mask = torch.where(total_unknown > 0, num_mask, torch.zeros_like(num_mask))

        vals = torch.rand(B, T, D, device=device, generator=generator)
        perm = torch.argsort(vals.reshape(B, L), dim=1)
        ranks = torch.argsort(perm, dim=1)
        masked_mask = (ranks < num_mask.unsqueeze(1)).reshape(B, T, D)

        if self.no_mask_token_prob > 0:
            prob = torch.rand(B, T, D, device=device, generator=generator)
            unmask = (prob < self.no_mask_token_prob) & masked_mask
            masked_mask = masked_mask & ~unmask

        return torch.where(masked_mask, IGNORE_TOKEN, input_tokens)

    def _forward_logits(self, vl_embs_list, token_ids, state_features, device):
        B = token_ids.shape[0]
        x = _to_embed_ids(token_ids.reshape(B, -1), self.mask_token_id)
        action_emb = self.token_embedding(x)

        if self.position_embedding is not None:
            pos_len = action_emb.shape[1]
            pos_ids = torch.arange(pos_len, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_emb = action_emb + pos_embs

        future = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        if state_features is not None:
            sa_embs = torch.cat([state_features, future, action_emb], dim=1)
        else:
            sa_embs = torch.cat([future, action_emb], dim=1)

        logits = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,
            timestep=torch.zeros(B, device=device, dtype=torch.long),
        )

        n_prefix = (1 + future.shape[1]) if state_features is not None else future.shape[1]
        action_logits = logits[:, n_prefix:, :]
        return action_logits.reshape(B, self.action_horizon, self.action_dim, self.binning.logits_dim)

    def forward(self, vl_embs_list, actions, state=None, logging: bool = False):
        _, T, D = actions.shape
        assert T == self.action_horizon and D == self.action_dim

        target_bins = self.binning.encode(actions)
        input_tokens = self.apply_mask(target_bins)
        loss_mask = input_tokens == IGNORE_TOKEN
        x = _to_embed_ids(input_tokens, self.mask_token_id)

        state_feat = None
        if state is not None and self.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_feat = self.state_encoder(state).unsqueeze(1)

        logits = self._forward_logits(vl_embs_list, x, state_feat, actions.device)
        if logging:
            print(f"target_bins shape: {target_bins.shape}")
            print(f"target_bins first batch first action: {target_bins[0, 0, :]}")
        return logits, target_bins, {"loss_mask": loss_mask, "actions": actions}

    def loss(self, pred, target, loss_mask=None, actions=None, **kwargs):
        B = pred.shape[0]
        device = pred.device

        if loss_mask is None:
            loss_mask = torch.ones(B, *target.shape[1:], dtype=torch.bool, device=device)

        logits_dim = self.binning.logits_dim
        pred_flat = pred.reshape(B, self.seq_len, logits_dim)

        if self.binning.representation == "bin":
            ce = F.cross_entropy(
                pred_flat.reshape(-1, logits_dim),
                target.reshape(-1),
                reduction="none",
            ).reshape(B, *target.shape[1:])
        else:
            bit_targets = self.binning.indices_to_bit_targets(target)
            bce = F.binary_cross_entropy_with_logits(pred_flat, bit_targets, reduction="none").mean(dim=-1)
            ce = bce

        ce_masked = torch.where(loss_mask, ce, torch.zeros_like(ce))
        num_masked = loss_mask.float().sum(dim=(1, 2)) + 1e-8
        ce_loss = (ce_masked.sum(dim=(1, 2)) / num_masked).mean()

        l1_loss = torch.tensor(0.0, device=device)
        if self.l1_loss_weight > 0 and actions is not None:
            pred_cont = self.binning.decode_logits(pred_flat)
            l1_loss = (pred_cont - actions).abs().mean()

        return ce_loss + self.l1_loss_weight * l1_loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list,
        state=None,
        choice_temperature=0.1,
        decode_temperature=1.0,
        use_simple_max=False,
    ):
        """
        use_simple_max: if True, single forward pass with all-mask input,
            argmax per position, decode. No iterative MaskGIT decode.
        """
        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        L = self.seq_len

        cur_seqs = torch.full(
            (B, self.action_horizon, self.action_dim),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        state_feat = None
        if state is not None and self.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_feat = self.state_encoder(state).unsqueeze(1)

        if use_simple_max:
            logits = self._forward_logits(vl_embs_list, cur_seqs, state_feat, device)
            return self.binning.decode_logits(logits)

        deterministic_decode = decode_temperature == 0
        deterministic_choice = choice_temperature == 0
        unknown_init = torch.full((B,), L, dtype=torch.long, device=device)

        for step_idx in range(self.num_inference_steps):
            logits = self._forward_logits(vl_embs_list, cur_seqs, state_feat, device)
            safe_temp = max(decode_temperature, 1e-8)
            sampled, selected_probs = self.binning.sample_indices_from_logits(
                logits,
                temperature=safe_temp,
                deterministic=deterministic_decode,
            )

            unknown_map = cur_seqs == self.mask_token_id
            sampled = torch.where(unknown_map, sampled, cur_seqs)

            ratio = (step_idx + 1.0) / self.num_inference_steps
            mask_ratio = decode_mask_schedule(torch.tensor(ratio, device=device), self.decode_schedule)
            mask_len = (unknown_init.float() * mask_ratio).long()
            min_len = torch.full_like(mask_len, 1)
            max_len = (unknown_init - 1).clamp(min=0)
            mask_len = mask_len.clamp(min=min_len, max=max_len)
            if step_idx == self.num_inference_steps - 1:
                mask_len = torch.zeros_like(mask_len)

            # Once unmasked, positions stay fixed (no remask).
            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                torch.full_like(selected_probs, float("inf")),
            )

            selected_flat = selected_probs.reshape(B, L)
            if deterministic_choice:
                action_mask_flat = mask_by_deterministic_lowest(selected_flat, mask_len)
            else:
                temp = choice_temperature * (1.0 - ratio)
                action_mask_flat = mask_by_random_topk(selected_flat, mask_len, temperature=temp)
            action_mask = action_mask_flat.reshape(B, self.action_horizon, self.action_dim)
            cur_seqs = torch.where(action_mask, self.mask_token_id, sampled)

        return self.binning.decode(cur_seqs)

    @torch.no_grad()
    def predict_action_realtime(
        self,
        vl_embs_list,
        state=None,
        prev_action_chunk: torch.Tensor | None = None,
        inference_delay: int = 1,
        execution_horizon: int | None = None,
        choice_temperature: float = 0.1,
        decode_temperature: float = 1.0,
        fixed_steps: bool = False,
        hard_mask: bool = False,
        early_stop: bool = False,
    ) -> torch.Tensor:
        """
        RTC-aware MaskGIT decode.

        The previous chunk's un-executed actions become the known prefix:
            prefix_length = action_horizon - execution_horizon
        Only the last `execution_horizon` positions need to be generated.

        Schedule over action horizon H with execution_horizon s:
            positions  0 .. H-s-1   : known prefix (from prev chunk)
            positions  H-s .. H-1   : masked, to be generated

        Args:
            prev_action_chunk: (B, T, action_dim) continuous actions from
                the previous prediction.
            inference_delay: number of actions executed between predictions.
                Used as fallback for execution_horizon if not provided.
            execution_horizon: number of new positions to generate (s).
                Defaults to inference_delay.
            fixed_steps: if True, always use self.num_inference_steps;
                if False (default), scale steps proportionally to the
                fraction of tokens that need to be generated.
        """
        if prev_action_chunk is None or inference_delay <= 0:
            return self.predict_action(
                vl_embs_list,
                state,
                choice_temperature=choice_temperature,
                decode_temperature=decode_temperature,
            )

        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        L = self.seq_len
        deterministic_decode = decode_temperature == 0
        deterministic_choice = choice_temperature == 0

        if execution_horizon is None:
            execution_horizon = inference_delay
        execution_horizon = min(execution_horizon, self.action_horizon)

        if hard_mask:
            prefix_length = inference_delay
        else:
            prefix_length = self.action_horizon - execution_horizon

        # Pad prev_action_chunk to action_horizon if shorter (right-pad with zeros;
        # padded positions fall under prefix_mask=False and get replaced by mask_token).
        T_prev = prev_action_chunk.shape[1]
        if T_prev < self.action_horizon:
            pad = torch.zeros(
                B,
                self.action_horizon - T_prev,
                prev_action_chunk.shape[2],
                device=device,
                dtype=prev_action_chunk.dtype,
            )
            prev_action_chunk = torch.cat([prev_action_chunk, pad], dim=1)

        # Encode the prefix into bin indices
        prefix_bins = self.binning.encode(prev_action_chunk)
        prefix_mask = (torch.arange(self.action_horizon, device=device)[None, :, None] < prefix_length).expand(
            B, self.action_horizon, self.action_dim
        )

        cur_seqs = torch.where(
            prefix_mask,
            prefix_bins,
            torch.full_like(prefix_bins, self.mask_token_id),
        )

        # Count actual masked tokens from cur_seqs
        num_unknown_tokens = (cur_seqs == self.mask_token_id).sum(dim=(1, 2))  # (B,)
        unknown_init = num_unknown_tokens
        if fixed_steps:
            num_steps = self.num_inference_steps
        else:
            # Use the max across the batch to determine step count
            num_steps = max(1, int(self.num_inference_steps * num_unknown_tokens.max().item() / L))

        state_feat = None
        if state is not None and self.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_feat = self.state_encoder(state).unsqueeze(1)

        for step_idx in range(num_steps):
            logits = self._forward_logits(vl_embs_list, cur_seqs, state_feat, device)
            safe_temp = max(decode_temperature, 1e-8)
            sampled, selected_probs = self.binning.sample_indices_from_logits(
                logits,
                temperature=safe_temp,
                deterministic=deterministic_decode,
            )

            unknown_map = cur_seqs == self.mask_token_id
            # Force prefix positions to stay as encoded prefix
            sampled = torch.where(prefix_mask, prefix_bins, sampled)
            sampled = torch.where(unknown_map, sampled, cur_seqs)

            ratio = (step_idx + 1.0) / num_steps
            mask_ratio = decode_mask_schedule(torch.tensor(ratio, device=device), self.decode_schedule)
            mask_len = (unknown_init.float() * mask_ratio).long()
            min_len = torch.full_like(mask_len, 1)
            max_len = (unknown_init - 1).clamp(min=0)
            mask_len = mask_len.clamp(min=min_len, max=max_len)
            if step_idx == num_steps - 1:
                mask_len = torch.zeros_like(mask_len)

            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                torch.full_like(selected_probs, float("inf")),
            )

            selected_flat = selected_probs.reshape(B, L)
            if deterministic_choice:
                action_mask_flat = mask_by_deterministic_lowest(selected_flat, mask_len)
            else:
                temp = choice_temperature * (1.0 - ratio)
                action_mask_flat = mask_by_random_topk(selected_flat, mask_len, temperature=temp)
            action_mask = action_mask_flat.reshape(B, self.action_horizon, self.action_dim)
            cur_seqs = torch.where(
                prefix_mask,
                prefix_bins,
                torch.where(action_mask, self.mask_token_id, sampled),
            )

            # Early stop if all non-prefix positions are unmasked
            if early_stop:
                if (cur_seqs[:, prefix_length:, :] != self.mask_token_id).all():
                    break

        return self.binning.decode(cur_seqs)


def get_action_model(config=None, action_norm_stats=None):
    return LayerwiseDiscreteDiffusionActionHead(global_config=config, action_norm_stats=action_norm_stats)
