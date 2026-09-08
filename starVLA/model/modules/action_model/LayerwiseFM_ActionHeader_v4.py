"""QwenPI_v4 action head backed by the fused self/cross-attention DiT."""

import torch
from torch import nn

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    ActionEncoder,
    DiTConfig,
    LayerwiseFlowmatchingActionHead,
    MLP,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit_v4 import (
    QwenPIv4DiT,
)


class LayerwiseFlowmatchingActionHeadV4(LayerwiseFlowmatchingActionHead):
    """Layer-wise flow-matching head with QwenPI_v4's fused DiT.

    The sampling and loss code is shared with the existing layer-wise head,
    while construction is intentionally separate so QwenPI_v4 cannot fall
    back to the legacy DiT or its forward-mode switches.
    """

    def __init__(self, global_config, **kwargs):
        # This class intentionally initializes nn.Module directly.  The parent
        # implementation hard-codes the legacy DiT; the rest of its methods
        # are architecture-agnostic and are reused below.
        nn.Module.__init__(self)

        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg
        for key, value in DiTConfig.items():
            if diffusion_model_cfg.get(key, None) is None:
                diffusion_model_cfg[key] = value

        # These are framework hints for old DiT variants, not v4 controls.
        # Drop them if a copied/legacy YAML happens to carry them so v4 always
        # has one fixed attention topology.
        ignored_legacy_keys = {
            "action_dit_hidden_dim",
            "interleave_self_attention",
            "use_canonical_forward",
        }
        diffusion_model_cfg_kwargs = {
            key: value
            for key, value in diffusion_model_cfg.items()
            if key not in ignored_legacy_keys
        }

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = QwenPIv4DiT(**diffusion_model_cfg_kwargs)
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                output_dim=self.input_embedding_dim,
            )
            if action_config.state_dim
            else None
        )
        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(
            action_config.num_target_vision_tokens,
            self.input_embedding_dim,
        )
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                action_config.max_seq_len,
                self.input_embedding_dim,
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        from torch.distributions import Beta

        self.beta_dist = Beta(
            action_config.noise_beta_alpha,
            action_config.noise_beta_beta,
        )
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def _normalize_encoder_states(self, vl_embs):
        """Accept QwenPI layer-wise states and GR00T single-layer states.

        QwenPI supplies one encoder hidden-state tensor per DiT block.  The
        GR00T action head supplies one tensor (the final VLM layer), which is
        reused by every DiT block.  Keep the broadcast as a list of references
        so the DiT sees the same fixed per-block interface without copying the
        encoder activations.
        """
        if isinstance(vl_embs, torch.Tensor):
            return [vl_embs] * len(self.model.transformer_blocks)
        if isinstance(vl_embs, (list, tuple)):
            return list(vl_embs)
        raise TypeError(
            "QwenPI_v4 expects VLM encoder states as a tensor or a list/tuple "
            f"of tensors, got {type(vl_embs).__name__}."
        )

    def forward(
        self,
        vl_embs_list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ):
        return super().forward(
            self._normalize_encoder_states(vl_embs_list),
            actions,
            state,
            encoder_attention_mask=encoder_attention_mask,
        )

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        return super().predict_action(
            self._normalize_encoder_states(vl_embs_list),
            state,
            encoder_attention_mask=encoder_attention_mask,
        )

    def predict_action_realtime(
        self,
        vl_embs_list,
        state: torch.Tensor = None,
        prev_action_chunk: torch.Tensor = None,
        inference_delay: int = 1,
        mode: str = "pigdm",
        suffix_length: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        return super().predict_action_realtime(
            self._normalize_encoder_states(vl_embs_list),
            state,
            prev_action_chunk,
            inference_delay,
            mode,
            suffix_length,
            prefix_attention_schedule,
            max_guidance_weight,
            encoder_attention_mask=encoder_attention_mask,
        )


def get_action_model_v4(config=None):
    """Build the fixed-topology QwenPI_v4 action head."""

    return LayerwiseFlowmatchingActionHeadV4(global_config=config)
