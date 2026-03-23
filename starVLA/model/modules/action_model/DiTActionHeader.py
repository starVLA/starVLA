# Copyright 2025 CogACT. All rights reserved.
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [add global config ].
"""
Diffusion-based action prediction head (DiT variant).

Provides:
  - Size presets (S/B/L) for transformer-based temporal action diffusion backbone
  - ActionModel: wraps diffusion process (training + optional DDIM sampling creation)
"""

import torch
from torch import nn

from starVLA.model.modules.action_model import create_diffusion
from starVLA.model.modules.action_model.DiT_modules.models import DiT

from .DiT_modules import gaussian_diffusion as gd


# Create model sizes of ActionModels
def DiT_S(**kwargs):  # TODO move to config for reproducibility
    """
    Small DiT variant.

    Args:
        **kwargs: Passed through to DiT constructor.

    Returns:
        DiT: Initialized small model.
    """
    return DiT(depth=6, token_size=384, num_heads=4, **kwargs)


def DiT_B(**kwargs):
    """
    Base DiT variant.

    Args:
        **kwargs: Passed through to DiT constructor.

    Returns:
        DiT: Initialized base model.
    """
    return DiT(depth=12, token_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    """
    Large DiT variant.

    Args:
        **kwargs: Passed through to DiT constructor.

    Returns:
        DiT: Initialized large model.
    """
    return DiT(depth=24, token_size=1024, num_heads=16, **kwargs)


# Model size
DiT_models = {"DiT-S": DiT_S, "DiT-B": DiT_B, "DiT-L": DiT_L}


# Create ActionModel
class ActionModel(nn.Module):
    """
    Diffusion temporal action head.

    Components:
        - DiT transformer backbone (token-wise denoiser)
        - Gaussian diffusion scheduler (noise forward/backward)
        - Optional DDIM sampler (created lazily)

    Responsibilities:
        - Forward: add noise + predict denoised residual
        - loss(): simple MSE on noise prediction
        - create_ddim(): build deterministic sampler
    """

    def __init__(
        self,
        action_hidden_dim,
        model_type,
        in_channels,
        future_action_window_size,
        past_action_window_size,
        diffusion_steps=100,
        noise_schedule="squaredcos_cap_v2",
    ):
        """
        Initialize diffusion model and backbone.

        Args:
            action_hidden_dim: Hidden size of conditioning tokens (QFormer output dim).
            model_type: One of {'DiT-S','DiT-B','DiT-L'}.
            in_channels: Action dimensionality (per timestep).
            future_action_window_size: Number of future steps modeled.
            past_action_window_size: Number of past steps possibly encoded (for context).
            diffusion_steps: Total diffusion timesteps.
            noise_schedule: Scheduler type string.
        """
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.token_size = action_hidden_dim  # QFormer output size
        self.net = DiT_models[model_type](
            in_channels=in_channels,
            class_dropout_prob=0.1,
            learn_sigma=learn_sigma,
            future_action_window_size=future_action_window_size,
            past_action_window_size=past_action_window_size,
        )

    def forward(self, gt_action, condition, **kwargs):
        """
        Perform one diffusion training step.

        Args:
            gt_action: Ground truth action tensor [B, T, C].
            condition: Conditioning tokens [B, L, D].
            **kwargs: Ignored (reserved).

        Returns:
            tuple:
                noise_pred: Predicted noise tensor.
                noise: Sampled noise tensor.
                timestep: Timesteps used per batch element.
        """
        # sample random noise and timestep
        noise = torch.randn_like(gt_action)  # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (gt_action.size(0),), device=gt_action.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(gt_action, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, condition)

        assert noise_pred.shape == noise.shape == gt_action.shape

        return noise_pred, noise, timestep

    def loss(self, noise_pred, noise):
        """
        Compute MSE noise prediction loss.

        Args:
            noise_pred: Predicted noise tensor.
            noise: Target noise tensor.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    def create_ddim(self, ddim_step=10):
        """
        Lazily create DDIM sampler instance.

        Args:
            ddim_step: Number of DDIM steps.

        Returns:
            Diffusion: DDIM diffusion object.
        """
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim" + str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False,
        )
        return self.ddim_diffusion


def get_action_model(model_typ="DiT-B", config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        model_typ: (Unused override; model type inferred from config).
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        ActionModel: Initialized diffusion action head.
    """
    action_model_cfg = config.framework.action_model

    model_type = action_model_cfg.action_model_type
    action_hidden_dim = action_model_cfg.action_hidden_dim
    action_dim = action_model_cfg.action_dim
    future_action_window_size = action_model_cfg.future_action_window_size
    past_action_window_size = action_model_cfg.past_action_window_size

    return ActionModel(
        model_type=model_type,  # Model type, e.g., 'DiT-B'
        action_hidden_dim=action_hidden_dim,  # Hidden size of action tokens
        in_channels=action_dim,  # Input channel size
        future_action_window_size=future_action_window_size,  # Future action window size
        past_action_window_size=past_action_window_size,  # Past action window size
    )
