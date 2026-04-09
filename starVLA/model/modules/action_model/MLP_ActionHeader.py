# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Implementations of various action heads, which serve as alternatives to VLM sequential token prediction."""

"this file is adap from https://github.com/moojink/openvla-oft/blob/main/prismatic/models/action_heads.py"

import torch.nn as nn


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""

    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.relu(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x


class L1RegressionActionHead(nn.Module):
    """Simple MLP-based action head that generates continuous actions via L1 regression."""

    def __init__(
        self,
        input_dim=2048,
        hidden_dim=4096,
        action_dim=7,
        NUM_ACTIONS_CHUNK=8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.NUM_ACTIONS_CHUNK = NUM_ACTIONS_CHUNK

        self.model = MLPResNet(num_blocks=2, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=action_dim)

    def predict_action(self, actions_hidden_states):
        """
        actions_hidden_states: (B, chunk_len, hidden_dim)
        Returns: (B, chunk_len, action_dim)
        """
        batch_size, chunk_len, hidden_dim = actions_hidden_states.shape
        x = actions_hidden_states.reshape(batch_size * chunk_len, hidden_dim)
        x = self.model(x)  # (B * chunk_len, action_dim)
        actions = x.view(batch_size, chunk_len, self.action_dim)
        return actions

    def forward(self, actions_hidden_states):
        return self.predict_action(actions_hidden_states)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
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

    action_model = L1RegressionActionHead(
        input_dim=action_hidden_dim,
        hidden_dim=action_hidden_dim * 2,
        action_dim=action_dim,
        NUM_ACTIONS_CHUNK=past_action_window_size + 1 + future_action_window_size,
    )

    return action_model
