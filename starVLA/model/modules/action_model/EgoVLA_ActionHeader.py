# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoVLA action head vendored into starVLA in [2025].
#
# Source: EgoVLA (VILA-based bimanual VLA), `llava/model/ego_vla_decoder`.
# Original decoder: `TransformerSplitActV2` + `TrajDecoder` wrapper.
#
"""EgoVLA trajectory-decoder action head.

This is the EgoVLA-specific action head, vendored natively so EgoVLA becomes a
selectable ``framework.name: EgoVLA`` backbone (see ``VLM4A/EgoVLA.py``) — the
same way GR00T's flow-matching head lives in ``GR00T_ActionHeader.py`` and
openpi's head in ``OpenPI_ActionHead.py``.

Contract (differs from the MLP/L1 head): instead of ``predict_action(hidden)``
this head consumes the EgoVLA decoder inputs directly —

    forward(latent, input_dict, memory, memory_mask) -> {"pred": (B*T, out_dim)}

where
  * ``latent``      : action-query hidden states gathered from the VLM,
                      flat (B*T_query, H).
  * ``memory``      : full multimodal hidden-state sequence from the VLM,
                      (B, L, H) — the transformer cross-attends to it.
  * ``memory_mask`` : (B, L) bool, True = valid token (attention key kept).
  * ``input_dict``  : proprioception dict with camera-frame quantities
                      (``proprio``, ``proprio_3d``, ``proprio_rot``,
                      ``proprio_hand_finger_tip`` …).

Output layout (EgoVLA, ``out_dim = 48``, ``sep_proprio=True``): per timestep the
decoder emits ``[left(3+6+15), right(3+6+15)]`` which the framework re-slices to
wrist-translation(2x3) / MANO-hand(2x15) / wrist-rot6d(2x6).

The module nesting (``EgoVLATrajDecoder.decoder.*``) mirrors the original
``traj_decoder.decoder.*`` so an EgoVLA checkpoint's state-dict keys line up.
"""

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────
#  TransformerSplitActV2 — verbatim EgoVLA decoder (pure torch, no MANO/pt3d)
# ──────────────────────────────────────────────────────────────────────────
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerSplitActV2(nn.Module):
    """EgoVLA split-action transformer decoder.

    Predicts a per-hand action chunk (left / right emitted from even / odd
    action-query positions) while cross-attending to the VLM ``memory``
    sequence and (optionally) prepended proprioception tokens.
    """

    def __init__(
        self,
        hidden_size,
        proprio_size,
        out_dim,
        use_proprio,
        sep_proprio,
        num_transformer_encoder_layers: int = 6,
        **kwargs,
    ):
        super().__init__()

        self.use_proprio = use_proprio
        self.sep_proprio = sep_proprio
        self.proprio_size = proprio_size
        self.out_dim = out_dim

        # Per-hand output width: 3 translation + 6 rot6d + 15 MANO hand pose.
        self._per_hand_dim = 3 + 6 + 15  # = 24 ; two hands => out_dim == 48

        self.proprio_projection = nn.Sequential(
            nn.Linear(self.proprio_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proprio_projection_3d = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proprio_projection_rot = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.proprio_projection_hand = nn.Sequential(
            nn.Linear(5 * 3, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.first_norm = nn.LayerNorm(hidden_size)

        self.output_projection_left = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self._per_hand_dim),
        )
        self.output_projection_right = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, self._per_hand_dim),
        )

        self.layers = nn.ModuleList()
        for _ in range(num_transformer_encoder_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=1,
                batch_first=True,
                activation=torch.nn.functional.elu,
            )
            self.layers.append(encoder_layer)

    def forward(self, latent, input_dict, memory, memory_mask):
        # Plain proprio projection (kept for state-dict / non-sep parity).
        proprio_input = input_dict["proprio"]
        proprio_input = self.proprio_projection(proprio_input)
        proprio_input = proprio_input.unsqueeze(1)  # (B, 1, D)

        # Regroup flat action-query latent (B*T, H) -> (B, T, H).
        latent = latent.reshape(
            proprio_input.shape[0],
            latent.shape[0] // proprio_input.shape[0],
            latent.shape[1],
        )

        if self.use_proprio and self.sep_proprio:
            proprio_input_3d = input_dict["proprio_3d"].reshape(-1, 2, 3)
            proprio_input_3d = self.proprio_projection_3d(proprio_input_3d)

            proprio_input_rot = input_dict["proprio_rot"].reshape(-1, 2, 3)
            proprio_input_rot = self.proprio_projection_rot(proprio_input_rot)

            proprio_input_hand = input_dict["proprio_hand_finger_tip"].reshape(-1, 2, 5 * 3)
            proprio_input_hand = self.proprio_projection_hand(proprio_input_hand)

        if self.use_proprio:
            if self.sep_proprio:
                latent = torch.cat(
                    [proprio_input_3d, proprio_input_rot, proprio_input_hand, latent],
                    dim=1,
                )
            else:
                latent = torch.cat([proprio_input, latent], dim=1)

        _, latent_len, _ = latent.shape

        # Build key-padding mask over [memory ; latent]; True = ignore.
        memory_mask = ~memory_mask
        memory_mask = memory_mask.detach()
        src_key_padding_mask = torch.zeros(latent.shape[0], latent.shape[1]).bool().to(memory_mask.device)
        src_key_padding_mask = torch.concat([memory_mask, src_key_padding_mask], dim=1)

        latent = self.first_norm(latent)
        memory = self.first_norm(memory)

        for layer in self.layers:
            input_latent = torch.concat([memory, latent], dim=1)
            latent = layer(input_latent, src_key_padding_mask=src_key_padding_mask)
            latent = latent[:, -latent_len:]

        # Drop the prepended proprio tokens.
        if self.use_proprio:
            latent = latent[:, 6:, :] if self.sep_proprio else latent[:, 1:, :]

        out_left = self.output_projection_left(latent[:, ::2]).reshape(-1, 1, self._per_hand_dim)
        out_right = self.output_projection_right(latent[:, 1::2]).reshape(-1, 1, self._per_hand_dim)

        output = torch.cat([out_left, out_right], dim=1).reshape(-1, 2 * self._per_hand_dim)
        return {"pred": output}

    def inference(self, latent, input_dict, memory, memory_mask, x=None, return_kl=False):
        return self.forward(latent, input_dict, memory, memory_mask)


# ──────────────────────────────────────────────────────────────────────────
#  TrajDecoder wrapper — mirrors EgoVLA `traj_decoder.decoder.*` nesting
# ──────────────────────────────────────────────────────────────────────────
class EgoVLATrajDecoder(nn.Module):
    """Thin wrapper exposing ``.decoder`` so checkpoint keys match EgoVLA."""

    def __init__(
        self,
        hidden_size: int,
        proprio_size: int,
        out_dim: int,
        use_proprio: bool = True,
        sep_proprio: bool = True,
        decoder_type: str = "transformer_split_action_v2",
    ):
        super().__init__()
        self.decoder_type = decoder_type
        self.hidden_size = hidden_size
        self.out_dim = out_dim
        self.proprio_size = proprio_size
        self.use_proprio = use_proprio
        self.sep_proprio = sep_proprio

        if decoder_type != "transformer_split_action_v2":
            raise NotImplementedError(
                f"EgoVLA_ActionHeader only vendors 'transformer_split_action_v2', got '{decoder_type}'."
            )
        self.decoder = TransformerSplitActV2(
            hidden_size, proprio_size, out_dim, use_proprio, sep_proprio
        )

    def forward(self, latent, input_dict=None, memory=None, memory_mask=None):
        return self.decoder(latent, input_dict, memory=memory, memory_mask=memory_mask)

    def inference(self, latent, input_dict=None, memory=None, memory_mask=None, return_kl=False):
        return self.decoder.inference(
            latent, input_dict, memory=memory, memory_mask=memory_mask, return_kl=return_kl
        )


def get_action_model(config=None):
    """Build the EgoVLA trajectory decoder from a starVLA framework config.

    Reads ``config.framework.action_model`` for:
      - ``action_dim``    -> decoder ``out_dim``  (EgoVLA: 48)
      - ``action_hidden_dim`` -> VLM ``hidden_size`` (EgoVLA: 1536)
      - ``proprio_size``  (default 16)
      - ``use_proprio``   (default True)
      - ``sep_proprio``   (default True)
    """
    action_model_cfg = config.framework.action_model
    model_type = getattr(action_model_cfg, "action_model_type", "EgoVLATrajDecoder")
    if model_type not in ("EgoVLATrajDecoder", "transformer_split_action_v2"):
        raise ValueError(f"EgoVLA_ActionHeader.get_action_model got unexpected type '{model_type}'.")

    return EgoVLATrajDecoder(
        hidden_size=int(action_model_cfg.action_hidden_dim),
        proprio_size=int(getattr(action_model_cfg, "proprio_size", 16)),
        out_dim=int(action_model_cfg.action_dim),
        use_proprio=bool(getattr(action_model_cfg, "use_proprio", True)),
        sep_proprio=bool(getattr(action_model_cfg, "sep_proprio", True)),
        decoder_type=str(getattr(action_model_cfg, "traj_decoder_type", "transformer_split_action_v2")),
    )
