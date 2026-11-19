# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
#
# Test-Time-Training (TTT) layer, ported from RoboTTT (arxiv 2607.15275, "RoboTTT:
# Context Scaling for Robot Policies", Jiang et al.) and the original TTT formulation
# (Sun et al., "Learning to (Learn at Test Time)").
#
# A TTT layer compresses a token sequence into **fast weights** that are updated by
# gradient descent *during both training and inference*, replacing the KV-cache memory
# of attention. Concretely (RoboTTT §2, Eq. 1–2):
#
#   K_t = θ_K(x_t),  V_t = θ_V(x_t),  Q_t = θ_Q(x_t)              (projections)
#   ℓ(W; K_t, V_t) = ||f(W; K_t) − V_t||²                          (self-supervised MSE)
#   W_t = W_{t-1} − η ∇_W ℓ(W_{t-1}; K_t, V_t)                     (Eq. 1, fast-weight update)
#   out_t = f(W_t; Q_t)                                             (Eq. 2, apply)
#
# where f is a two-layer MLP with GeLU (RoboTTT Appendix A.1), η = base_lr (0.1) ·
# softplus(θ_lr) is a *learned* inner learning rate, and the fast-weight init W0 is
# meta-learned through the outer task loss. A learned gate g (init ≈ 0.001) scales the
# TTT contribution before the residual add (RoboTTT Eq. 3), preserving pretrained
# capabilities at the start of training.
#
# For tractability over long contexts, we use the standard **mini-batch / chunked** TTT:
# the sequence is split into chunks of length L; within a chunk, the inner gradient is
# accumulated over the L steps at the chunk-start W, one gradient step is taken → W',
# and outputs for all L steps are computed with W' (parallel). W' is carried to the next
# chunk. TBPTT truncates the graph at segment boundaries (fast-weight *values* carry,
# gradients detach) so memory depends on segment length, not total sequence length.
#
# The inner gradient is computed via torch.autograd.grad with create_graph=self.training,
# so during training the outer task loss backpropagates through the inner updates into
# θ_K/θ_V/θ_Q/θ_lr/W0 (meta-learning). At inference (no-grad), the inner gradient is
# computed under a local torch.enable_grad() and detached.

from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class TTTLayer(nn.Module):
    """Multi-head TTT-MLP layer (RoboTTT / Sun et al.).

    Input  ``x``: ``[B, T, C]`` (B = batch, T = sequence / time length, C = channels).
    Output ``y``: ``[B, T, C]`` with ``y = x + gate · out_proj(TTT_out)``.
    Returns ``(y, new_state)`` where ``new_state`` is the carried fast-weight state
    ``(W1, W2)`` for streaming / TBPTT across calls.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_inner_dim: int,
        base_lr: float = 0.1,
        gate_init: float = 0.001,
        rope_theta: float = 10000.0,
        chunk_size: int = 8,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mlp_inner_dim = mlp_inner_dim
        self.base_lr = base_lr
        self.chunk_size = chunk_size
        H, d, n = num_heads, self.head_dim, mlp_inner_dim

        # Projections θ_K, θ_V, θ_Q (per-head 1×1, no cross-head mixing — standard TTT).
        self.theta_K = nn.Parameter(0.02 * torch.randn(H, d, d))
        self.theta_V = nn.Parameter(0.02 * torch.randn(H, d, d))
        self.theta_Q = nn.Parameter(0.02 * torch.randn(H, d, d))

        # Learned fast-weight init W0 (per-head, bias-free 2-layer MLP). Meta-learned via
        # the outer task loss through the inner gradient steps.
        self.W1_0 = nn.Parameter(0.02 * torch.randn(H, d, n))
        self.W2_0 = nn.Parameter(0.02 * torch.randn(H, n, d))

        # Learned inner learning rate: η = base_lr · softplus(θ_lr).
        self.theta_lr = nn.Parameter(torch.zeros(num_heads))  # softplus(0) ≈ 0.693 → small η

        # Output projection + residual gate (Eq. 3).
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # RoPE across the time dimension (RoboTTT Appendix A.1: "We use RoPE").
        self.rope_theta = rope_theta
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    # ── helpers ──────────────────────────────────────────────────────────
    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply 1D RoPE across the time dim. x: [B, H, T, d]."""
        B, H, T, d = x.shape
        half = d // 2
        pos = torch.arange(T, device=x.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.einsum("t,f->tf", pos, self.rope_inv_freq)  # [T, half]
        cos = freqs.cos()[None, None, :, :]  # [1,1,T,half]
        sin = freqs.sin()[None, None, :, :]
        x1 = x[..., :half]
        x2 = x[..., half:]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.cat([rot1, rot2], dim=-1)

    @staticmethod
    def _fast(W1: torch.Tensor, W2: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Functional fast model f(W; x) = W2 · GeLU(W1 x). No bias.
        x: [B,H,L,d], W1: [B,H,d,n], W2: [B,H,n,d] → [B,H,L,d]."""
        h = F.gelu(torch.einsum("bhtd,bhdn->bhtn", x, W1))  # [B,H,L,n]
        return torch.einsum("bhtn,bhnd->bhtd", h, W2)  # [B,H,L,d]

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fresh fast-weight state = W0 broadcast to per-sample tensors."""
        W1 = self.W1_0.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B,H,d,n]
        W2 = self.W2_0.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B,H,n,d]
        return W1, W2

    # ── forward ───────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        tbptt_segment_length: Optional[int] = None,
        update: bool = True,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, T, C].
            state: optional carried fast weights (W1, W2) from a previous call/segment.
                ``None`` → initialise from the learned W0.
            tbptt_segment_length: if set, detach the fast weights every this many
                timesteps so the autograd graph truncates (TBPTT). Fast-weight values
                still carry across the boundary. Only meaningful in training mode.
            update: if False, skip the fast-weight gradient step and only *apply* the
                current W (``out = f(W, Q)``). Used at inference so the K denoising steps
                reuse the fast weights rolled from context rather than updating K times.

        Returns:
            (y [B, T, C], new_state).
        """
        # Build the meta-graph (outer task loss backprops through the inner fast-weight
        # update into θ_K/θ_V/θ_Q/θ_lr/W0) only when the *outer* autograd context is
        # grad-enabled AND the module is in training mode. ``self.training`` alone is NOT
        # a reliable signal at inference: ``predict_action`` runs under ``no_grad`` but
        # ``set_frozen_modules_to_eval_mode`` leaves the trainable DiT (and thus this
        # layer) in ``train()``, so ``self.training`` would be True and we'd take the
        # ``create_graph=True`` / ``retain_graph=True`` branch under ``no_grad`` — which
        # is ill-formed and raises "differentiated Tensors ... not ... used in the graph".
        # ``torch.is_grad_enabled()`` at entry reflects whether the caller will backprop.
        outer_grad_enabled = torch.is_grad_enabled()
        training = outer_grad_enabled and self.training
        B, T, C = x.shape
        H, d = self.num_heads, self.head_dim
        x_h = x.reshape(B, H, T, d)
        x_h = self._rope(x_h)
        K = torch.einsum("bhtd,hde->bhte", x_h, self.theta_K)
        V = torch.einsum("bhtd,hde->bhte", x_h, self.theta_V)
        Q = torch.einsum("bhtd,hde->bhte", x_h, self.theta_Q)

        if state is None:
            W1, W2 = self.init_state(B)
        else:
            W1, W2 = state

        eta = self.base_lr * F.softplus(self.theta_lr)  # [H] → broadcast later

        # ── Apply-only path (inference denoising): reuse current fast weights, no update.
        if not update:
            out = self._fast(W1, W2, Q)  # [B, H, T, d]
            out = out.reshape(B, T, C)
            out = self.out_proj(out)
            y = x + self.gate * out
            return y, (W1, W2)

        # Inner gradient step needs a graph. During a training forward the outer graph is
        # already enabled; during inference (outer no_grad) we re-enable grad locally and
        # detach outputs so the inner update runs but nothing leaks into the outer graph.
        grad_ctx = torch.enable_grad() if not outer_grad_enabled else nullcontext()

        chunk = self.chunk_size
        outs = []
        t0 = 0
        seg_done = 0
        with grad_ctx:
            while t0 < T:
                t1 = min(t0 + chunk, T)
                Kc = K[:, :, t0:t1]
                Vc = V[:, :, t0:t1]
                Qc = Q[:, :, t0:t1]

                if training:
                    w1, w2 = W1, W2  # carry the graph (views of W0 / carried tensors)
                else:
                    w1 = W1.detach().requires_grad_(True)
                    w2 = W2.detach().requires_grad_(True)

                pred = self._fast(w1, w2, Kc)
                loss = ((pred - Vc) ** 2).mean(dim=[-1, -2])  # [B, H]
                g1, g2 = torch.autograd.grad(
                    loss.sum(),
                    [w1, w2],
                    create_graph=training,
                    retain_graph=training,
                )
                w1n = w1 - eta.view(1, -1, 1, 1) * g1
                w2n = w2 - eta.view(1, -1, 1, 1) * g2
                out_c = self._fast(w1n, w2n, Qc)  # [B, H, Lc, d]

                if not training:
                    w1n = w1n.detach()
                    w2n = w2n.detach()
                    out_c = out_c.detach()

                outs.append(out_c)
                W1, W2 = w1n, w2n
                seg_done += t1 - t0
                # TBPTT: truncate graph at segment boundaries (values carry, grads detach).
                if training and tbptt_segment_length and seg_done >= tbptt_segment_length:
                    W1 = W1.detach().requires_grad_(True)
                    W2 = W2.detach().requires_grad_(True)
                    seg_done = 0
                t0 = t1

        out = torch.cat(outs, dim=2)  # [B, H, T, d]
        out = out.reshape(B, T, C)
        out = self.out_proj(out)
        y = x + self.gate * out
        return y, (W1, W2)
