import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, encoder_hidden_state, encoder_attention_mask=None):
        """
        Cross-attention block forward.
        Args:
            query (Tensor): Shape [B, Q, D]. Learnable query tokens propagated across layers.
            encoder_hidden_state (Tensor): Shape [B, L, D]. Features from one encoder layer.
            encoder_attention_mask (Tensor | None): Shape [B, L]. 1/True=keep (visible), 0/False=mask. None disables masking.
        Returns:
            Tensor: Updated query tokens of shape [B, Q, D].
        Details:
            1. LayerNorm + MultiheadAttention (Q = query, K/V = encoder_hidden_state).
            2. Residual path: query = query + attn_output, then add MLP residual.
            3. Dropout is applied only on the MLP output.
        """
        q = self.norm1(query)
        kv = encoder_hidden_state

        if encoder_attention_mask is not None:
            attn_mask = encoder_attention_mask.unsqueeze(1).to(dtype=torch.bool)  # [B, 1, L]
        else:
            attn_mask = None

        attn_output, _ = self.cross_attn(q, kv, kv, key_padding_mask=attn_mask)
        query = query + attn_output
        query = query + self.dropout(self.mlp(self.norm2(query)))
        return query


class LayerwiseQFormer(nn.Module):
    def __init__(
        self, input_hidden_dim=2048, output_hidden_dim=768, num_query_tokens=64, num_layers=37, num_heads=8, config=None
    ):
        super().__init__()
        self.input_hidden_dim = input_hidden_dim
        self.output_hidden_dim = output_hidden_dim
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.config = config
        # Project input to output dimension
        self.proj = nn.Linear(input_hidden_dim, output_hidden_dim)
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, output_hidden_dim))

        # Independent cross-attention blocks (one per encoder layer)
        self.layers = nn.ModuleList([CrossAttentionBlock(output_hidden_dim, num_heads) for _ in range(num_layers)])

    def forward(self, hidden_states_list, encoder_attention_mask=None):
        """
        Layer-wise Q-Former forward pass.
        Args:
            hidden_states_list (List[Tensor]): Length == num_layers. Each tensor is [B, L, Din], raw encoder layer outputs (before projection).
            encoder_attention_mask (Tensor | None): Shape [B, L]. Same semantics as in CrossAttentionBlock.
        Returns:
            Tensor: Aggregated query tokens of shape [B, Q, Dout].
        Pipeline:
            1. Stack per-layer features to [B, N, L, Din] and linearly project to Dout.
            2. Expand global learnable query tokens to batch: [B, Q, Dout].
            3. Apply cross-attention layer-by-layer: each query attends only to the corresponding encoder layer features.
        Notes:
            - Asserts len(hidden_states_list) == num_layers.
            - Does not modify gradient flow of hidden_states_list.
        """
        # hidden_states_list = self.scale_hook(hidden_states_list)

        assert (
            len(hidden_states_list) == self.num_layers
        ), f"Expected {self.num_layers} layers, got {len(hidden_states_list)}"

        B = hidden_states_list[0].size(0)
        # Project input hidden states to output dimension
        #    Result shape [B, N, L, Din]
        hs = torch.stack(hidden_states_list, dim=1)
        #    proj_hs shape [B, N, L, Dout]
        proj_hs = self.proj(hs)
        # 3) Unbind back to list, each element restored to [B, L, Dout]
        hidden_states_list = list(proj_hs.unbind(dim=1))

        # Expand query tokens for each batch
        query = self.query_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

        # Iterate through each layer and apply cross-attention
        for i, layer in enumerate(self.layers):
            query = layer(query, hidden_states_list[i], encoder_attention_mask)

        return query

    def scale_hook(self, hidden_states_list, scale_factor=0.1):
        """
        (Experimental / optional) Register gradient scaling hooks on each layer's hidden states.
        Args:
            hidden_states_list (List[Tensor]): Per-layer feature tensors.
            scale_factor (float): Gradient scaling factor (effective only if enabled via config and != 1).
        Returns:
            List[Tensor]: Original list (no data copy); hooks may be attached in-place.
        Design:
            - Currently returns immediately (guard condition hard-coded False) as a placeholder.
            - Uses attribute _scaled_hook to avoid duplicate hook registration in distributed settings.
            - Can be enabled later for gradient dampening or perturbation experiments.
        Performance:
            - Excessive hook registrations can hurt speed; kept lazy by default.
        """
        # --- 1. Register gradient scaling hooks on input hidden_states_list ---
        if (
            self.config
            and hasattr(self.config.vla, "layer_qformer")
            and hasattr(self.config.vla.layer_qformer, "grad_scale")
            and self.config.vla.layer_qformer.grad_scale != 1
        ):
            scale_factor = self.config.vla.layer_qformer.grad_scale
        else:
            return hidden_states_list  # If grad_scale is not configured, return the original list

        scaled_hidden_states_list = []
        for hidden_states in hidden_states_list:
            if hidden_states.requires_grad:
                # Ensure gradient scaling is executed only once in distributed settings
                if not hasattr(hidden_states, "_scaled_hook"):  # Prevent duplicate registration --> Seems to accelerate
                    hook = lambda grad: grad * scale_factor
                    hidden_states.register_hook(hook)
                    hidden_states._scaled_hook = True  # Mark as processed
            scaled_hidden_states_list.append(hidden_states)

        return hidden_states_list


import torch.nn as nn


def get_layerwise_qformer(num_heads=8, config=None, **kwargs):
    """
    Build a LayerwiseQFormer instance.
    Args:
        num_heads (int): Number of attention heads for CrossAttentionBlock.
        config: Configuration object; must contain config.framework.layer_qformer with:
            - qformer_start_layer / qformer_end_layer: range of layers (start inclusive, end exclusive).
            - num_query_tokens: Number of learnable query tokens.
            - input_dim: Input feature dimension (Din).
            - ouptput_dim: Output feature dimension (Dout).
        **kwargs: Reserved for future extensions (unused).
    Returns:
        LayerwiseQFormer: Instantiated model.
    Notes:
        - num_layers = end_layer - start_layer (half-open interval).
        - Does not perform weight loading or device moves here.
    """
    # dist.barrier()
    qformer_cfg = config.framework.layer_qformer
    num_layers = qformer_cfg.qformer_end_layer - qformer_cfg.qformer_start_layer if config else num_layers
    num_query_tokens = qformer_cfg.num_query_tokens
    input_hidden_dim = config.framework.layer_qformer.input_dim
    output_hidden_dim = config.framework.layer_qformer.ouptput_dim
    num_query_tokens = qformer_cfg.num_query_tokens

    qformer = LayerwiseQFormer(
        input_hidden_dim=input_hidden_dim,
        output_hidden_dim=output_hidden_dim,
        num_query_tokens=num_query_tokens,
        num_layers=num_layers,
        num_heads=num_heads,
        config=config,
    )
    return qformer
