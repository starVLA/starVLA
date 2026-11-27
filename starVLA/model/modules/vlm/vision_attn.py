"""Batched attention for the Qwen vision towers when every image has the same number of patches.

The vision attention of Qwen2-VL, Qwen2.5-VL and Qwen3-VL (the same code in transformers 4.57), and of Qwen3.5
(transformers 5), receives all patches
of all images as one sequence plus the chunk boundaries (cu_seqlens). With any implementation but flash-attention it
reads the chunk lengths back to the host in every layer (`lengths.tolist()`: a GPU sync per layer) and then runs one
attention call per chunk: 24 layers x 32 images per training step at batch 16 x 2 views.

When all chunks have the same length (the usual case in training, where every view is resized to one resolution) the
chunks are a batch: one attention call per layer on [chunks, heads, length, dim], and no host read in the layers.
The layout is derived once per vision forward from grid_thw on the host. Qwen2.5-VL alternates between full
attention (one chunk per frame) and window attention (one chunk per window); each kind is batched when it is
uniform. Any other case falls back to the upstream code.

Batching is the same arithmetic, but a GPU library may pick a different matrix-product kernel for a different batch
size, and then the last bit can differ (on B300 with bf16: 32 chunks of 256 tokens are bitwise identical to 32
single calls at head dimension 64, not at 72 or 80; chunks of 64 tokens are identical at all three). So before a
layout is batched, `vision_attn` runs both forms once on random tensors of exactly that shape, dtype and device,
forward and backward, and batches the layout only if they agree bit for bit; otherwise that layout keeps the
upstream per-chunk calls.
"""
import sys

import torch
import torch.nn.functional as F

_FAMILIES = {  # transformers modeling module -> (attention class, vision model class, has window attention)
    "transformers.models.qwen3_vl.modeling_qwen3_vl": ("Qwen3VLVisionAttention", "Qwen3VLVisionModel", False),
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl": (
        "Qwen2_5_VLVisionAttention", "Qwen2_5_VisionTransformerPretrainedModel", True),
    "transformers.models.qwen2_vl.modeling_qwen2_vl": ("VisionAttention", "Qwen2VisionTransformerPretrainedModel", False),
    # transformers 5: the same attention code (per-chunk calls after lengths.tolist() unless flash-attention)
    "transformers.models.qwen3_5.modeling_qwen3_5": ("Qwen3_5VisionAttention", "Qwen3_5VisionModel", False),
}
_installed = set()
_MAX_ENTRIES = 16
decisions = {}  # (chunks, length, heads, head_dim, dtype, device, with gradients) -> batched call is bitwise identical


class _State:
    """Chunk layouts of the vision forward in progress, each (number of chunks, chunk length) or None when the
    chunks differ: `frames` for full attention, `windows` for window attention. Set once per vision forward."""

    frames = None
    windows = None
    windowed = False


def _uniform(lengths):
    if lengths and all(x == lengths[0] for x in lengths):
        return (len(lengths), lengths[0])
    return None


def _frame_layout(rows):
    return _uniform([h * w for t, h, w in rows for _ in range(t)])  # cu_seqlens: one chunk of h*w patches per frame


def _window_layout(vision_model, grid_thw):
    bounds = vision_model.get_window_index(grid_thw)[1]  # cumulative window ends, as the forward computes them
    bounds = [b for i, b in enumerate(bounds) if i == 0 or b != bounds[i - 1]]  # torch.unique_consecutive
    return _uniform([b - a for a, b in zip(bounds, bounds[1:])])


# attention implementations that keep the upstream code: flash-attention already makes one varlen call per layer
_UPSTREAM_ONLY = ("eager", "flash_attention_2")


def _bitwise(layout, heads, head_dim, dtype, device):
    """Is one attention call on [chunks, heads, length, head_dim] bitwise identical to one call per chunk here?"""
    if layout is None:
        return layout
    with_grad = torch.is_grad_enabled()
    key = (*layout, heads, head_dim, str(dtype), str(device), with_grad)
    if key not in decisions:
        chunks, length = layout
        generator = torch.Generator().manual_seed(0)

        def random():
            return torch.randn(chunks, heads, length, head_dim, generator=generator).to(device=device, dtype=dtype)

        with torch.autocast(device.type, enabled=False):
            q, k, v, weight = random(), random(), random(), random()
            if with_grad:
                q, k, v = q.requires_grad_(), k.requires_grad_(), v.requires_grad_()
            def attention(q, k, v):  # the call transformers' "sdpa" attention makes for these layers
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0, scale=head_dim**-0.5, is_causal=False
                )

            batched = attention(q, k, v)
            single = torch.cat([attention(q[i:i + 1], k[i:i + 1], v[i:i + 1]) for i in range(chunks)])
            same = torch.equal(batched, single)
            if same and with_grad:
                grads = [torch.autograd.grad((out * weight).sum(), (q, k, v)) for out in (batched, single)]
                same = all(torch.equal(a, b) for a, b in zip(*grads))
        decisions[key] = same
        if not same:
            print(f"[starVLA] vision_attn: {chunks} chunks x {length} tokens ({heads} heads x {head_dim}, "
                  f"{dtype}) is not bitwise identical when batched on {device}; these layers keep the upstream "
                  f"attention", flush=True)
    return layout if decisions[key] else None


def _layout(cu_seqlens):
    if not _State.windowed:
        return _State.frames
    chunks = cu_seqlens.shape[0] - 1
    for layout in (_State.frames, _State.windows):
        if layout is not None and layout[0] == chunks:
            return layout
    return None


def install(modeling):
    """Patch the vision attention and vision model classes of one transformers modeling module."""
    if modeling.__name__ in _installed:
        return
    attention_name, model_name, windowed = _FAMILIES[modeling.__name__]
    attention_class, model_class = getattr(modeling, attention_name), getattr(modeling, model_name)
    original = attention_class.forward
    original_model_forward = model_class.forward
    layouts = {}  # (model, grid) -> (frames, windows)

    def model_forward(self, hidden_states, grid_thw, **kwargs):
        if self.config._attn_implementation in _UPSTREAM_ONLY:
            return original_model_forward(self, hidden_states, grid_thw, **kwargs)
        rows = tuple(tuple(row) for row in grid_thw.tolist())  # one host read per forward
        attention = self.blocks[0].attn
        weight = attention.qkv.weight
        device = weight.device
        dtype = torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else weight.dtype
        key = (id(self), rows, dtype, torch.is_grad_enabled())
        if key not in layouts:
            if len(layouts) >= _MAX_ENTRIES:
                layouts.pop(next(iter(layouts)))
            shape = (attention.num_heads, weight.shape[1] // attention.num_heads, dtype, device)
            layouts[key] = (
                _bitwise(_frame_layout(rows), *shape),
                _bitwise(_window_layout(self, grid_thw), *shape) if windowed else None,
            )
        _State.frames, _State.windows = layouts[key]
        _State.windowed = windowed
        try:
            return original_model_forward(self, hidden_states, grid_thw, **kwargs)
        finally:
            _State.frames = _State.windows = None
            _State.windowed = False

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        impl = self.config._attn_implementation
        uniform = None if impl in _UPSTREAM_ONLY else _layout(cu_seqlens)
        if uniform is None:
            if rotary_pos_emb is not None:  # transformers 4.x only; the 5.x signature has no such argument
                kwargs["rotary_pos_emb"] = rotary_pos_emb
            return original(self, hidden_states, cu_seqlens, position_embeddings=position_embeddings, **kwargs)
        kwargs.pop("max_seqlen", None)  # transformers 5.x: used by the flash-attention branch only
        chunks, length = uniform
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = modeling.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # [seq, heads, d] -> [chunks, heads, length, d]: one batch element per chunk
        def per_chunk(states):
            return states.reshape(chunks, length, self.num_heads, -1).transpose(1, 2)

        attention_interface = modeling.ALL_ATTENTION_FUNCTIONS[impl]
        attn_output = attention_interface(
            self,
            per_chunk(query_states),
            per_chunk(key_states),
            per_chunk(value_states),
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )[0]  # [chunks, length, heads, d]
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    forward._starvla_original = original
    attention_class.forward = forward
    model_class.forward = model_forward
    _installed.add(modeling.__name__)


def _attention_implementation(*modules):
    """The attention implementation ("sdpa", "flash_attention_2", "eager") of the first of these Hugging Face
    (sub)models whose config states one."""
    for module in modules:
        implementation = getattr(getattr(module, "config", None), "_attn_implementation", None)
        if implementation is not None:
            return implementation
    return None


def apply(vlm):
    """Batch the vision attention if the backbone's vision tower is one of the Qwen families."""
    visual = getattr(getattr(getattr(vlm, "model", None), "model", None), "visual", None)
    family = type(visual).__module__
    if visual is None or family not in _FAMILIES:
        return False
    if _attention_implementation(visual, vlm.model) in _UPSTREAM_ONLY:
        return False
    install(sys.modules[family])
    return True
