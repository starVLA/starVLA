"""Patches to the Hugging Face model code behind the VLM interfaces. Every one leaves its results bitwise identical (in
deterministic mode, where the original code reproduces itself): it moves work off the critical path or leaves out work
whose result nobody reads. Each `patch(vlm) -> bool` checks the model it is given and returns False where it does not
apply; `get_vlm_model` runs them on every backbone interface (BACKBONE_SPEEDUPS in this package's __init__).

    patch_attention_mask       causal + padding mask from two tensor ops instead of nested torch.vmap
    skip_unused_lm_head        no vocabulary logits when the caller reads hidden states only
    batch_vision_attention     one attention call per vision layer instead of one per image (Qwen vision towers)
    cache_vision_positions     vision position tables built once per grid size (Qwen vision towers)
"""

import sys

import torch
import torch.nn.functional as F
from transformers import masking_utils

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------------------------------------------------
# Attention mask
# ---------------------------------------------------------------------------------------------------------------------
# A direct formulation of transformers' SDPA attention-mask builder.
#
# transformers (4.57) builds the 4-D boolean attention mask of a causal LM by applying a per-element predicate through
# four nested torch.vmap calls (masking_utils.sdpa_mask_recent_torch). For the common case, a causal mask combined with
# a 2-D padding mask, the result is simply
#
#     mask[b, 0, q, k] = (k_index <= q_index) & padding_mask[b, k_index]
#
# Building it with two broadcasted tensor ops gives the same boolean tensor; the vmap formulation costs about 150 ms of
# host time per call at batch 16 x 205 tokens, while the GPU waits. `patch_attention_mask` registers the direct builder
# for the "sdpa" attention implementation through transformers' own registry (AttentionMaskInterface.register); every
# case it does not cover (extra and/or mask functions, packed sequences, local attention) is delegated to the original.
#
# Applies to every backbone whose transformers model builds its mask with masking_utils.create_causal_mask (the Qwen,
# Gemma, Llama and Phi families in transformers 4.57), whatever the framework on top.


_mask_installed = None  # None: not tried; True / False: registered or found to disagree


def _install_attention_mask():
    global _mask_installed
    if _mask_installed is not None:
        return
    original = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_mask(
        batch_size,
        cache_position,
        kv_length,
        kv_offset=0,
        mask_function=masking_utils.causal_mask_function,
        attention_mask=None,
        local_size=None,
        allow_is_causal_skip=True,
        **kwargs,
    ):
        if mask_function is not masking_utils.causal_mask_function or local_size is not None:
            return original(
                batch_size=batch_size,
                cache_position=cache_position,
                kv_length=kv_length,
                kv_offset=kv_offset,
                mask_function=mask_function,
                attention_mask=attention_mask,
                local_size=local_size,
                allow_is_causal_skip=allow_is_causal_skip,
                **kwargs,
            )
        q_length = cache_position.shape[0]
        padding_mask = masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        if allow_is_causal_skip and masking_utils._ignore_causal_mask_sdpa(
            padding_mask, q_length, kv_length, kv_offset, local_size
        ):
            return None
        kv_arange = torch.arange(kv_length, device=cache_position.device)
        kv_arange += kv_offset
        mask = (kv_arange[None, :] <= cache_position[:, None])[None, None, :, :]  # causal_mask_function
        if padding_mask is not None:
            mask = mask & padding_mask[:, kv_arange][:, None, None, :].to(mask.device)  # padding_mask_function
        return mask.expand(batch_size, 1, q_length, kv_length).contiguous()

    if not _agrees(sdpa_mask, original):  # another transformers version may build the mask differently
        _mask_installed = False
        return
    sdpa_mask._starvla_original = original
    masking_utils.AttentionMaskInterface.register("sdpa", sdpa_mask)
    _mask_installed = True


def _agrees(ours, theirs):
    """The direct builder returns what the installed transformers returns, on padded and unpadded batches, with and
    without a key / value offset. Any exception counts as disagreement."""
    try:
        for batch, queries, offset, padded in ((3, 7, 0, True), (2, 5, 0, False), (2, 4, 3, True), (1, 1, 6, True)):
            kv_length = queries + offset
            attention_mask = torch.ones(batch, kv_length, dtype=torch.long)
            if padded:
                attention_mask[0, :2] = 0
                attention_mask[-1, -1:] = 0 if batch > 1 else 1
            arguments = dict(
                batch_size=batch,
                cache_position=torch.arange(offset, offset + queries),
                kv_length=kv_length,
                kv_offset=0,
                attention_mask=attention_mask,
            )
            for skip in (True, False):
                a, b = ours(allow_is_causal_skip=skip, **arguments), theirs(allow_is_causal_skip=skip, **arguments)
                if (a is None) != (b is None) or (a is not None and not (a.shape == b.shape and torch.equal(a, b))):
                    return False
        return True
    except Exception:
        return False


def _attention_implementation(*modules):
    """The attention implementation ("sdpa", "flash_attention_2", "eager") of the first of these Hugging Face
    (sub)models whose config states one."""
    for module in modules:
        implementation = getattr(getattr(module, "config", None), "_attn_implementation", None)
        if implementation is not None:
            return implementation
    return None


def patch_attention_mask(vlm):
    """Register the direct builder if the backbone's transformers model uses masking_utils."""
    model = getattr(vlm, "model", None)
    modeling = sys.modules.get(type(model).__module__)
    if not hasattr(modeling, "create_causal_mask") or "sdpa" not in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS:
        return False
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if _attention_implementation(language_model, model) != "sdpa":  # the other implementations use other builders
        return False
    _install_attention_mask()
    return bool(_mask_installed)


# ---------------------------------------------------------------------------------------------------------------------
# Hidden states without the vocabulary logits
# ---------------------------------------------------------------------------------------------------------------------
# No vocabulary logits for callers that only read hidden states.
#
# Most frameworks condition their action head on the backbone's hidden states:
#
#     outputs = vlm(**inputs, output_hidden_states=True, return_dict=True)
#     last_hidden = outputs.hidden_states[-1]
#
# The Hugging Face `...ForConditionalGeneration` wrapper then also projects every position onto the vocabulary
# (batch x tokens x 150 k logits, forward and backward), which nobody reads. Here such a call (hidden
# states requested, no labels) runs the wrapper's own forward with the language-model head switched off, so every
# other field of the output is exactly what upstream returns, including whichever convention the wrapper has for
# hidden_states[-1]. `logits` is computed on first access, so a caller that does read it still gets the upstream
# value. Calls with labels, or without output_hidden_states=True, are untouched.
#
# Applies to every backbone whose model has an `lm_head`. If the first such call fails without the head (a wrapper that
# post-processes its logits), the default forward is used from then on.


class HiddenOnlyOutput:
    """The wrapper's outputs, with the language-model head applied only if `logits` is read. `head` is the head
    module, called as a module so that its hooks run (DeepSpeed ZeRO-3 gathers the weight in a pre-forward hook)."""

    def __init__(self, outputs, head, head_input):
        self._outputs = outputs
        self._head = head
        self._head_input = head_input
        self._logits = None

    @property
    def logits(self):
        if self._logits is None:
            self._logits = self._head(self._head_input)
        return self._logits

    def __getattr__(self, name):  # hidden_states, past_key_values, attentions, loss, ...
        return getattr(self._outputs, name)

    def __getitem__(self, key):
        return self.logits if key == "logits" else self._outputs[key]


def skip_unused_lm_head(vlm):
    model = getattr(vlm, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        return False
    if getattr(model, "_starvla_hidden_only", False):
        return True
    upstream_forward, upstream_head = model.forward, lm_head.forward
    call = {"skip": False, "head_input": None, "works": None}  # works: None until the first such call has run

    def head_forward(hidden_states, *args, **kwargs):
        if not call["skip"]:
            return upstream_head(hidden_states, *args, **kwargs)
        call["head_input"] = hidden_states
        return None

    def forward(*args, **kwargs):
        hidden_only = (
            kwargs.get("output_hidden_states") is True
            and kwargs.get("labels") is None
            and kwargs.get("return_dict", True) is not False
            # an interface that already keeps the logits of a few positions only (Gemma 4, MiniCPM-V pass
            # logits_to_keep=1): nothing left to save, and Gemma 4 post-processes its logits (soft-capping)
            and not (isinstance(kwargs.get("logits_to_keep"), int) and kwargs["logits_to_keep"] > 0)
        )
        if not hidden_only:
            return upstream_forward(*args, **kwargs)
        if call["works"] is False:
            return upstream_forward(*args, **kwargs)
        call["skip"] = True
        try:
            outputs = upstream_forward(*args, **kwargs)
        except Exception as error:
            if call["works"]:  # it has worked before: this is the caller's error
                raise
            call["skip"], call["works"] = False, False  # a wrapper that needs its logits: leave it alone from now on
            logger.info(
                f"hidden_only: {type(model).__name__} cannot run without its language-model head "
                f"({type(error).__name__}: {error}); using the default forward"
            )
            return upstream_forward(*args, **kwargs)
        finally:
            call["skip"] = False
        head_input, call["head_input"] = call["head_input"], None
        if head_input is None:  # the head was not called: nothing was skipped, nothing to add
            return outputs
        call["works"] = True
        return HiddenOnlyOutput(outputs, lm_head, head_input)

    lm_head.forward = head_forward
    model.forward = forward
    model._starvla_hidden_only = True
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Batched vision attention
# ---------------------------------------------------------------------------------------------------------------------
# Batched attention for the Qwen vision towers when every image has the same number of patches.
#
# The vision attention of Qwen2-VL, Qwen2.5-VL and Qwen3-VL (the same code in transformers 4.57), and of Qwen3.5
# (transformers 5), receives all patches
# of all images as one sequence plus the chunk boundaries (cu_seqlens). With any implementation but flash-attention it
# reads the chunk lengths back to the host in every layer (`lengths.tolist()`: a GPU sync per layer) and then runs one
# attention call per chunk: 24 layers x 32 images per training step at batch 16 x 2 views.
#
# When all chunks have the same length (the usual case in training, where every view is resized to one resolution) the
# chunks are a batch: one attention call per layer on [chunks, heads, length, dim], and no host read in the layers.
# The layout is derived once per vision forward from grid_thw on the host. Qwen2.5-VL alternates between full
# attention (one chunk per frame) and window attention (one chunk per window); each kind is batched when it is
# uniform. Any other case falls back to the upstream code.
#
# Batching is the same arithmetic, but a GPU library may pick a different matrix-product kernel for a different batch
# size, and then the last bit can differ (on B300 with bf16: 32 chunks of 256 tokens are bitwise identical to 32
# single calls at head dimension 64, not at 72 or 80; chunks of 64 tokens are identical at all three). So before a
# layout is batched, `vision_attn` runs both forms once on random tensors of exactly that shape, dtype and device,
# forward and backward, and batches the layout only if they agree bit for bit; otherwise that layout keeps the
# upstream per-chunk calls.


_ATTENTION_FAMILIES = {  # transformers modeling module -> (attention class, vision model class, has window attention)
    "transformers.models.qwen3_vl.modeling_qwen3_vl": ("Qwen3VLVisionAttention", "Qwen3VLVisionModel", False),
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl": (
        "Qwen2_5_VLVisionAttention",
        "Qwen2_5_VisionTransformerPretrainedModel",
        True,
    ),
    "transformers.models.qwen2_vl.modeling_qwen2_vl": (
        "VisionAttention",
        "Qwen2VisionTransformerPretrainedModel",
        False,
    ),
    # transformers 5: the same attention code (per-chunk calls after lengths.tolist() unless flash-attention)
    "transformers.models.qwen3_5.modeling_qwen3_5": ("Qwen3_5VisionAttention", "Qwen3_5VisionModel", False),
}
_attention_installed = set()
_MAX_LAYOUTS = 16
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
    return _uniform([b - a for a, b in zip(bounds, bounds[1:], strict=False)])


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
            single = torch.cat([attention(q[i : i + 1], k[i : i + 1], v[i : i + 1]) for i in range(chunks)])
            same = torch.equal(batched, single)
            if same and with_grad:
                grads = [torch.autograd.grad((out * weight).sum(), (q, k, v)) for out in (batched, single)]
                same = all(torch.equal(a, b) for a, b in zip(*grads, strict=False))
        decisions[key] = same
        if not same:
            logger.info(
                f"vision_attn: {chunks} chunks x {length} tokens ({heads} heads x {head_dim}, "
                f"{dtype}) is not bitwise identical when batched on {device}; these layers keep the upstream "
                f"attention"
            )
    return layout if decisions[key] else None


def _layout(cu_seqlens):
    if not _State.windowed:
        return _State.frames
    chunks = cu_seqlens.shape[0] - 1
    for layout in (_State.frames, _State.windows):
        if layout is not None and layout[0] == chunks:
            return layout
    return None


def _install_vision_attention(modeling):
    """Patch the vision attention and vision model classes of one transformers modeling module."""
    if modeling.__name__ in _attention_installed:
        return
    attention_name, model_name, windowed = _ATTENTION_FAMILIES[modeling.__name__]
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
            if len(layouts) >= _MAX_LAYOUTS:
                layouts.pop(next(iter(layouts)))
            # not from weight.shape: DeepSpeed ZeRO-3 leaves an empty placeholder outside the module's forward
            shape = (attention.num_heads, attention.qkv.in_features // attention.num_heads, dtype, device)
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
    _attention_installed.add(modeling.__name__)


def batch_vision_attention(vlm):
    """Batch the vision attention if the backbone's vision tower is one of the Qwen families."""
    visual = getattr(getattr(getattr(vlm, "model", None), "model", None), "visual", None)
    family = type(visual).__module__
    if visual is None or family not in _ATTENTION_FAMILIES:
        return False
    if _attention_implementation(visual, vlm.model) in _UPSTREAM_ONLY:
        return False
    _install_vision_attention(sys.modules[family])
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Vision position tables
# ---------------------------------------------------------------------------------------------------------------------
# Memoized position tables for the Qwen vision towers (and, with transformers 5, every tower built on vision_utils).
#
# The vision models of Qwen2-VL, Qwen2.5-VL and Qwen3-VL compute, in every forward and with a Python loop over the
# images that reads GPU scalars back to the host, things that depend only on the grid sizes (grid_thw), not on any
# weight:
#
#   rot_pos_emb(grid_thw)                 the rotary position table of every patch (a lookup in a fixed buffer)
#   get_window_index(grid_thw)            Qwen2.5-VL: the window order of the patches and the window boundaries
#   fast_pos_embed_interpolate(grid_thw)  Qwen3-VL: the four neighbour indices and bilinear weights of every patch
#
# In training every batch has the same grids (all views are resized to one resolution), so each is built once per
# distinct grid_thw and reused. Of fast_pos_embed_interpolate only the indices and weights are reused: the part that
# touches the trainable position embedding (lookup, weighting, the sum of the four neighbours, the per-image
# reordering) runs every time exactly as upstream, so its values and gradients are the upstream ones. About 27 ms of
# host time per step at 32 images on Qwen3-VL.


_positions_installed = set()
_MAX_TABLES = 16


def _grid_key(grid_thw):
    return tuple(tuple(row) for row in grid_thw.tolist())  # one host read per forward


def _table_key(grid_thw, *rest):
    """The key of a table: the grid, and whether inference mode is on. A table made under torch.inference_mode()
    (predict_action) is an inference tensor, which autograd refuses to save for a training step's backward, so the
    two modes keep separate tables."""
    return (_grid_key(grid_thw), torch.is_inference_mode_enabled(), *rest)


def _remember(cache, key, value):
    if len(cache) >= _MAX_TABLES:
        cache.pop(next(iter(cache)))
    cache[key] = value
    return value


def _memoize_rot_pos_emb(model_class):
    original_rot = model_class.rot_pos_emb
    rot_cache = {}

    def rot_pos_emb(self, grid_thw):
        inv_freq = self.rotary_pos_emb.inv_freq
        key = _table_key(grid_thw, str(inv_freq.device), str(inv_freq.dtype), id(self))
        hit = rot_cache.get(key)
        if hit is None:
            hit = _remember(rot_cache, key, original_rot(self, grid_thw).detach())
        return hit

    rot_pos_emb._starvla_original = original_rot
    model_class.rot_pos_emb = rot_pos_emb


def _memoize_window_index(model_class):
    original = model_class.get_window_index
    cache = {}

    def get_window_index(self, grid_thw):
        key = _table_key(grid_thw, id(self))
        hit = cache.get(key)
        if hit is None:
            hit = _remember(cache, key, original(self, grid_thw))
        window_index, cu_window_seqlens = hit
        return window_index, list(cu_window_seqlens)

    get_window_index._starvla_original = original
    model_class.get_window_index = get_window_index


def _install_qwen3(qwen3):
    _memoize_rot_pos_emb(qwen3.Qwen3VLVisionModel)
    interp_cache = {}

    def interp_indices(self, grid):
        """idx_tensor and weight_tensor of the upstream function, built from host integers (same arithmetic)."""
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for t, h, w in grid:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=self.pos_embed.weight.device)
        return idx_tensor, weight_tensor

    def fast_pos_embed_interpolate(self, grid_thw):
        grid = _grid_key(grid_thw)
        weight = self.pos_embed.weight
        key = (grid, torch.is_inference_mode_enabled(), str(weight.device), str(weight.dtype), self.num_grid_per_side)
        hit = interp_cache.get(key)
        if hit is None:
            hit = _remember(interp_cache, key, interp_indices(self, grid))
        idx_tensor, weight_tensor = hit

        # from here on: upstream, with host integers for the grid sizes
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for _, h, w in grid])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid, strict=False):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    fast_pos_embed_interpolate._starvla_original = qwen3.Qwen3VLVisionModel.fast_pos_embed_interpolate
    qwen3.Qwen3VLVisionModel.fast_pos_embed_interpolate = fast_pos_embed_interpolate


def _install_qwen2_5(qwen2_5):
    _memoize_rot_pos_emb(qwen2_5.Qwen2_5_VisionTransformerPretrainedModel)
    _memoize_window_index(qwen2_5.Qwen2_5_VisionTransformerPretrainedModel)


def _install_qwen2(qwen2):
    _memoize_rot_pos_emb(qwen2.Qwen2VisionTransformerPretrainedModel)


_POSITION_FAMILIES = {
    "transformers.models.qwen3_vl.modeling_qwen3_vl": _install_qwen3,
    "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl": _install_qwen2_5,
    "transformers.models.qwen2_vl.modeling_qwen2_vl": _install_qwen2,
}


# ---- transformers 5: the vision towers take these tables from module-level helpers (transformers.vision_utils), each
# of which returns a precomputed value instead when the caller passes one. Qwen3-VL, Qwen3.5 and MiniCPM-V 4.6 / 4.7
# call them in every forward: per-image Python loops (position ids), dozens of small GPU ops (interpolation indices,
# cumulative lengths) and host reads. Memoizing the helpers per grid gives the same tensors without that work.
_PRECOMPUTED = frozenset(
    {
        "cu_seqlens",
        "max_seqlen",
        "position_ids",
        "interp_indices",
        "interp_weights",
        "bilinear_indices",
        "bilinear_weights",
        "window_index",
        "cu_window_seqlens",
        "merged_shape",
    }
)
_last_grid = [None, None, None]  # the grid tensor last read, its version counter, its key


def _frozen(value):
    return tuple(_frozen(v) for v in value) if isinstance(value, list) else value


def _tensor_key(tensor):
    """The contents of a small index tensor (grid_thw, target sizes): one host read, shared by the helpers of a forward.

    The key holds whether inference mode is on: a table made under it is an inference tensor, which autograd refuses
    to save for a training step's backward."""
    mode = torch.is_inference_mode_enabled()
    if tensor.is_inference():  # no version counter: read it every time
        return (_frozen(tensor.tolist()), tensor.dtype, str(tensor.device), mode)
    if _last_grid[0] is tensor and _last_grid[1] == tensor._version:
        return _last_grid[2] + (mode,)
    key = (_frozen(tensor.tolist()), tensor.dtype, str(tensor.device))
    _last_grid[:] = [tensor, tensor._version, key]
    return (*key, mode)


def _argument_key(value):
    if isinstance(value, torch.Tensor):
        return _tensor_key(value)
    if hasattr(value, "_attn_implementation"):  # a config: only the attention implementation matters (max_seqlen)
        return ("config", id(value), value._attn_implementation)
    hash(value)  # TypeError for anything else unhashable: no memoization
    return value


def _memoized_helper(helper):
    cache = {}

    def memoized(grid, *args, kwargs=None, **options):
        if kwargs and _PRECOMPUTED & kwargs.keys():  # the caller passes its own table: upstream pops and returns it
            return helper(grid, *args, kwargs=kwargs, **options)
        try:
            key = (
                _argument_key(grid),
                tuple(_argument_key(a) for a in args),
                tuple(sorted((k, _argument_key(v)) for k, v in options.items())),
            )
        except TypeError:
            return helper(grid, *args, kwargs=kwargs, **options)
        if key not in cache:
            _remember(cache, key, helper(grid, *args, kwargs=kwargs, **options))
        return cache[key]

    memoized._starvla_original = helper
    return memoized


def _install_vision_utils(modeling):
    """Memoize the transformers.vision_utils helpers that this modeling module calls. False if it calls none."""
    try:
        from transformers import vision_utils
    except ImportError:
        return False
    found = False
    for name, value in list(vars(modeling).items()):
        if name.startswith("get_") and callable(value) and getattr(vision_utils, name, None) is value:
            setattr(modeling, name, _memoized_helper(value))
            found = True
        elif getattr(value, "_starvla_original", None) is not None and name.startswith("get_"):
            found = True
    return found


def _vision_tower(vlm):
    core = getattr(getattr(vlm, "model", None), "model", None)
    for name in ("visual", "vision_tower"):
        tower = getattr(core, name, None)
        if tower is not None:
            return tower
    return None


def cache_vision_positions(vlm):
    """Memoize the position tables if the backbone's vision tower is one of the Qwen families (transformers 4), or
    builds them with the helpers of transformers.vision_utils (transformers 5)."""
    visual = _vision_tower(vlm)
    if visual is None:
        return False
    family = type(visual).__module__
    modeling = sys.modules.get(family)
    if family in _positions_installed:
        return True
    if modeling is not None and _install_vision_utils(modeling):
        _positions_installed.add(family)
        return True
    if family not in _POSITION_FAMILIES:
        return False
    try:
        _POSITION_FAMILIES[family](modeling)
    except AttributeError as error:  # another transformers version, whose classes lack these methods
        logger.info(f"vision_pos: {family} has another layout here ({error}); left alone")
        return False
    _positions_installed.add(family)
    return True
