"""A direct formulation of transformers' SDPA attention-mask builder.

transformers (4.57) builds the 4-D boolean attention mask of a causal LM by applying a per-element predicate through
four nested torch.vmap calls (masking_utils.sdpa_mask_recent_torch). For the common case, a causal mask combined with
a 2-D padding mask, the result is simply

    mask[b, 0, q, k] = (k_index <= q_index) & padding_mask[b, k_index]

Building it with two broadcasted tensor ops gives the same boolean tensor; the vmap formulation costs about 150 ms of
host time per call at batch 16 x 205 tokens, while the GPU waits. `install()` registers the direct builder for the
"sdpa" attention implementation through transformers' own registry (AttentionMaskInterface.register); every case it
does not cover (extra and/or mask functions, packed sequences, local attention) is delegated to the original.

Applies to every backbone whose transformers model builds its mask with masking_utils.create_causal_mask (the Qwen,
Gemma, Llama and Phi families in transformers 4.57), whatever the framework on top.
"""
import sys

import torch
from transformers import masking_utils

_installed = None  # None: not tried; True / False: registered or found to disagree


def install():
    global _installed
    if _installed is not None:
        return
    original = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS["sdpa"]

    def sdpa_mask(batch_size, cache_position, kv_length, kv_offset=0,
                  mask_function=masking_utils.causal_mask_function, attention_mask=None, local_size=None,
                  allow_is_causal_skip=True, **kwargs):
        if mask_function is not masking_utils.causal_mask_function or local_size is not None:
            return original(batch_size=batch_size, cache_position=cache_position, kv_length=kv_length,
                            kv_offset=kv_offset, mask_function=mask_function, attention_mask=attention_mask,
                            local_size=local_size, allow_is_causal_skip=allow_is_causal_skip, **kwargs)
        q_length = cache_position.shape[0]
        padding_mask = masking_utils.prepare_padding_mask(attention_mask, kv_length, kv_offset, _slice=False)
        if allow_is_causal_skip and masking_utils._ignore_causal_mask_sdpa(
                padding_mask, q_length, kv_length, kv_offset, local_size):
            return None
        kv_arange = torch.arange(kv_length, device=cache_position.device)
        kv_arange += kv_offset
        mask = (kv_arange[None, :] <= cache_position[:, None])[None, None, :, :]      # causal_mask_function
        if padding_mask is not None:
            mask = mask & padding_mask[:, kv_arange][:, None, None, :].to(mask.device)  # padding_mask_function
        return mask.expand(batch_size, 1, q_length, kv_length).contiguous()

    if not _agrees(sdpa_mask, original):  # another transformers version may build the mask differently
        _installed = False
        return
    sdpa_mask._starvla_original = original
    masking_utils.AttentionMaskInterface.register("sdpa", sdpa_mask)
    _installed = True


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
            arguments = dict(batch_size=batch, cache_position=torch.arange(offset, offset + queries), kv_length=kv_length,
                             kv_offset=0, attention_mask=attention_mask)
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


def apply(vlm):
    """Register the direct builder if the backbone's transformers model uses masking_utils."""
    model = getattr(vlm, "model", None)
    modeling = sys.modules.get(type(model).__module__)
    if not hasattr(modeling, "create_causal_mask") or "sdpa" not in masking_utils.ALL_MASK_ATTENTION_FUNCTIONS:
        return False
    language_model = getattr(getattr(model, "model", None), "language_model", None)
    if _attention_implementation(language_model, model) != "sdpa":  # the other implementations use other builders
        return False
    install()
    return bool(_installed)
