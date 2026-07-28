# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# LLaVA/VILA multimodal splicing, vendored from EgoVLA `llava/model/llava_arch.py`
# (original LLaVA team / Haotian Tang / Jason Lu, Apache-2.0). Trimmed to the
# inference + single-GPU training path: no sequence-parallel, no input packing,
# no past_key_values/generation cache.
"""Multimodal input assembly for the EgoVLA VILA backbone.

``prepare_inputs_labels_for_multimodal`` splices projected image tokens into the
text-embedding stream at ``IMAGE_TOKEN_INDEX`` positions and produces the padded
``(inputs_embeds, attention_mask, position_ids, labels)`` the LLM consumes.
Kept numerically identical to VILA so an EgoVLA checkpoint behaves the same.
"""

import warnings

import torch

from . import IGNORE_INDEX, IMAGE_TOKEN_INDEX


def prepare_inputs_labels_for_multimodal(
    llm,
    embed_tokens,
    image_features,          # (num_images, T_img, H)  already projected
    input_ids,               # (B, L)  with IMAGE_TOKEN_INDEX placeholders
    attention_mask,          # (B, L) or None
    position_ids,            # (B, L) or None
    labels,                  # (B, L) or None
    padding_side: str = "right",
    max_length: int | None = None,
):
    """Interleave image tokens into the text stream (VILA-faithful).

    Returns (inputs_embeds, attention_mask, position_ids, labels) padded to the
    batch max sequence length. Image positions carry IGNORE_INDEX labels.
    """
    _labels = labels
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # Embed text; image placeholder ids are out of vocab, temporarily map to 0.
    input_ids_copy = input_ids.clone()
    input_ids_copy[input_ids_copy == IMAGE_TOKEN_INDEX] = 0
    input_embeds = embed_tokens(input_ids_copy)

    # Drop padding per-sample.
    input_ids = [cur[m] for cur, m in zip(input_ids, attention_mask)]
    input_embeds_1 = [cur[m] for cur, m in zip(input_embeds, attention_mask)]
    labels = [cur[m] for cur, m in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        cur_input_embeds = input_embeds_1[batch_idx]
        if num_images == 0:
            # No image: append a zero-width image slice to keep grad graph consistent.
            cur_cat = torch.cat([cur_input_embeds, image_features[cur_image_idx][0:0]], dim=0)
            new_input_embeds.append(cur_cat)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = (
            [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        )
        cur_labels = labels[batch_idx]
        cur_ids_noim, cur_labels_noim, cur_embeds_noim = [], [], []
        for i in range(len(image_token_indices) - 1):
            lo, hi = image_token_indices[i] + 1, image_token_indices[i + 1]
            cur_ids_noim.append(cur_input_ids[lo:hi])
            cur_labels_noim.append(cur_labels[lo:hi])
            cur_embeds_noim.append(cur_input_embeds[lo:hi])

        cur_new_embeds, cur_new_labels = [], []
        for i in range(num_images + 1):
            cur_new_embeds.append(cur_embeds_noim[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_img = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_embeds.append(cur_img)
                cur_new_labels.append(
                    torch.full((cur_img.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype)
                )

        new_input_embeds.append(torch.cat(cur_new_embeds))
        new_labels.append(torch.cat(cur_new_labels))

    # Optional truncation.
    if max_length is not None:
        if any(x.shape[0] > max_length for x in new_input_embeds):
            warnings.warn("EgoVLA multimodal inputs truncated to max_length.")
        new_input_embeds = [x[:max_length] for x in new_input_embeds]
        new_labels = [x[:max_length] for x in new_labels]

    # Pad to batch max length.
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)
    H = new_input_embeds[0].shape[-1]
    dtype, device = new_input_embeds[0].dtype, new_input_embeds[0].device

    embeds_padded = torch.zeros((batch_size, max_len, H), dtype=dtype, device=device)
    labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=device)
    attn_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    pos_padded = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, (emb, lab) in enumerate(zip(new_input_embeds, new_labels)):
        n = emb.shape[0]
        if padding_side == "left":
            embeds_padded[i, max_len - n :] = emb
            labels_padded[i, max_len - n :] = lab
            attn_padded[i, max_len - n :] = True
            pos_padded[i, max_len - n :] = torch.arange(n, dtype=torch.long, device=device)
        else:
            embeds_padded[i, :n] = emb
            labels_padded[i, :n] = lab
            attn_padded[i, :n] = True
            pos_padded[i, :n] = torch.arange(n, dtype=torch.long, device=device)

    return embeds_padded, attn_padded, pos_padded, labels_padded
