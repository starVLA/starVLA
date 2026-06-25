# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
"""
MiniCPM-V VL Interface for starVLA.

Mirrors `_QWen3_VL_Interface` so existing VLM4A frameworks can swap in
`openbmb/MiniCPM-V-4.6` through `framework.qwenvl.base_vlm` without framework
changes.

MiniCPM-V 4.6 is a compact VLM with a SigLIP2-400M vision encoder and a
Qwen3.5-0.8B text tower (1.3B total parameters). It uses the standard
Transformers `AutoModelForImageTextToText` / `AutoProcessor` path introduced
for transformers >= 5.7.0.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


class _MiniCPM_VL_Interface(nn.Module):
    """
    Lightweight wrapper around `openbmb/MiniCPM-V-4.6` checkpoints.

    Purpose:
        - Match the interface of `_QWen3_VL_Interface` exactly so framework files
          can be VLM-agnostic.
        - Centralize MiniCPM-V preprocessing (chat template + multimodal packing).
        - Surface `model.config.hidden_size = text_config.hidden_size` so downstream
          DiT / flow-matching heads can read it the same way they read it for Qwen3-VL.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "openbmb/MiniCPM-V-4.6")
        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        enable_grad_ckpt = bool(qwenvl_config.get("enable_gradient_checkpointing", False))

        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[MiniCPM-V][WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"

        if enable_grad_ckpt:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                print("[MiniCPM-V] gradient_checkpointing ENABLED (use_reentrant=False)", flush=True)
            except Exception as e:
                print(f"[MiniCPM-V] failed to enable gradient_checkpointing: {e}", flush=True)

        self.model = model
        self.processor = processor
        self.config = config

        text_cfg = getattr(self.model.config, "text_config", None)
        if text_cfg is None or not hasattr(text_cfg, "hidden_size"):
            raise RuntimeError(
                f"[MiniCPM-V] could not locate text_config.hidden_size on `{model_id}`. "
                "Check that the checkpoint is a MiniCPM-V 4.6 multimodal checkpoint."
            )
        self.model.config.hidden_size = text_cfg.hidden_size

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass through the underlying MiniCPM-V backbone.

        Action / flow-matching heads consume only `hidden_states`, so we cap
        `logits_to_keep=1` by default when the model accepts it to avoid
        materializing a large (B, L, vocab) bf16 tensor. If the installed
        Transformers implementation does not accept this kwarg, retry without it.
        """
        kwargs.setdefault("logits_to_keep", 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            try:
                outputs = self.model(**kwargs)
            except TypeError as e:
                if "logits_to_keep" not in str(e):
                    raise
                kwargs.pop("logits_to_keep", None)
                outputs = self.model(**kwargs)
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build a model-ready batch from raw (images, instructions). Same name and
        signature as `_QWen3_VL_Interface.build_qwenvl_inputs`.

        Args:
            images:       list[list[PIL.Image]] — multi-view images per sample.
            instructions: list[str]            — one instruction per sample.
            solutions:    optional list[str]   — assistant turn for SFT-style training.

        Returns:
            BatchFeature on `self.model.device`.
        """
        assert len(images) == len(instructions), "images and instructions must batch-align"

        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:
                cot_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = cot_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                msg.append(
                    {"role": "assistant", "content": [{"type": "text", "text": solutions[len(messages)]}]}
                )
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=(solutions is None),
            return_dict=True,
            return_tensors="pt",
        )
        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse

    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="openbmb/MiniCPM-V-4.6",
        help="HF model id or local path for MiniCPM-V 4.6.",
    )
    parser.add_argument("--attn", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {
                    "base_vlm": args.model_id,
                    "attn_implementation": args.attn,
                }
            },
            "datasets": {"vla_data": {}},
        }
    )

    iface = _MiniCPM_VL_Interface(cfg)
    print(f"[MiniCPM-V] hidden_size = {iface.model.config.hidden_size}")
    print(f"[MiniCPM-V] num_hidden_layers (text) = {iface.model.config.text_config.num_hidden_layers}")
    print(f"[MiniCPM-V] device = {next(iface.model.parameters()).device}")

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    batch = iface.build_qwenvl_inputs(
        images=[[img, img], [img, img]],
        instructions=["Pick up the red block.", "Stack the green cube on the blue cube."],
    )
    print(f"[MiniCPM-V] input_ids shape = {batch['input_ids'].shape}")
    out = iface(**batch, output_hidden_states=True, return_dict=True)
    print(f"[MiniCPM-V] num hidden states emitted = {len(out.hidden_states)}")
    print(f"[MiniCPM-V] last hidden state shape = {tuple(out.hidden_states[-1].shape)}")
    print("[MiniCPM-V] OK")
