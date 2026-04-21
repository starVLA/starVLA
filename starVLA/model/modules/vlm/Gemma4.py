# Copyright 2026 starVLA / gemma-vla community.
# Licensed under the MIT License.
"""
Gemma 4 VL Interface for gemma-vla.

Mirrors `_QWen3_VL_Interface` (see ./QWen3.py) but loads `google/gemma-4-E2B-it`
via `Gemma4ForConditionalGeneration`. Designed so that any starVLA framework that
already calls `qwen_vl_interface.{forward, build_qwenvl_inputs}` works unchanged
when the dispatcher in `__init__.py` returns this class instead.

Verified config (from google/gemma-4-E2B-it/config.json):
    architectures           = ["Gemma4ForConditionalGeneration"]
    text_config.hidden_size = 1536
    text_config.num_hidden_layers = 35
    text_config.num_attention_heads = 8
    text_config.num_key_value_heads = 1
    text_config.sliding_window = 512
    text_config.vocab_size  = 262144
    vision_config.hidden_size = 768
    vision_config.num_hidden_layers = 16
    vision_config.patch_size = 16
"""

import torch
import torch.nn as nn
from typing import Optional, List

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoProcessor

try:
    from transformers import Gemma4ForConditionalGeneration  # transformers >= 5.x
except ImportError as _e:  # pragma: no cover
    Gemma4ForConditionalGeneration = None
    _GEMMA4_IMPORT_ERROR = _e

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100


class _Gemma4_VL_Interface(nn.Module):
    """
    Lightweight wrapper around `Gemma4ForConditionalGeneration` (`google/gemma-4-E2B-it`).

    Purpose:
        - Match the interface of `_QWen3_VL_Interface` exactly so framework files
          can be VLM-agnostic.
        - Centralize Gemma 4 specific preprocessing (chat template + image-budget pinning).
        - Surface `model.config.hidden_size = text_config.hidden_size` so downstream
          DiT heads can read it the same way they read it for Qwen3-VL.

    Notes:
        - Audio tower is loaded by default; can be deleted post-load to save memory if needed
          (set `framework.qwenvl.drop_audio_tower=True` in config).
        - PLE (Per-Layer Embeddings) injection happens *before* image soft tokens are merged.
          Image-position hidden states are still produced by the standard decoder forward
          and should carry useful visual signal — verified in M0 smoke test.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        if Gemma4ForConditionalGeneration is None:
            raise ImportError(
                "Gemma4ForConditionalGeneration is not available in the installed `transformers` version. "
                "Upgrade with `pip install --upgrade 'transformers>=5.5.0'`. "
                f"Original import error: {_GEMMA4_IMPORT_ERROR}"
            )

        # We reuse the `framework.qwenvl` namespace so config files stay simple.
        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "google/gemma-4-E2B-it")
        attn_impl = qwenvl_config.get("attn_implementation", "flash_attention_2")
        drop_audio = bool(qwenvl_config.get("drop_audio_tower", False))
        # Gradient checkpointing is OFF by default. Training scripts must set this
        # to True via config (`framework.qwenvl.enable_gradient_checkpointing: true`)
        # — note that starVLA's `trainer.enable_gradient_checkpointing` flag is dead
        # config (issue #41), this flag actually wires it to the underlying HF model.
        enable_grad_ckpt = bool(qwenvl_config.get("enable_gradient_checkpointing", False))

        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        # Left padding so the rightmost positions hold the most recent text/action context.
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"

        # Optional: drop audio tower to save memory if we never use it.
        if drop_audio:
            for attr in ("audio_tower", "audio_model", "audio_encoder"):
                if hasattr(model, attr):
                    delattr(model, attr)
                    logger.info(f"[Gemma4] dropped attribute `{attr}` to save memory.")
                    break

        # Real gradient checkpointing — saves ~30-50% activation memory at the cost
        # of one extra forward pass per backward. Critical for fitting Gemma 4 E2B
        # + DiT-36 PI head into 80GB at BS>2.
        if enable_grad_ckpt:
            try:
                # use_reentrant=False is the modern path; required for nn.Module hooks
                # and works with deepspeed ZeRO-2/3.
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                # HF requires inputs to require grad when ckpt is on; this is the
                # canonical way to enable that without breaking embedding layers.
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                # Use print rather than accelerate.logging — the latter requires an
                # Accelerator() to be initialized (fine during training, breaks during
                # standalone eval / inference scripts that never construct one).
                print("[Gemma4] gradient_checkpointing ENABLED (use_reentrant=False)", flush=True)
            except Exception as e:
                print(f"[Gemma4] failed to enable gradient_checkpointing: {e}", flush=True)

        self.model = model
        self.processor = processor
        self.config = config

        # Surface text hidden size at the top level so downstream code that
        # reads `model.config.hidden_size` (e.g. QwenGR00T's cross_attention_dim
        # alignment) keeps working without a special case.
        try:
            self.model.config.hidden_size = self.model.config.text_config.hidden_size
        except AttributeError as e:
            raise RuntimeError(
                f"[Gemma4] could not align hidden_size from text_config: {e}. "
                f"Check that `{model_id}` is a Gemma4 multimodal checkpoint."
            )

        assert self.model.config.hidden_size == 1536, (
            f"[Gemma4] expected hidden_size=1536 for E2B, got {self.model.config.hidden_size}. "
            f"Update DiT cross_attention_dim accordingly."
        )

    # ----- forward / generate -------------------------------------------------

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass through the underlying Gemma 4 backbone. Honors `output_hidden_states=True`,
        which the PI / GR00T heads rely on.

        Note: PI/GR00T never consume the LM logits (only `hidden_states`), so we cap
        `logits_to_keep=1` by default. This avoids materializing a (B, L, vocab=262144) bf16
        tensor (~150 MB at L=280). Callers can override.
        """
        kwargs.setdefault("logits_to_keep", 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**kwargs)
        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    # ----- input building -----------------------------------------------------

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build a model-ready batch from raw (images, instructions). Same name and signature as
        `_QWen3_VL_Interface.build_qwenvl_inputs` so framework files do not need to know which
        backend they are talking to.

        Args:
            images:       list[list[PIL.Image]] — one inner list of multi-view images per sample.
            instructions: list[str]            — one instruction per sample.
            solutions:    optional list[str]   — assistant turn for SFT-style training. Not used by
                                                 the flow-matching VLA frameworks; included for parity.

        Returns:
            BatchFeature on `self.model.device`, containing at minimum `input_ids`, `attention_mask`,
            and the multimodal pixel features the Gemma 4 processor produces.
        """
        assert len(images) == len(instructions), "images and instructions must batch-align"

        # Gemma 4 best practice: image content blocks BEFORE the text instruction.
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            # Optional CoT prompt rewrite, mirrors QWen3.py behavior.
            cot_prompt = None
            if hasattr(self.config, "datasets") and hasattr(self.config.datasets, "vla_data"):
                cot_prompt = getattr(self.config.datasets.vla_data, "CoT_prompt", None)
            if cot_prompt:
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
    # Smoke test: load the model on a single GPU and verify hidden-state shapes.
    import argparse
    from PIL import Image
    import numpy as np
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HF model id or local path",
    )
    parser.add_argument("--attn", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "framework": {
                "qwenvl": {
                    "base_vlm": args.model_id,
                    "attn_implementation": args.attn,
                    "drop_audio_tower": True,
                }
            },
            "datasets": {"vla_data": {}},
        }
    )

    iface = _Gemma4_VL_Interface(cfg)
    print(f"[Gemma4] hidden_size = {iface.model.config.hidden_size}")
    print(f"[Gemma4] num_hidden_layers (text) = {iface.model.config.text_config.num_hidden_layers}")
    print(f"[Gemma4] device = {next(iface.model.parameters()).device}")

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    batch = iface.build_qwenvl_inputs(
        images=[[img, img], [img, img]],
        instructions=["Pick up the red block.", "Stack the green cube on the blue cube."],
    )
    print(f"[Gemma4] input_ids shape = {batch['input_ids'].shape}")
    out = iface(**batch, output_hidden_states=True, return_dict=True)
    hs = out.hidden_states  # tuple of length num_layers + 1
    print(f"[Gemma4] num hidden states emitted = {len(hs)}")
    print(f"[Gemma4] last hidden state shape = {tuple(hs[-1].shape)}")
    # PLE / sliding-window sanity: per-layer image-position variance should be non-trivial.
    var_per_layer = torch.stack([h.float().var(dim=(0, 1)).mean() for h in hs])
    print(f"[Gemma4] mean per-layer feature variance: {var_per_layer.tolist()}")
    print("[Gemma4] OK")
