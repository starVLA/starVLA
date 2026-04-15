# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import os
import torch
from typing import Optional
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoProcessor, AutoModelForMultimodalLM


from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100


def _distributed_world_size() -> int:
    """World size for DDP / Accelerate / torchrun; 1 if not in a multi-process job."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())
    for key in ("WORLD_SIZE", "ACCELERATE_TOTAL_PROCESSES"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return max(1, int(val))
            except ValueError:
                break
    return 1


_ACTION_TOKEN_MIN = 151669 # how can we know this range? check how you add fast tokens into VLM
_ACTION_TOKEN_MAX = 153716 # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md

import torch.nn as nn


class _Gemma_VL_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Gemma 4 multimodal model.

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the Gemma 4 multimodal wrapper.
        """
        super().__init__()

        Gemma_vl_config = config.framework.get("Gemma_vl", config.framework.get("qwenvl", {}))
        model_id = Gemma_vl_config.get("base_vlm", "google/gemma-4-E2B-it")
        # HF + Accelerate: device_map="auto" is incompatible with multi-process training (DDP / DeepSpeed).
        # Load on CPU / default device here; Accelerate.prepare() places shards on each rank.
        device_map = None if _distributed_world_size() > 1 else "auto"
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        processor = AutoProcessor.from_pretrained(model_id)
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.config = config

        if not hasattr(self.model.config, "hidden_size"):
            if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
                self.model.config.hidden_size = self.model.config.text_config.hidden_size
            elif hasattr(self.model, "language_model") and hasattr(self.model.language_model.config, "hidden_size"):
                self.model.config.hidden_size = self.model.language_model.config.hidden_size

        # only for fast base model
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

    def _get_model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return getattr(self.model, "device", torch.device("cpu"))

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to the underlying Gemma 4 backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )

        return outputs

    def generate(
        self,
        **kwargs,
    ):
        """
        High-level generation interface (auto-regressive decoding), optionally vision-conditioned.

        Args:
            **kwargs: fully follow raw model.generate() signature.
        Returns:
            GenerateOutput | Model-dependent generation return.
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_Gemma_vl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Build model inputs from raw data (images + instructions + optional solutions).
        Follow the official Gemma 4 multimodal chat format.
        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=solutions is None,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "padding": True,
                "return_mm_token_type_ids": True,
            },
        )

        # if solutions, mask out the solution tokens in labels
        if solutions is not None: #  here only for fast_tokenizer now. 
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer, see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md
            labels = batch_inputs['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning (f"action token are on in yout tokenizer, plz see starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md.")
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_inputs['labels'] = labels

        return batch_inputs.to(self._get_model_device())

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        # Backward-compatible alias for older framework call sites.
        return self.build_Gemma_vl_inputs(images=images, instructions=instructions, solutions=solutions, **kwargs)




if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy  # type: ignore[reportMissingImports]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    
    if "Gemma_vl" not in cfg.framework:
        cfg.framework.Gemma_vl = {}
    cfg.framework.Gemma_vl.base_vlm = "./playground/Pretrained_models/google/gemma-4-E2B-it"
    gemma_vl = _Gemma_VL_Interface(cfg)
    pass
