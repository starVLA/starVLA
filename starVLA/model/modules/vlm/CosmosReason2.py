# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Haron Wan / CUHK Shenzhen] in [2026].

from typing import Optional

import torch
import torch.nn as nn
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).parents[1]
SEPARATOR = "-" * 20

PIXELS_PER_TOKEN = 32**2
"""Number of pixels per visual token."""


class _CosmosReason2_Interface(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()
        qwenvl_config = config.framework.get("qwenvl", {})
        model_name = qwenvl_config.get("base_vlm", "nvidia/Cosmos-Reason2-2B")

        attn_implementation = qwenvl_config.get("attn_implementation", "sdpa")
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation
        )
        self.processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)
        self.config = config

        self.model.config.hidden_size = self.model.config.text_config.hidden_size

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(
                **kwargs,
            )
        return outputs

    def generate(
        self,
        **kwargs,
    ):
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(
                **kwargs,
            )
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
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

            messages.append(msg)

        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            # fps=4,
        )

        return inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "playground/Pretrained_models/nvidia/Cosmos-Reason2-2B"
    cfg.framework.qwenvl.attn_implementation = "sdpa"
    qwen_vl = _CosmosReason2_Interface(cfg)

    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "assets/starvla_LIBERO.png",
                },
                {"type": "text", "text": "What is the robot most likely to do?"},
            ],
        },
    ]

    # Process inputs
    inputs = qwen_vl.processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Run inference
    generated_ids = qwen_vl.model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = qwen_vl.processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)

    print("Done!")


