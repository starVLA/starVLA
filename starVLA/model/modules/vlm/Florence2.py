# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import torch
from typing import Optional, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Dict, Optional, List
from typing import List, Union, Dict, Optional


import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 


from accelerate.logging import get_logger

logger = get_logger(__name__)

# IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = 151655
# VIDEO_TOKEN_INDEX = 151656
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_VIDEO_TOKEN = "<video>"

# [151936, 153984]

import torch.nn as nn
def _construct_prompts(text):

    return text

class _Florence_Interface(nn.Module):
    """
    This exists because of the diversity of VLMs, so we encapsulate the changes here.
    Lightweight wrapper around Qwen3-VL (Qwen3VLForConditionalGeneration).

    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.

    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the VLM wrapper.
        Following https://huggingface.co/microsoft/Florence-2-large

        """
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "microsoft/Florence-2-large")

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=True, attn_implementation="eager" ) # 强制使用 eager 注意力
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.processor._construct_prompts = _construct_prompts
        self.config = config

        # alin with qwen2.5
        self.model.config.hidden_size = self.model.config.projection_dim


        # del unused moduals to save memory
        if hasattr(self.model, "decoder"):
            del self.model.decoder
        if hasattr(self.model, "lm_head"):
            del self.model.lm_head

    def forward(
        self,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying Qwen2.5-VL backbone.
        """

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.forward_vlm(
                **kwargs,
            )

        return outputs

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, C, H, W] --> [B, H, W]
        **kwargs
    ):
        """
        # copyright from X-VLA https://github.com/2toinf/X-VLA/blob/main/models/modeling_florence2.py

        Encode text + multi-view images via Florence2 encoder.
        Returns:
          enc_out.hidden_states: [B, T_enc, D]
        """
        # get image features
           
        param_dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(self.model.device, dtype=param_dtype)
        valid_feats = self.model._encode_image(pixel_values)      # [B, N, D]
        B_multiview, N, D = valid_feats.shape
        # get text embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)  # [B, L, D]

        # # olny support single image from florence, your can modify here for multi-image support by merge each image features
        # like pixel_values: B*N_view, C, H, W --> B*N_view, N_token, D -> B, N_view*N_token, D -> image_features
        B, L, _ = inputs_embeds.shape
        image_features = valid_feats.view(B, -1, D)  # [B, N_view*N, D]

        # merge image features and text embeddings
        merged_embeds, attention_mask = self.model._merge_input_ids_with_image_features(
            image_features,  # first view: [B, N, D]
            inputs_embeds,         # [B, L, D]
        )
        
        # TODO should return text index and image index for later index masking
        
        enc_out = self.model.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )
        enc_out.hidden_states = [enc_out.last_hidden_state]
        # last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
        return enc_out
    
    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        """
        Build model inputs from raw data (images + instructions).
        Follow Oficial Florence 2 format: https://huggingface.co/microsoft/Florence-2-large
        """

        # Create messages: one message per sample
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        assert len(images[0]) == 1, "Florence2 only support batch size 1 for now"
        # # # olny support single image from florence, your can modify here for multi-image support by merge each image features
        flatten_batch_images = []
        for exameple_images in images:
            flatten_batch_images.extend(exameple_images)
        # images = [image[0] for image in  images]
        task_prompt = "Locate the objects with category name in the image." #"Locate the objects with category name in the image."
        for index in range(len(instructions)):
            instruction = instructions[index]
            instructions[index] = task_prompt + " " + instruction
        
        # olny support single image for a text input from florence, your can modify here for multi-image support by merge each image features
        inputs = self.processor(text=instructions, images=flatten_batch_images, return_tensors="pt", padding=True, truncation=True,)
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs.to(self.model.device)





if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # model_id = "microsoft/Florence-2-large"
    model_id = "playground/Pretrained_models/Florence-2-large"
    cfg.framework.qwenvl.base_vlm = model_id
    qwen_vl = _Florence_Interface(cfg)
    qwen_vl.model.eval()

    import requests

    import torch
    from PIL import Image

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    prompt = "<OD>"

    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = qwen_vl.build_qwenvl_inputs(images=[[image]], instructions=[prompt])
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = qwen_vl.forward_vlm(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
        )
    print(f"forward_vlm last_hidden_state shape: {outputs.last_hidden_state.shape}")
    print(f"forward_vlm hidden_states length: {len(outputs.hidden_states)}")

