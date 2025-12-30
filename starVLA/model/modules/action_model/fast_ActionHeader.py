# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""Fast Action Tokenizer Adapter
"this file is adapted from https://huggingface.co/physical-intelligence/fast"

Overview:
    This module encapsulates a lightweight "action → language model-readable sequence" converter (Fast_Action_Tokenizer).
    Its core objective is to convert continuous/discrete raw robot actions (raw_actions) into
    pseudo-natural language token strings like <robot_action_12><robot_action_3><robot_action_87> ...
    This facilitates direct integration into multimodal large models (VLM/LLM) dialogue templates,
    leveraging their language modeling capabilities for action prediction.
"""

import torch.nn as nn
from typing import List, Dict, Any, Callable, Optional
import os
import numpy as np
from transformers import AutoProcessor



class Fast_Action_Tokenizer(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, fast_tokenizer_name="playground/Pretrained_models/fast"):
        super().__init__()
        
        self.fast_tokenizer = AutoProcessor.from_pretrained(
            fast_tokenizer_name, trust_remote_code=True
        ) # load https://huggingface.co/physical-intelligence/fast


    def encoder_action2fastoken(self, raw_actions):
        # x: (batch_size, chunck, dim)
        batch_actions = np.stack(raw_actions, axis=0)  # (B, T, D)
        batch_fast_tokens = self.fast_tokenizer(batch_actions)

        return batch_fast_tokens # List[str]
    
    def decoder_action(self, generated_ids):
        # api https://huggingface.co/physical-intelligence/fast
        # return: (batch_size, chunck, dim)
        pred_actions = self.fast_tokenizer.decode([generated_ids - self._ACTION_TOKEN_MIN])
        return pred_actions
    

    def fit_tokenizer_on_datasets(self, action_dataset, datasets_path="<your_local_path>", ):
        # 如果 datasets_path 存在， 直接读取
        if os.path.exists(datasets_path):

            self.fast_tokenizer = AutoProcessor.from_pretrained(
            datasets_path, trust_remote_code=True
        )
            return
        else:
            # 如果不存在，Fit the tokenizer on the new dataset
            new_tokenizer = self.fast_tokenizer.tokenizer.fit(action_dataset)
            self.fast_tokenizer = new_tokenizer

            # Save the new tokenizer, optionally push it to the Hugging Face model hub
            self.fast_tokenizer.save_pretrained(datasets_path)


def get_action_model(config=None):
    """
    Factory: build ActionModel from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).
    Returns:
        ActionModel: Initialized diffusion action head.
    """
    action_model = Fast_Action_Tokenizer()

    return action_model


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10094))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10094 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True

if __name__ == "__main__":

    start_debugpy_once()

    fast_tokenizer_name = "physical-intelligence/fast"
    fast_tokenizer = Fast_Action_Tokenizer(fast_tokenizer_name=fast_tokenizer_name)
    raw_actions = [np.random.randn(16, 7), np.random.randn(16, 7)]

    # Load the tokenizer from the Hugging Face hub
    tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_name, trust_remote_code=True)

    # basic test
    # Tokenize & decode action chunks (we use dummy data here)
    action_data = np.random.rand(2, 16, 7)    # one batch of action chunks
    tokens = tokenizer(action_data)              # tokens = list[int]
    decoded_actions = tokenizer.decode(tokens)

    # self func test
    vlm_tokens = fast_tokenizer.encoder_action2vlmtoken(raw_actions)
    print(vlm_tokens)
    pred_actions = fast_tokenizer.decoder_action(np.array([12,3,45,87]))
    print(pred_actions)





