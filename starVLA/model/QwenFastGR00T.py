# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025]. 

"""
Qwen-Fast Framework

A lightweight implementation for autoregressive discrete action prediction conditioned on multi-view images + instruction.
fast tokenizer is copyright from physical-intelligence/fast

Key Points:
  - Qwen2.5 vision-language backbone
  - Unified action learning via next-token prediction (fast tokenizer)
  - Autoregressive action tokens derived from discretized / symbolized continuous actions

Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
"""

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info


from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images

from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model as get_groot_action_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead

@FRAMEWORK_REGISTRY.register("QwenFast")
class Qwenvl_Fast(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.action_model = get_action_model(config=self.config)
        self.groot_action_model: FlowmatchingActionHead = get_groot_action_model(config=self.config) 

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.hidden_dim = config.framework.action_model.action_hidden_dim
        
        self.action_model.fast_tokenizer.time_horizon = self.future_action_window_size + 1
        self.action_model.fast_tokenizer.action_dim = self.config.framework.action_model.action_dim

        

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward: directly predict future actions via next-token prediction (no diffusion).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B, len, 7]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # step 0: map_raw_action_to_vlm_action
        vlm_action_tokens = self.action_model.encoder_action2vlmtoken(actions)  # List[str]


        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions, solutions=vlm_action_tokens)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        
        vlm_action_loss = qwenvl_outputs.loss
        if vlm_action_loss is None or torch.isnan(vlm_action_loss): 
            vlm_action_loss = torch.tensor(0.0, device=self.qwen_vl_interface.model.device)

        
        
        # Step 2: Create encoder_attention_mask based
        # last_hidden_state: [B, seq_len, H]
        last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
        # index label = -100 的token embedding set
        # Create encoder_attention_mask based
        encoder_attention_mask = (qwen_inputs.labels == -100).to(dtype=torch.float32)  # -100 -> 1, others -> 0

        # Step 3: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            encoder_attention_mask = encoder_attention_mask.repeat(repeated_diffusion_steps, 1)
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            fm_action_loss = self.groot_action_model(last_hidden_repeated, actions_target_repeated, state_repeated, encoder_attention_mask=encoder_attention_mask)  # (B, chunk_len, action_dim)


        return {"action_loss": vlm_action_loss + fm_action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: single forward pass to obtain future actions (no diffusion sampling).
        # can be batch forward
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
    
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction for instruction in instructions]


        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        
        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


    def _extract_action_token_ids(
        self,
        generated_ids: torch.LongTensor,
    ) -> List[List[int]]:
        """
        Extract action tokens (with offset) from the generated token sequence and return a 2D list:
        ret[b] = [vlm_action_token_id_0, vlm_action_token_id_1, ...]
        Rule: keep all tokens falling within [_ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX] in order of appearance.
        You may change it to "take only the first occurrence followed by continuous segment" as needed.
        """
        act_min = self.action_model._ACTION_TOKEN_MIN
        act_max = self.action_model._ACTION_TOKEN_MAX
        mask = (generated_ids >= act_min) & (generated_ids <= act_max)  # [B, L]
        results = []
        for b in range(generated_ids.size(0)):
            idx = mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                results.append([])
                continue
            # all action tokens
            tokens = generated_ids[b, idx].tolist()
            results.append(tokens)
        return results

    def _decode_action_tokens(self, batch_vlm_tokens: List[List[int]]) -> List[Any]:
        """
        Decode the offset VLM action token list back to fast tokenizer semantics.
        fast_tokenizer.decode expects the original fast token id sequence (without offset).
        """
        act_min = self.action_model._ACTION_TOKEN_MIN
        batch_fast_token_ids = []
        for seq in batch_vlm_tokens:
            if not seq:
                batch_fast_token_ids.append(None)
                continue
            fast_ids = [t - act_min for t in seq]
            
            batch_fast_token_ids.append(fast_ids)
        
        return batch_fast_token_ids



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/starvla_cotrain_libero.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()
    # args.config_yaml = "/mnt/petrelfs/yejinhui/Projects/starVLA/results/Checkpoints/0_bar/1025_libero4in1_qwenfast/config.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action"

    # try get model
    model = Qwenvl_Fast(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "The fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action. for new model, it didn't learn to predict action token, so you would meet empty action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")




    # # test with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset_cfg.video_backend = "torchvision_av"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)
    pass
    action = model.predict_action(batch[0])
