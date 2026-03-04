# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-Adapter Framework
A lightweight implementation that Qwen-VL + Adapter Action head to directly predict continuous actions
Action head is copyright from VLA-Adapter,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from deployment.model_server.tools.image_tools import to_pil_preserve


from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.VLA_AdapterHeader import get_action_model, VLA_Adapter_L1RegressionActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.vlm.QWen3 import IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX

def get_image_token_counts(batch_inputs):
    IMAGE_TOKEN_ID = IMAGE_TOKEN_INDEX 
    
    # input_ids shape: [Batch_Size, Seq_Len]
    # result shape: [Batch_Size]
    num_tokens_per_sample = torch.sum(batch_inputs['input_ids'] == IMAGE_TOKEN_ID, dim=1)
    # also get the last index of the image token for each sample if needed
    last_index_per_sample = (batch_inputs['input_ids'] == IMAGE_TOKEN_ID).int().cumsum(dim=1).argmax(dim=1)
    # also get the first index of the image token for each sample if needed
    first_index_per_sample = (batch_inputs['input_ids'] == IMAGE_TOKEN_ID).int().cumsum(dim=1).argmin(dim=1)
    
    return num_tokens_per_sample, first_index_per_sample, last_index_per_sample


class ProprioProjector(nn.Module):
    """
    Projects proprio state inputs into the LLM's embedding space.
    """
    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.llm_dim = llm_dim
        self.proprio_dim = proprio_dim

        self.fc1 = nn.Linear(self.proprio_dim, self.llm_dim, bias=True)
        self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor = None) -> torch.Tensor:
        # proprio: (bsz, proprio_dim)
        projected_features = self.fc1(proprio)
        projected_features = self.act_fn1(projected_features)
        projected_features = self.fc2(projected_features)
        return projected_features

# Only support for Qwen2.5 now @ PR 60
@FRAMEWORK_REGISTRY.register("QwenAdapter")
class Qwen_Adapter(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings

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
        self.phase = self.config.framework.action_model.get("phase", "Training")
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.qwenvl.vl_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_query_num = self.config.framework.action_model.get("action_query_num", 64)
        self.action_model: VLA_Adapter_L1RegressionActionHead = get_action_model(config=self.config)
        self.action_query = nn.Parameter(torch.randn(self.action_query_num, self.qwen_vl_interface.model.config.hidden_size))
        self.dummy_action_token = "🔍" # TODO also can add spacail token to Qwen, but too complex
        self.dummy_action_token_id = self.qwen_vl_interface.processor.tokenizer("🔍", add_special_tokens=False)["input_ids"][0]
        self.dummy_action_prompt = self.dummy_action_token * self.action_query_num
        self.chunk_len = self.config.framework.action_model.get("num_actions_chunk", None)
        if self.chunk_len is None:
            raise ValueError("num_actions_chunk must be specified in action_model config.")
        if self.config.framework.action_model.get("use_proprio", False):
            self.proprio_projector = ProprioProjector(
                llm_dim=self.qwen_vl_interface.model.config.hidden_size,
                proprio_dim=self.config.framework.action_model.get("state_dim", 14),
            )
        else:
            self.proprio_projector = None
        nn.init.normal_(self.action_query, mean=0.0, std=0.02)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        gt_actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        # # debug print 
        # print(f'gt action shape is {np.array(gt_actions).shape}')
        # raise NotImplementedError("Debug stop here.")
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        # ! often state is None 
        # ============================================================
        # FIX: Insert action placeholder tokens BEFORE tokenization
        # ============================================================
        
        # Append to instruction text (will be tokenized naturally)
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.dummy_action_prompt}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]
        
        # Step 1: Build Qwen-VL inputs with modified instructions
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions
        )
        # Now: [BOS, text, <img>, more_text, 🔍, 🔍, ..., 🔍, EOS]
        #                                    ^^^^^^^^^^^^^^^^
        #                                    Action placeholders BEFORE EOS
        # Create mask for action token positions
        input_ids = qwen_inputs['input_ids']
        action_mask = (input_ids == self.dummy_action_token_id)  # [B, L]
            
        # ============================================================
        # Hook to replace action token embeddings (OPTIMIZED)
        # ============================================================
        # Pre-compute action positions outside the hook
        batch_size = qwen_inputs['input_ids'].shape[0]
        device = qwen_inputs['input_ids'].device
        action_positions_tensor = torch.full((batch_size, self.action_query_num), 0, dtype=torch.long, device=device)
        valid_counts = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            act_pos = torch.where(action_mask[b])[0]
            if len(act_pos) == self.action_query_num:
                action_positions_tensor[b] = act_pos
                valid_counts[b] = True

        def inject_query_hook(module, inputs, output):
            """Replace action placeholder embeddings with learnable queries (VECTORIZED)."""
            query_embed = self.action_query.to(dtype=output.dtype, device=output.device)  # [N, H]
            
            # Vectorized replacement using advanced indexing
            batch_indices = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
            
            # Only update valid samples (where action token count matches)
            valid_batch_indices = batch_indices[valid_counts]
            valid_action_positions = action_positions_tensor[valid_counts]
            
            if len(valid_batch_indices) > 0:
                output[valid_batch_indices, valid_action_positions, :] = query_embed.unsqueeze(0)
            
            return output
        # Register hook on text embedding layer (this is OK!)
        embedding_layer = self.qwen_vl_interface.model.model.get_input_embeddings()
        hook_handle = embedding_layer.register_forward_hook(inject_query_hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()
                
        hidden_states = qwenvl_outputs.hidden_states # list of [B, L, H]
        # ============================================================
        # Extract features (FULLY VECTORIZED)
        # ============================================================
        multi_layer_hidden_states = []
        num_images, first_index_per_sample, last_index_per_sample = get_image_token_counts(qwen_inputs)
        
        max_patch_len = -999
        for b in range(batch_size):
            sample_patch_len = last_index_per_sample[b] - first_index_per_sample[b] + 1
            if sample_patch_len > max_patch_len:
                max_patch_len = sample_patch_len.item()
        
        for layer_hidden in hidden_states[0:]:
            # layer_hidden: [B, L, H]
            
            # ============================================================
            # 1. Vision Features (Fully Vectorized)
            # ============================================================
            # Create batch of indices [B, max_patch_len]
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_patch_len)  # [B, max_patch_len]
            seq_indices = torch.arange(max_patch_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, max_patch_len]

            # Add first_index_per_sample offset to get actual positions
            seq_indices = seq_indices + first_index_per_sample.unsqueeze(1)  # [B, max_patch_len]

            # Clamp to valid range (shouldn't exceed last_index_per_sample)
            seq_indices = torch.clamp(seq_indices, max=last_index_per_sample.unsqueeze(1))  # [B, max_patch_len]

            # Advanced indexing to extract vision features
            batch_vision_states = layer_hidden[batch_indices, seq_indices, :]  # [B, max_patch_len, H]

            # Mask padding - now based on actual vision patch lengths per sample
            vision_patch_lengths = last_index_per_sample - first_index_per_sample + 1  # [B]
            padding_mask = torch.arange(max_patch_len, device=device).unsqueeze(0) >= vision_patch_lengths.unsqueeze(1)  # [B, max_patch_len]
            batch_vision_states = batch_vision_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
            # ============================================================
            # 2. Action Query Features (Fully Vectorized)
            # ============================================================
            # Use advanced indexing
            # When you index with two tensors in the first two dims, PyTorch treats them as matching coordinates:
            # batch_indices_action is shape [B, N]
            # action_positions_tensor is shape [B, N]
            
            batch_indices_action = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
            action_query_states = layer_hidden[batch_indices_action, action_positions_tensor, :]  # [B, action_query_num, H]
            
            # ============================================================
            # 3. Concatenate
            # ============================================================
            all_hidden_states = torch.cat([
                batch_vision_states.unsqueeze(1),  # [B, 1, max_patch_len, H]
                action_query_states.unsqueeze(1)   # [B, 1, action_query_num, H]
            ], dim=2)  # [B, 1, L_total, H]
            
            multi_layer_hidden_states.append(all_hidden_states)

        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)  # [B, num_layers, L_total, H]
        state_projected = None
        if state is not None: # repeat state 
            state = torch.tensor(
                    np.array(state), device=multi_layer_hidden_states.device, dtype=multi_layer_hidden_states.dtype
                ) #  [B, 1, state_dim]
            if self.proprio_projector is not None:
                state_projected = self.proprio_projector(proprio=state.squeeze(1))  # [B, llm_dim]

        # Step 3: Action Expert Forward
        self.action_model = self.action_model.to(device=multi_layer_hidden_states.device, dtype=multi_layer_hidden_states.dtype)
        predicted_actions = self.action_model.predict_action(
            multi_layer_hidden_states,
            vision_hidden_len=max_patch_len,
            state_projected=state_projected,
            phase=self.phase,
        ) # (B, chunk_len, action_dim)

        gt_actions = torch.tensor(np.stack(gt_actions)).to(
            device=predicted_actions.device, 
            dtype=predicted_actions.dtype
        )

        loss = torch.nn.L1Loss()(predicted_actions, gt_actions)

        return {"action_loss": loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: Predict future continuous actions aligned with the Forward logic (Hook + Multi-layer states).
        
        Steps:
          1. Resize images to training resolution (if specified)
          2. Insert action placeholder tokens into instruction
          3. Encode with QwenVL (hidden states retained) with hook to inject action queries
          4. Extract multi-layer features at action query positions
          5. Predict actions via action model
          6. Return normalized action trajectory
        
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, chunk_len, action_dim], predicted normalized actions.
        """
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
    
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        
        # ============================================================
        # Insert action placeholder tokens into instruction
        # ============================================================
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.dummy_action_prompt}<action>."
        instructions = [instruction + prompt_suffix for instruction in instructions]
        
        # Step 1: Build Qwen-VL inputs with modified instructions
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions
        )
        
        # Create mask for action token positions
        input_ids = qwen_inputs['input_ids']
        action_mask = (input_ids == self.dummy_action_token_id)  # [B, L]
        
        # ============================================================
        # Hook to replace action token embeddings (OPTIMIZED)
        # ============================================================
        # Pre-compute action positions outside the hook
        batch_size = qwen_inputs['input_ids'].shape[0]
        device = qwen_inputs['input_ids'].device
        action_positions_tensor = torch.full((batch_size, self.action_query_num), 0, dtype=torch.long, device=device)
        valid_counts = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for b in range(batch_size):
            act_pos = torch.where(action_mask[b])[0]
            if len(act_pos) == self.action_query_num:
                action_positions_tensor[b] = act_pos
                valid_counts[b] = True

        def inject_query_hook(module, inputs, output):
            """Replace action placeholder embeddings with learnable queries (VECTORIZED)."""
            query_embed = self.action_query.to(dtype=output.dtype, device=output.device)  # [N, H]
            
            # Vectorized replacement using advanced indexing
            batch_indices = torch.arange(batch_size, device=output.device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
            
            # Only update valid samples (where action token count matches)
            valid_batch_indices = batch_indices[valid_counts]
            valid_action_positions = action_positions_tensor[valid_counts]
            
            if len(valid_batch_indices) > 0:
                output[valid_batch_indices, valid_action_positions, :] = query_embed.unsqueeze(0)
            
            return output
        # Register hook on text embedding layer (this is OK!)
        embedding_layer = self.qwen_vl_interface.model.model.get_input_embeddings()
        hook_handle = embedding_layer.register_forward_hook(inject_query_hook)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            hook_handle.remove()
                
        hidden_states = qwenvl_outputs.hidden_states # list of [B, L, H]
        # ============================================================
        # Extract features (FULLY VECTORIZED)
        # ============================================================
        multi_layer_hidden_states = []
        num_images, first_index_per_sample, last_index_per_sample = get_image_token_counts(qwen_inputs)
        
        max_patch_len = -999
        for b in range(batch_size):
            sample_patch_len = last_index_per_sample[b] - first_index_per_sample[b] + 1
            if sample_patch_len > max_patch_len:
                max_patch_len = sample_patch_len.item()
        
        for layer_hidden in hidden_states[0:]:
            # layer_hidden: [B, L, H]
            
            # ============================================================
            # 1. Vision Features (Fully Vectorized)
            # ============================================================
            # Create batch of indices [B, max_patch_len]
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, max_patch_len)  # [B, max_patch_len]
            seq_indices = torch.arange(max_patch_len, device=device).unsqueeze(0).expand(batch_size, -1)  # [B, max_patch_len]

            # Add first_index_per_sample offset to get actual positions
            seq_indices = seq_indices + first_index_per_sample.unsqueeze(1)  # [B, max_patch_len]

            # Clamp to valid range (shouldn't exceed last_index_per_sample)
            seq_indices = torch.clamp(seq_indices, max=last_index_per_sample.unsqueeze(1))  # [B, max_patch_len]

            # Advanced indexing to extract vision features
            batch_vision_states = layer_hidden[batch_indices, seq_indices, :]  # [B, max_patch_len, H]

            # Mask padding - now based on actual vision patch lengths per sample
            vision_patch_lengths = last_index_per_sample - first_index_per_sample + 1  # [B]
            padding_mask = torch.arange(max_patch_len, device=device).unsqueeze(0) >= vision_patch_lengths.unsqueeze(1)  # [B, max_patch_len]
            batch_vision_states = batch_vision_states.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            
            # ============================================================
            # 2. Action Query Features (Fully Vectorized)
            # ============================================================
            # Use advanced indexing
            # When you index with two tensors in the first two dims, PyTorch treats them as matching coordinates:
            # batch_indices_action is shape [B, N]
            # action_positions_tensor is shape [B, N]
            
            batch_indices_action = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, self.action_query_num)  # [B, N]
            action_query_states = layer_hidden[batch_indices_action, action_positions_tensor, :]  # [B, action_query_num, H]
            
            # ============================================================
            # 3. Concatenate
            # ============================================================
            all_hidden_states = torch.cat([
                batch_vision_states.unsqueeze(1),  # [B, 1, max_patch_len, H]
                action_query_states.unsqueeze(1)   # [B, 1, action_query_num, H]
            ], dim=2)  # [B, 1, L_total, H]
            
            multi_layer_hidden_states.append(all_hidden_states)
            
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)  # [B, num_layers, L_total, H]
        state_projected = None
        if state is not None: # repeat state 
            state = torch.tensor(
                    np.array(state), device=multi_layer_hidden_states.device, dtype=multi_layer_hidden_states.dtype
                ) #  [B, 1, state_dim]
            if self.proprio_projector is not None:
                state_projected = self.proprio_projector(proprio=state.squeeze(1))  # [B, llm_dim]
        
        # ============================================================
        # Action prediction
        # ============================================================
        with torch.autocast("cuda", dtype=torch.float32):
            self.action_model = self.action_model.to(
                device=multi_layer_hidden_states.device, 
                dtype=multi_layer_hidden_states.dtype
            )
            predicted_actions = self.action_model.predict_action(
                multi_layer_hidden_states,
                vision_hidden_len=max_patch_len,
                state_projected=state_projected,
                phase=self.phase,
            )  # (B, chunk_len, action_dim)
        
        normalized_actions = predicted_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions} 


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_train_adapter.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct"
    
    model: Qwen_Adapter = Qwen_Adapter(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 14)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
