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


from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.VLA_AdapterHeader import get_action_model, L1RegressionActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

def get_image_token_counts(batch_inputs):
    IMAGE_TOKEN_ID = 151655 
    
    # input_ids shape: [Batch_Size, Seq_Len]
    # result shape: [Batch_Size]
    num_tokens_per_sample = torch.sum(batch_inputs['input_ids'] == IMAGE_TOKEN_ID, dim=1)
    
    return num_tokens_per_sample


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
        self.use_proprio = self.config.framework.action_model.get("use_proprio", False)
        self.proprio_projector = ProprioProjector(
            llm_dim=self.qwen_vl_interface.model.config.hidden_size,
            proprio_dim=self.config.framework.action_model.get("state_dim", 0),
        ) if self.use_proprio else None
        self.phase = self.config.framework.action_model.get("phase", "Training")
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.qwenvl.vl_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
        self.action_query_num = self.config.framework.action_model.get("action_query_num", 64)
        self.action_model: L1RegressionActionHead = get_action_model(config=self.config)
        self.action_query = nn.Parameter(torch.randn(self.action_query_num, self.qwen_vl_interface.model.config.hidden_size))
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
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        num_patches = get_image_token_counts(qwen_inputs) # [B, ]

        batch_size = qwen_inputs['input_ids'].shape[0]
        pad_id = self.qwen_vl_interface.model.config.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.qwen_vl_interface.model.config, "eos_token_id", 0)
            if pad_id is None:
                pad_id = 0
        device = qwen_inputs['input_ids'].device

        dummy_ids = torch.full(
            (batch_size, self.action_query_num),pad_id, device=device, dtype=qwen_inputs['input_ids'].dtype
        )  # (B, action_query_num)
        dummy_mask = torch.ones(
            (batch_size, self.action_query_num), device=device, dtype=qwen_inputs['attention_mask'].dtype
        )  # (B, action_query_num)

        qwen_inputs['input_ids'] = torch.cat([qwen_inputs['input_ids'], dummy_ids], dim=1)  # (B, L + action_query_num)
        qwen_inputs['attention_mask'] = torch.cat([qwen_inputs['attention_mask'], dummy_mask], dim=1)  # (B, L + action_query_num)

        def inject_query_hook(module, inputs, output):
            query_embed = self.action_query.to(dtype=output.dtype, device=output.device)  # (action_query_num, hidden_dim)
            batch_query = query_embed.unsqueeze(0).expand(output.shape[0], -1, -1)  # (B, action_query_num, hidden_dim)
            output[:, -self.action_query_num:, :] = batch_query
            return output

        embedding_layer = self.qwen_vl_interface.model.model.get_input_embeddings()
        assert isinstance(embedding_layer, nn.Embedding), "Cannot find qwenvl embedding layer"
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
                

        multi_layer_hidden_states = []

        for item in qwenvl_outputs.hidden_states[0:]:
            batch_size = item.shape[0]
            hidden_dim = item.shape[-1]

            max_patch_len = num_patches.max().item()

            batch_vision_states = []
            batch_action_query_states = []

            for i in range(batch_size):
                n_p = num_patches[i].item()
                vis_feat = item[i, :n_p, :]  # [n_p, D]
                if n_p < max_patch_len:
                    pad_len = max_patch_len - n_p
                    padding = torch.zeros((pad_len, hidden_dim), device=vis_feat.device, dtype=vis_feat.dtype)
                    vis_feat = torch.cat([vis_feat, padding], dim=0)  # [max_patch_len, D]

                batch_vision_states.append(vis_feat)
                
                act_query_feat = item[i, -self.action_query_num:, :]  # [action_query_num, D]
                batch_action_query_states.append(act_query_feat)
            
            vision_hidden_states = torch.stack(batch_vision_states).unsqueeze(1)  # [B, 1, max_patch_len, D]
            action_query_hidden_states = torch.stack(batch_action_query_states).unsqueeze(1)  # [B, 1, action_query_num, D]

            all_hidden_states = torch.cat([vision_hidden_states, action_query_hidden_states], dim=2)  # [B, 1, max_patch_len + action_query_num, D]
            multi_layer_hidden_states.append(all_hidden_states)  # [num_layers][B, 1, L_total, D]

        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)  # [B, num_layers, L_total, D]


        # Step 3: Action Expert Forward
        self.action_model = self.action_model.to(device=multi_layer_hidden_states.device, dtype=multi_layer_hidden_states.dtype)
        predicted_actions = self.action_model.predict_action(
            multi_layer_hidden_states,
            proprio=state if self.use_proprio else None,
            proprio_projector=self.proprio_projector if self.use_proprio else None,
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
        batch_images: List[List[Image.Image]], 
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: Predict future continuous actions aligned with the Forward logic (Hook + Multi-layer states).
        """
        # Step 0: Preprocessing (Resize)
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            # Assuming resize_images is a valid helper function available in context
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        num_patches = get_image_token_counts(qwen_inputs) # [B, ]

        # Step 2: Prepare Dummy Tokens & Hook for Action Query Injection
        batch_size = qwen_inputs['input_ids'].shape[0]
        pad_id = self.qwen_vl_interface.model.config.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.qwen_vl_interface.model.config, "eos_token_id", 0)
            if pad_id is None:
                pad_id = 0
        device = qwen_inputs['input_ids'].device

        # Create dummy placeholders
        dummy_ids = torch.full(
            (batch_size, self.action_query_num), pad_id, device=device, dtype=qwen_inputs['input_ids'].dtype
        )
        dummy_mask = torch.ones(
            (batch_size, self.action_query_num), device=device, dtype=qwen_inputs['attention_mask'].dtype
        )

        # Concat to inputs
        qwen_inputs['input_ids'] = torch.cat([qwen_inputs['input_ids'], dummy_ids], dim=1)
        qwen_inputs['attention_mask'] = torch.cat([qwen_inputs['attention_mask'], dummy_mask], dim=1)

        # Define Hook
        def inject_query_hook(module, inputs, output):
            query_embed = self.action_query.to(dtype=output.dtype, device=output.device)
            batch_query = query_embed.unsqueeze(0).expand(output.shape[0], -1, -1)
            output[:, -self.action_query_num:, :] = batch_query
            return output

        # Register Hook
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

        # Step 3: Process Multi-layer Hidden States
        # Logic copied from forward: Concatenate Vision Tokens + Action Query Tokens across layers
        multi_layer_hidden_states = []

        for item in qwenvl_outputs.hidden_states[0:]:
            # item shape: [B, Seq_Len, H]
            hidden_dim = item.shape[-1]
            max_patch_len = num_patches.max().item()

            batch_vision_states = []
            batch_action_query_states = []

            for i in range(batch_size):
                n_p = num_patches[i].item()
                # Extract vision features
                vis_feat = item[i, :n_p, :]  # [n_p, D]
                
                # Handle padding for vision tokens if batch has varying aspect ratios
                if n_p < max_patch_len:
                    pad_len = max_patch_len - n_p
                    padding = torch.zeros((pad_len, hidden_dim), device=vis_feat.device, dtype=vis_feat.dtype)
                    vis_feat = torch.cat([vis_feat, padding], dim=0)  # [max_patch_len, D]

                batch_vision_states.append(vis_feat)
                
                # Extract action query features (last N tokens)
                act_query_feat = item[i, -self.action_query_num:, :]  # [action_query_num, D]
                batch_action_query_states.append(act_query_feat)
            
            vision_hidden_states = torch.stack(batch_vision_states).unsqueeze(1)  # [B, 1, max_patch_len, D]
            action_query_hidden_states = torch.stack(batch_action_query_states).unsqueeze(1)  # [B, 1, action_query_num, D]

            all_hidden_states = torch.cat([vision_hidden_states, action_query_hidden_states], dim=2)
            multi_layer_hidden_states.append(all_hidden_states)

        # [B, num_layers, L_total, D]
        multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)

        # Step 4: Proprioception Processing
        proprio_tensor = None
        if self.use_proprio and state is not None:
            # Convert numpy state to tensor, match device/dtype of hidden states
            proprio_tensor = torch.from_numpy(np.array(state)).to(
                device=multi_layer_hidden_states.device, 
                dtype=multi_layer_hidden_states.dtype
            )
            # Ensure shape is [B, proprio_dim] or [B, 1, proprio_dim] depending on projector expectation
            if proprio_tensor.ndim == 1:
                proprio_tensor = proprio_tensor.unsqueeze(0) # Batch dim
            if proprio_tensor.ndim == 3:
                proprio_tensor = proprio_tensor.squeeze(1) # Remove seq dim if present for projector

        # Step 5: Action Expert Prediction
        # Handle DDP wrapper if present, otherwise call model directly
        action_model_ref = self.action_model.module if hasattr(self.action_model, "module") else self.action_model
        
        # Ensure model is on correct device/dtype
        action_model_ref = action_model_ref.to(device=multi_layer_hidden_states.device, dtype=multi_layer_hidden_states.dtype)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = action_model_ref.predict_action(
                multi_layer_hidden_states,
                proprio=proprio_tensor,
                proprio_projector=self.proprio_projector if self.use_proprio else None,
                phase="Inference", # Explicitly set phase for inference logic
            )  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
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
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
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
