import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))
import os

CHECKPOINT_BASEDIR = os.getenv("CHECKPOINT_BASEDIR", None)
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.AML_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY, CrossAttention, preprocess_images
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for ABot_M0
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ABot_M0DefaultConfig:
    """ABot_M0 framework default parameters.

    Qwen-VL + VGGT spatial geometry backbone + Action Manifold Learning
    flow-matching head.  All fields can be overridden by the corresponding
    key in the YAML ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "ABot_M0"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
    })

    # === Action head (AML Flow-matching) ===
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_hidden_dim": 1024,
        "hidden_size": 1024,
        "add_pos_embed": True,
        "max_seq_len": 1024,
        "action_dim": 14,
        "state_dim": 14,
        "future_action_window_size": 15,
        "action_horizon": 16,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 4,
        "num_inference_timesteps": 4,
        "diffusion_model_cfg": {
            "cross_attention_dim": 2048,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 16,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
    })

    # === Observation image size (optional resize before encoding) ===
    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("ABot_M0")
class ABot_M0(baseframework):
    """
    ABot-M0: Vision-Language-Action model with spatial geometry.

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - VGGT spatial geometry backbone for 3D scene understanding
      - Action Manifold Learning flow-matching head for large action-dim prediction

    Focus: Predict future continuous actions conditioned on images + instruction,
    significantly outperforms GR00T on large action_dim * action_chunk.
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
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(ABot_M0DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align cross_attention_dim to VLM hidden_size at runtime
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
            self.qwen_vl_interface.model.config.hidden_size
        )

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # optional, TODO: pip install -e path_to_vggt (https://github.com/facebookresearch/vggt)

        # from vggt.models.vggt import VGGT
        self.spatial_model = spatial_model = VGGT.from_pretrained("facebook/VGGT-1B")
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.spatial_projector = nn.Linear(2048, hidden_size)
        self.fuser = CrossAttention(d_model=hidden_size, d_hidden=hidden_size, kv_dim=hidden_size)

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:

        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        action_mask = (
            [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None
        )  # [B, action_dim]

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
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            # feed forward pass of vggt
            with torch.no_grad():
                spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(
                    qwen_inputs["pixel_values"].device
                )
                aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
            spatial_tokens = aggregated_tokens_list[-1][:, 0, ps_idx:, :]
            spatial_tokens = self.spatial_projector(spatial_tokens)
            last_hidden = self.fuser(last_hidden, spatial_tokens)

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=last_hidden.device, dtype=last_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_mask_repeated = None
            if action_mask is not None:
                action_mask_tensor = torch.tensor(
                    np.array(action_mask), device=last_hidden.device, dtype=torch.bool
                )  # [B, action_dim]
                action_mask_repeated = action_mask_tensor.repeat(
                    repeated_diffusion_steps, 1
                )  # [B*repeated_diffusion_steps, action_dim]

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated, action_mask=action_mask_repeated
            )  # (B, chunk_len, action_dim)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
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

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

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
            last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, H]

            # feed forward pass of vggt
            with torch.no_grad():
                spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(
                    qwen_inputs["pixel_values"].device
                )
                aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
            spatial_tokens = aggregated_tokens_list[-1][:, 0, ps_idx:, :]
            spatial_tokens = self.spatial_projector(spatial_tokens)
            last_hidden = self.fuser(last_hidden, spatial_tokens)

        state = (
            torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype)
            if state is not None
            else None
        )

        # Step 4: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = f"{CHECKPOINT_BASEDIR}/Florence-2-large"

    model: ABot_M0 = ABot_M0(cfg)
    print(model)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16),  # action_chunk, action_dim
        "image": [image],  # three views
        "lang": (
            "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room."
        ),
        "state": np.random.uniform(-1, 1, size=(1, 14)).astype(np.float16),  # chunk, state_dim
    }

    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample])  # , state=[batch[0]["state"]]
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # vla_dataset_cfg = cfg.datasets.vla_data
    # from torch.utils.data import DataLoader
    # from starVLA.dataloader.lerobot_datasets import collate_fn, get_vla_dataset
    # cfg.datasets.vla_data.include_state = "False"
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    # train_dataloader = DataLoader(dataset, batch_size=2, num_workers=1, collate_fn=collate_fn)
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)
    # action = model.predict_action(examples=batch)
    print("Finished")
