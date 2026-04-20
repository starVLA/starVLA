# Copyright 2025 InternVLA-M1. All rights reserved.
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [add fake sample and predict_action to match with starVLA].
"""
InternVLA M1 framework:
Vision-Language-Action diffusion model integrating:
  - Qwen2.5 vision-language backbone
  - Layer-wise QFormer aggregation
  - DINO multi-view visual encoder
  - DiT diffusion head for future action sequence prediction
Primary goal: predict continuous future actions conditioned on multi-view images + instruction.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from qwen_vl_utils import process_vision_info

from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model
from starVLA.model.modules.dino_model.dino import get_dino_model
from starVLA.model.modules.projector.QFormer import get_layerwise_qformer
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images


# ──────────────────────────────────────────────────────────────────────
#  Default Config for InternVLA-M1
#  - Documents every framework-level parameter with type + description
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class InternVLA_M1DefaultConfig:
    """InternVLA-M1 framework default parameters.

    VLM + Layer-wise QFormer + DINO + DiT diffusion head.
    All fields can be overridden by the corresponding key in the YAML
    ``framework:`` section.
    """

    # --- Registry identifier ---
    name: str = "InternVLA-M1"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2",
    })

    # === DINO encoder (multi-view spatial tokens) ===
    dino: dict = field(default_factory=lambda: {
        "dino_backbone": "dinov2_vits14",
    })

    # === Layer-wise QFormer (multi-layer feature aggregation) ===
    layer_qformer: dict = field(default_factory=lambda: {
        # Start layer index for QFormer (inclusive)
        "qformer_start_layer": 20,
        # End layer index for QFormer (exclusive)
        "qformer_end_layer": 36,
        # Number of learnable query tokens (must match DiT's n_conditon_token, default 64)
        "num_query_tokens": 64,
        # Input feature dim (aligned to VLM hidden_size, e.g. Qwen3-VL-4B-Instruct = 2048)
        "input_dim": 2048,
        # Output feature dim (aligned to DiT-B token_size = 768)
        "ouptput_dim": 768,
    })

    # === Action head (DiT diffusion) ===
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "DiT-B",
        "action_dim": 7,
        "future_action_window_size": 15,
        "past_action_window_size": 0,
        "repeated_diffusion_steps": 4,
    })

    # === Observation image size (optional resize before encoding) ===
    obs_image_size: Optional[list] = None


@FRAMEWORK_REGISTRY.register("InternVLA-M1")
class InternVLA_M1(baseframework):
    """
    Multimodal vision-language-action model (M1 variant).

    Components:
      - Qwen2.5-VL / Qwen3-VL backbone for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINOv2 encoder for dense multi-view spatial tokens
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
        # Merge framework defaults with YAML config (YAML wins on conflicts)
        self.config = merge_framework_config(InternVLA_M1DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.layer_qformer = get_layerwise_qformer(config=self.config)
        self.action_model = get_action_model(config=self.config)
        self.dino_encoder = get_dino_model(
            backone_name=getattr(self.config.framework.dino, "dino_backbone", "dinov2_vits14")
        )
        self.dino_pro = nn.Linear(
            in_features=self.dino_encoder.num_channels, out_features=self.qwen_vl_interface.model.config.hidden_size
        )

        self.future_action_window_size = self.config.framework.action_model.future_action_window_size
        self.past_action_window_size = self.config.framework.action_model.past_action_window_size

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass for training (diffusion objective).

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          3. Encode images with DINO, flatten multi-view tokens and project
          4. Concatenate per-layer language tokens with visual tokens
          5. Fuse via layer-wise QFormer -> action condition embeddings
          6. Prepare repeated future action windows (for diffusion efficiency)
          7. Predict noise and compute diffusion loss

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
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            pass

        # Step 2: DINO Forward
        image_tensors = self.dino_encoder.prepare_dino_input(batch_images)  #
        B = len(batch_images)
        dino_features = self.dino_encoder(image_tensors)  # DINO output is [B*num_view, token, dim]
        dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])  # [B, num_view * token, dim]
        dino_encoded_features = self.dino_pro(dino_encoded_features)  # [B, num_view * token, hidden_size]

        # Step 3: aggregation condition for Action expert
        start_layer = self.config.framework.layer_qformer.qformer_start_layer
        end_layer = self.config.framework.layer_qformer.qformer_end_layer
        condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]

        cat_conditions = []
        for layer_index in range(len(condition_features)):
            layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
            layer_features = torch.cat(
                [layer_features, dino_encoded_features], dim=1
            )  # [B, n_qformer_token + num_view * token, D]
            cat_conditions.append(layer_features)

        action_condition = self.layer_qformer(cat_conditions)  # [B, 64, D_action]

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):

            # here is a tips to accelerate training speed, by repeating each sample for several times @ref to CogACT
            actions = torch.tensor(np.array(actions), device=action_condition.device)  # [B, chunk, 7]
            actions_future = actions[:, -(self.future_action_window_size + 1) :, :]

            # tips: Repeat 'actions' 'repeated_diffusion_steps' times, resulting in [repeated_diffusion_steps*B, T, D]
            repeated_diffusion_steps = (
                self.config.framework.action_model.get("repeated_diffusion_steps", 4)
                if self.config and hasattr(self.config, "framework")
                else 4
            )
            actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
            action_condition = action_condition.repeat(
                repeated_diffusion_steps, 1, 1
            )  # [repeated_diffusion_steps*B, T, D_action]

            # DiT noise add and predict
            noise_pred, noise, timestep = self.action_model(actions_repeated, action_condition)

            # perdition loss
            action_loss = self.action_model.loss(noise_pred, noise)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 5,
        **kwargs: str,
    ) -> dict:
        """
        Inference: generate future normalized action sequence via diffusion sampling.

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          3. Extract DINO tokens and project to vlm hidden size
          4. Build multi-layer fused QwenVL and DINO features via QFormer
          5. Run diffusion sampling (DDIM optional, CFG optional)
          6. Return normalized action trajectory

        Args:
            examples: List[dict], each dict has "image" (PIL.Image or list) and "lang" (str).
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [example["image"] if isinstance(example["image"], list) else [example["image"]] for example in examples]
        instructions = [example["lang"] for example in examples]

        # align obs and lang
        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
        instructions = [instruction.lower() for instruction in instructions]

        inferface_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        qwen_inputs = inferface_inputs

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_hidden_states=True,
                return_dict=True,
            )

            B = len(batch_images)  # dino don't have smart resize in processing
            image_tensors = self.dino_encoder.prepare_dino_input(batch_images)
            dino_features = self.dino_encoder(image_tensors)

            B = len(batch_images)
            dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])  # [B, num_view * token, dim]
            dino_encoded_features = self.dino_pro(dino_encoded_features)  # [B, 256, D]

        with torch.autocast("cuda", dtype=torch.bfloat16):

            start_layer = self.config.framework.layer_qformer.qformer_start_layer
            end_layer = self.config.framework.layer_qformer.qformer_end_layer
            condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]
            cat_conditions = []
            for layer_index in range(len(condition_features)):
                layer_features = condition_features[layer_index]  # [B, n_qformer_token, D]
                layer_features = torch.cat(
                    [layer_features, dino_encoded_features], dim=1
                )  # [B, n_qformer_token + num_view * token, D]
                cat_conditions.append(layer_features)

            action_condition_feature = self.layer_qformer(cat_conditions)  # [B, 64, D_action]

            using_cfg = cfg_scale > 1.0

            model_dtype = next(self.action_model.net.parameters()).dtype
            B = action_condition_feature.shape[0]

            # Sample random noise
            noise = torch.randn(
                B,
                self.future_action_window_size + 1,
                self.action_model.in_channels,
                device=action_condition_feature.device,
            ).to(
                model_dtype
            )  # [B, T, D]

            # Setup classifier-free guidance:
            if using_cfg:
                noise = torch.cat([noise, noise], 0)  # [2,16,7]
                uncondition = self.action_model.net.z_embedder.uncondition  # [64, 768]
                uncondition_shape = uncondition.shape
                uncondition = uncondition.unsqueeze(0)  # [1, 64, D]
                uncondition = uncondition.expand(
                    B, uncondition_shape[0], uncondition_shape[1]
                )  # [B, n_qformer_token, D]
                z = torch.cat([action_condition_feature, uncondition], 0)  # [2, 64, 768]
                cfg_scale = cfg_scale
                model_kwargs = dict(z=z, cfg_scale=cfg_scale)
                sample_fn = self.action_model.net.forward_with_cfg
            else:
                model_kwargs = dict(z=action_condition_feature)
                sample_fn = self.action_model.net.forward

            # DDIM Sampling
            if use_ddim and num_ddim_steps is not None:
                if self.action_model.ddim_diffusion is None:
                    self.action_model.create_ddim(ddim_step=num_ddim_steps)
                samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                    sample_fn,
                    noise.shape,
                    noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=action_condition_feature.device,
                    eta=0.0,
                )

            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            normalized_actions = samples.cpu().numpy()

            raw_actions = None

        return {"normalized_actions": normalized_actions}  # [B, T, action_dim]

    @torch.inference_mode()
    def chat_with_M1(
        self,
        image: Image.Image,
        text: str,
        max_new_tokens: int = 128,
        device: Optional[str] = "cuda",
    ) -> List[str]:
        processor = getattr(self.qwen_vl_interface, "processor", None)
        model = getattr(self.qwen_vl_interface, "model", None)
        # if processor is None or model is None:
        #     raise RuntimeError("qwen_vl_interface is missing processor or model.")

        messages0 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        messages = [messages0]
        # text info
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        # visual info
        image_inputs, video_inputs = process_vision_info(messages)

        # tokenizer
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        model.eval()
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_libero.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    model = InternVLA_M1(cfg)
    print(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image, image],
        "lang": "This is a fake instruction for testing.",
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."

    batch = [sample, sample2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    predict_output = model.predict_action(examples=[batch[0]])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    print("Finished")
