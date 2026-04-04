# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Wan2.2-TI2V World Model Interface.

Wraps Wan-AI/Wan2.2-TI2V-5B (diffusion-based Text+Image-to-Video model) as a
world-model backend for starVLA action prediction frameworks.

Architecture:
  - UMT5EncoderModel: text instruction → text embeddings [B, L_text, 4096]
  - CLIPVisionModel: observation image → image embeddings [B, N_patches, 1280]
  - AutoencoderKLWan (VAE): observation image → video latents [B, C, T, H, W]
  - WanTransformer3DModel: 30-layer DiT, hidden_dim=3072 (24 heads × 128 dim)
    Takes noised latents + text/image embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Key differences from CosmoPredict2:
  - Text encoder: UMT5 (dim=4096) vs T5 (dim=1024)
  - Image conditioning: CLIP embeddings + VAE latents (dual path)
  - DiT hidden dim: 3072 (24×128) vs 4096 (32×128)
  - Wan2.2 TI2V uses expand_timesteps with first_frame_mask for image conditioning

Key differences from VLM wrappers:
  - No chat template / processor — uses UMT5 for text, CLIP+VAE for vision
  - Hidden states come from DiT blocks, not autoregressive LM
"""

from typing import Optional

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _Wan2_Interface(nn.Module):
    """
    World model wrapper for Wan2.2-TI2V-5B (diffusers-based).

    The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      collected into a [B, N_tokens, hidden_dim] tensor that the action
      head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "Wan-AI/Wan2.2-TI2V-5B"),
        )
        self.config = config

        from diffusers import (
            AutoencoderKLWan,
            FlowMatchEulerDiscreteScheduler,
            WanTransformer3DModel,
        )
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPVisionModel,
            UMT5EncoderModel,
        )

        logger.info(f"Loading Wan2.2-TI2V from {model_name}")

        # --- Text encoder: UMT5-XXL ---
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )

        # --- Image encoder: CLIP (for cross-attention conditioning) ---
        self.image_processor = CLIPImageProcessor.from_pretrained(
            model_name, subfolder="image_encoder"
        )
        self.image_encoder = CLIPVisionModel.from_pretrained(
            model_name, subfolder="image_encoder", torch_dtype=torch.bfloat16
        )

        # --- DiT transformer ---
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )

        # --- VAE (image → latents for DiT input) ---
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )

        # --- Scheduler ---
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Freeze VAE, text encoder, image encoder by default
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)

        # DiT: 24 heads × 128 dim = 3072
        self._hidden_size = (
            self.transformer.config.num_attention_heads
            * self.transformer.config.attention_head_dim
        )

        # Config-like shim for framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._hooks = []

        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._register_hooks()

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.backbone.model.config.hidden_size"""

        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _register_hooks(self):
        """Register forward hooks on selected transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output."""
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using UMT5."""
        device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        mask = text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=mask,
            ).last_hidden_state  # [B, L, 4096]

        # Trim to actual lengths then re-pad (matches Wan pipeline)
        text_embeds = text_embeds.to(dtype=torch.bfloat16)
        trimmed = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_length - u.size(0), u.size(1))])
                for u in trimmed
            ],
            dim=0,
        )
        return text_embeds  # [B, max_length, 4096]

    def _encode_image_clip(self, images):
        """Encode images through CLIP for cross-attention conditioning.

        Args:
            images: List of List of PIL Images [B, [imgs...]]

        Returns:
            image_embeds: [B, N_patches, 1280]
        """
        device = next(self.image_encoder.parameters()).device

        # Take the last image from each sample
        pil_images = []
        for sample_imgs in images:
            if isinstance(sample_imgs, (list, tuple)):
                pil_images.append(sample_imgs[-1])
            else:
                pil_images.append(sample_imgs)

        pixel_values = self.image_processor(
            images=pil_images, return_tensors="pt"
        ).pixel_values.to(device, dtype=torch.bfloat16)

        with torch.no_grad():
            image_embeds = self.image_encoder(
                pixel_values, output_hidden_states=True
            ).hidden_states[-2]  # [B, N_patches, 1280]

        return image_embeds

    def _encode_images_vae(self, images):
        """Encode observation images through VAE to get latent tokens.

        Args:
            images: List of List of PIL Images [B, [imgs...]]

        Returns:
            latents: [B, C, T=1, H', W'] video latent tensor
        """
        import torchvision.transforms.functional as TF

        device = next(self.vae.parameters()).device

        frames = []
        for sample_imgs in images:
            if isinstance(sample_imgs, (list, tuple)):
                img = sample_imgs[-1]
            else:
                img = sample_imgs
            # PIL → tensor [C, H, W] in [-1, 1]
            t = TF.to_tensor(img).to(device, dtype=torch.bfloat16) * 2.0 - 1.0
            # Wan expects height/width multiples of (vae_spatial=8 * patch_h=2) = 16
            t = TF.resize(t, [480, 832])
            frames.append(t)

        # [B, C, H, W] → [B, C, T=1, H, W]
        pixel_values = torch.stack(frames, dim=0).unsqueeze(2)

        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()

        # Normalize latents (Wan convention)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = (latents - latents_mean) * latents_std

        return latents

    def build_inputs(self, images, instructions, **kwargs):
        """Build inputs for the Wan DiT world model.

        Encoding pipeline:
        1. Text → UMT5 → text embeddings [B, L, 4096]
        2. Image → CLIP → image embeddings [B, N, 1280] (cross-attn conditioning)
        3. Image → VAE → latents [B, C, T, H', W'] (DiT input)

        Returns:
            dict with keys matching forward() expectations
        """
        assert len(images) == len(instructions)

        text_embeds = self._encode_text(instructions)
        image_embeds = self._encode_image_clip(images)
        latents = self._encode_images_vae(images)

        batch_size = latents.shape[0]
        device = latents.device

        # Wan2.2 TI2V uses expand_timesteps: timestep is per-token
        # For feature extraction at σ≈0, use zeros
        # Shape: [B, seq_len] where seq_len = num_latent_frames * (H'//p_h) * (W'//p_w)
        p_t, p_h, p_w = self.transformer.config.patch_size
        _, _, T, H, W = latents.shape
        seq_len = (T // p_t) * (H // p_h) * (W // p_w)
        timestep = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "encoder_hidden_states_image": image_embeds,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        """Forward pass through the Wan DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        Returns an output object with .hidden_states for compatibility.
        """
        kwargs.pop("_is_wm_input", False)
        kwargs.pop("output_hidden_states", False)
        kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)

        self._intermediate_features.clear()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
                encoder_hidden_states_image=kwargs.get("encoder_hidden_states_image"),
            )

        # Collect features from hooks
        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                # [B, C, T, H, W] -> [B, T*H*W, C]
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)

        # Fallback: use transformer output directly
        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss

        return _WMOutput(hidden_states=tuple(extracted))

    def generate(self, **kwargs):
        """Video generation using the full WanImageToVideoPipeline.

        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        """
        from diffusers import WanImageToVideoPipeline

        pipe = WanImageToVideoPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            image_processor=self.image_processor,
            image_encoder=self.image_encoder,
            vae=self.vae,
            transformer=self.transformer,
            scheduler=self.scheduler,
        )
        return pipe(**kwargs)
