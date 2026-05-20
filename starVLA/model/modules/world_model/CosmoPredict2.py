# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2 World Model Interface.

Wraps NVIDIA Cosmos-Predict2 (diffusion-based Video2World model) as a
world-model backend for starVLA action prediction frameworks.

Architecture:
  - T5EncoderModel: text instruction → text embeddings [B, L_text, 1024]
  - AutoencoderKLWan (VAE): observation images → video latents [B, C, T, H, W]
  - CosmosTransformer3DModel (DiT): 28-layer transformer, hidden_dim=4096
    Takes noised latents + text embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Key difference from VLM wrappers:
  - No chat template / processor — uses T5 for text, VAE for vision
  - Hidden states come from DiT blocks, not autoregressive LM
  - The `build_inputs` interface provides a clean world-model API
    that does not depend on VLM-specific naming conventions.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _CosmoPredict2_Interface(nn.Module):
    """
    World model wrapper for Cosmos-Predict2 (diffusers-based).

    Exposes a compatible interface with VLM wrappers so that framework
    code can swap VLM ↔ WM transparently. The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      concatenated/pooled to produce a [B, N_tokens, hidden_dim] tensor
      that the action head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "nvidia/Cosmos-Predict2-2B-Video2World"),
        )
        self.config = config
        self._base_wm_path = model_name

        # Import diffusers components
        from diffusers import (
            AutoencoderKLWan,
            CosmosTransformer3DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import T5EncoderModel, T5TokenizerFast

        logger.info(f"Loading Cosmos-Predict2 from {model_name}")

        # Load components individually (Pipeline is not nn.Module; split loading enables per-component freeze/finetune)
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        transformer_checkpoint = wm_cfg.get("transformer_checkpoint", None)
        if transformer_checkpoint:
            self._load_transformer_checkpoint(transformer_checkpoint)
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Use diffusers' VideoProcessor for image/video preprocessing (resize, normalize, etc.)
        from diffusers.video_processor import VideoProcessor
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Freeze VAE and text encoder by default
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Expose config compatible with framework expectations
        # DiT: 16 heads × 128 dim = 2048
        self._hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        # Create a config-like object for the framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._hooks = []

        # Which transformer blocks to extract features from (-1 = last)
        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._multiview_mode = wm_cfg.get("multiview_mode", "horizontal_concat")
        self._register_hooks()

    def _load_transformer_checkpoint(self, checkpoint_path: str):
        """Load a standalone Cosmos transformer checkpoint after from_pretrained.

        ``base_wm`` must still point to a valid Cosmos-Predict2 diffusers
        directory because tokenizer/text_encoder/vae/scheduler/config are
        loaded from there.  This optional file only overrides transformer
        weights.
        """
        logger.info(f"Loading Cosmos transformer checkpoint from {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        def _looks_like_state_dict(candidate):
            return isinstance(candidate, dict) and any(torch.is_tensor(v) for v in candidate.values())

        if isinstance(state_dict, dict) and not _looks_like_state_dict(state_dict):
            for key in ("state_dict", "model", "module", "model_state_dict", "ema", "model_ema"):
                maybe_state = state_dict.get(key, None)
                if _looks_like_state_dict(maybe_state):
                    state_dict = maybe_state
                    break

        if not _looks_like_state_dict(state_dict):
            raise ValueError(f"Unsupported Cosmos transformer checkpoint format: {checkpoint_path}")

        prefixes = (
            "module._orig_mod.cosmos_backbone.transformer.",
            "module._orig_mod.backbone.transformer.",
            "module._orig_mod.world_model.transformer.",
            "module._orig_mod.net.",
            "_orig_mod.cosmos_backbone.transformer.",
            "_orig_mod.backbone.transformer.",
            "_orig_mod.world_model.transformer.",
            "_orig_mod.net.",
            "model.cosmos_backbone.transformer.",
            "model.backbone.transformer.",
            "model.world_model.transformer.",
            "model.net.",
            "module.cosmos_backbone.transformer.",
            "module.backbone.transformer.",
            "module.world_model.transformer.",
            "module.net.",
            "cosmos_backbone.transformer.",
            "backbone.transformer.",
            "world_model.transformer.",
            "model.transformer.",
            "module.transformer.",
            "transformer.",
            "net.",
        )
        target_state = self.transformer.state_dict()
        target_keys = set(target_state.keys())
        normalized = {}
        ignored_keys = []
        shape_mismatch = []

        for key, value in state_dict.items():
            normalized_key = key
            for prefix in prefixes:
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
                    break
            if normalized_key not in target_keys and ".transformer." in normalized_key:
                normalized_key = normalized_key.rsplit(".transformer.", maxsplit=1)[-1]

            if normalized_key not in target_keys:
                ignored_keys.append(key)
                continue
            if tuple(value.shape) != tuple(target_state[normalized_key].shape):
                shape_mismatch.append((key, normalized_key, tuple(value.shape), tuple(target_state[normalized_key].shape)))
                continue
            normalized[normalized_key] = value

        if not normalized and self._looks_like_original_cosmos_checkpoint(state_dict):
            if self._load_transformer_from_single_file(checkpoint_path):
                return

        if not normalized:
            sample_keys = list(state_dict.keys())[:8]
            raise ValueError(
                "No compatible Cosmos transformer weights found in checkpoint "
                f"{checkpoint_path}. Sample checkpoint keys: {sample_keys}. "
                f"Sample target keys: {list(target_keys)[:8]}"
            )

        missing, unexpected = self.transformer.load_state_dict(normalized, strict=False)
        logger.info(
            "Loaded Cosmos transformer checkpoint: "
            f"loaded={len(normalized)}, missing={len(missing)}, "
            f"ignored={len(ignored_keys)}, shape_mismatch={len(shape_mismatch)}, "
            f"unexpected_after_filter={len(unexpected)}"
        )
        if ignored_keys:
            logger.warning(f"Ignored Cosmos checkpoint keys sample: {ignored_keys[:8]}")
        if shape_mismatch:
            logger.warning(f"Shape-mismatched Cosmos checkpoint keys sample: {shape_mismatch[:4]}")

    @staticmethod
    def _looks_like_original_cosmos_checkpoint(state_dict):
        keys = state_dict.keys()
        return (
            "net.x_embedder.proj.1.weight" in state_dict
            and any(key.startswith("net.blocks.") for key in keys)
        )

    def _load_transformer_from_single_file(self, checkpoint_path: str) -> bool:
        """Use diffusers' original-format Cosmos converter when key names differ."""
        from_single_file = getattr(self.transformer.__class__, "from_single_file", None)
        if from_single_file is None:
            logger.warning(
                "Cosmos checkpoint is original-format, but this diffusers version "
                "does not expose CosmosTransformer3DModel.from_single_file()."
            )
            return False

        dtype = next(self.transformer.parameters()).dtype
        load_attempts = (
            {
                "config": self._base_wm_path,
                "subfolder": "transformer",
                "torch_dtype": dtype,
                "low_cpu_mem_usage": False,
            },
            {
                "config": self._base_wm_path,
                "subfolder": "transformer",
                "torch_dtype": dtype,
            },
            {"torch_dtype": dtype, "low_cpu_mem_usage": False},
            {"torch_dtype": dtype},
        )
        last_error = None
        for kwargs in load_attempts:
            try:
                logger.info(
                    "Loading original-format Cosmos transformer checkpoint via "
                    f"from_single_file with kwargs={list(kwargs.keys())}"
                )
                converted = self.transformer.__class__.from_single_file(checkpoint_path, **kwargs)
                meta_tensors = [
                    name
                    for name, tensor in list(converted.named_parameters()) + list(converted.named_buffers())
                    if getattr(tensor, "is_meta", False)
                ]
                if meta_tensors:
                    raise RuntimeError(f"Converted Cosmos transformer still has meta tensors: {meta_tensors[:8]}")
                self.transformer = converted
                logger.info("Loaded original-format Cosmos transformer checkpoint via from_single_file.")
                return True
            except Exception as exc:
                last_error = exc
                logger.warning(f"Cosmos from_single_file attempt failed with kwargs={list(kwargs.keys())}: {exc}")

        logger.warning(f"All Cosmos from_single_file attempts failed: {last_error}")
        return False

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.qwen_vl_interface.model.config.hidden_size"""
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

        num_blocks = len(self.transformer.transformer_blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.transformer_blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output."""
        # DiT block output is a tuple; first element is hidden_states
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using T5."""
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # [B, L, 1024]

        return text_embeds, text_inputs.attention_mask

    @staticmethod
    def _to_pil_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = image * 255 if np.nanmax(image) <= 1.0 else image
                image = np.clip(image, 0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        if torch.is_tensor(image):
            array = image.detach().cpu()
            if array.ndim == 3 and array.shape[0] in {1, 3}:
                array = array.permute(1, 2, 0)
            array = array.numpy()
            if array.dtype != np.uint8:
                array = array * 255 if np.nanmax(array) <= 1.0 else array
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array).convert("RGB")
        raise TypeError(f"Unsupported image type for Cosmos multiview concat: {type(image)}")

    @classmethod
    def _concat_multiview_images(cls, sample_imgs):
        """Convert multiple camera views into one horizontal image frame.

        Cosmos-Predict2 is a video world model. CALVIN multiview samples are
        simultaneous camera views, not temporal frames, so we must not feed
        ``[static, wrist]`` as a two-frame video.  Instead, stitch views
        left-to-right and let Cosmos see a single observation frame.
        """
        if not isinstance(sample_imgs, (list, tuple)):
            return [cls._to_pil_image(sample_imgs)]
        if len(sample_imgs) == 0:
            raise ValueError("Expected at least one image for Cosmos input.")
        if len(sample_imgs) == 1:
            return [cls._to_pil_image(sample_imgs[0])]

        pil_images = [cls._to_pil_image(img) for img in sample_imgs]
        target_height = max(img.height for img in pil_images)
        resized = []
        for img in pil_images:
            if img.height != target_height:
                new_width = max(1, round(img.width * target_height / img.height))
                img = img.resize((new_width, target_height), Image.BICUBIC)
            resized.append(img)

        total_width = sum(img.width for img in resized)
        canvas = Image.new("RGB", (total_width, target_height))
        x_offset = 0
        for img in resized:
            canvas.paste(img, (x_offset, 0))
            x_offset += img.width
        return [canvas]

    def _prepare_sample_images(self, sample_imgs):
        if self._multiview_mode == "horizontal_concat":
            return self._concat_multiview_images(sample_imgs)
        if not isinstance(sample_imgs, (list, tuple)):
            return [self._to_pil_image(sample_imgs)]
        return [self._to_pil_image(img) for img in sample_imgs]

    def _encode_images(self, images, num_frames=None):
        """Encode observation images through VAE to get latent tokens.

        For CALVIN multiview, simultaneous camera views are first stitched
        horizontally into one frame.  We only keep temporal-frame semantics
        when ``world_model.multiview_mode`` is explicitly set away from
        ``horizontal_concat``.

        Args:
            images: List of List of PIL Images [B, [imgs...]]
            num_frames: If given, pad/truncate to this exact count.
                If None (default), pad to the max frame count in the batch.
                VAE temporal factor is 4, so T_latent = (num_frames-1)//4+1.

        Returns:
            latents: [B, C, T_latent, H/8, W/8] video latent tensor
            cond_frame_counts: list[int], real frame count per sample (before padding)
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        # 480×832 is the pretrained resolution; spatial dims must be multiples of 16.
        # Smaller sizes (e.g. 224×224) technically work but hurt quality due to positional embedding mismatch.
        # If saving VRAM, keep ~16:9 aspect ratio: 256×448 or 320×576.
        height, width = 320, 576

        # First pass: preprocess each sample, record real frame counts
        preprocessed = []
        cond_frame_counts = []
        for sample_imgs in images:
            sample_imgs = self._prepare_sample_images(sample_imgs)

            video_tensor = self.video_processor.preprocess_video(sample_imgs, height=height, width=width)
            video_tensor = video_tensor.to(device=device, dtype=dtype)  # [1, C, n_imgs, H, W]
            preprocessed.append(video_tensor)
            cond_frame_counts.append(video_tensor.shape[2])

        # Determine target frame count: use num_frames if specified, otherwise batch max
        # Ensure at least 1 frame (VAE needs T >= 1)
        if num_frames is None:
            target_frames = max(cond_frame_counts)
        else:
            target_frames = num_frames

        # Second pass: truncate or pad each sample to target_frames
        batch_videos = []
        for i, video_tensor in enumerate(preprocessed):
            n = video_tensor.shape[2]
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
                cond_frame_counts[i] = target_frames
            elif n < target_frames:
                # Pad with last-frame repetition (matches official pipeline)
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))  # [C, target_frames, H, W]

        # Stack to [B, C, target_frames, H, W]
        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            # T_latent = temporal latent frames = (num_frames-1)//4+1  (VAE temporal downsample factor=4)
            # e.g. 5 frames → 2 latent frames, 9 frames → 3 latent frames
            latents = self.vae.encode(video).latent_dist.sample()  # [B, 16, T_latent, H/8, W/8]

        # Normalize latents (matches official pipeline: prepare_latents) # TODO check if this normalization is actually needed
        if self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            ) 
            sigma_data = self.scheduler.config.sigma_data
            latents = (latents - latents_mean) / latents_std * sigma_data

        return latents, cond_frame_counts # latents: [B, C, T_latent, H/8, W/8], cond_frame_counts: list[int]

    def build_inputs(self, images, instructions, **kwargs):
        """Build inputs for the DiT world model.

        Instead of chat templates (VLM), we:
        1. Encode text with T5
        2. Encode images with VAE
        3. Package for the DiT forward pass

        Returns:
            dict with keys matching what forward() expects
        """
        assert len(images) == len(instructions)

        # Ensure encoders are on the right device
        device = next(self.transformer.parameters()).device
        # self.text_encoder.to(device)
        # self.vae.to(device)

        text_embeds, text_mask = self._encode_text(instructions)
        latents, cond_frame_counts = self._encode_images(images)

        # Offload T5 and VAE to CPU to free VRAM for the transformer
        # self.text_encoder.to("cpu")
        # self.vae.to("cpu")
        # torch.cuda.empty_cache()

        # For feature extraction, use timestep=0 (clean / minimal noise)
        batch_size = latents.shape[0]
        device = latents.device
        _, _, t_lat, h_lat, w_lat = latents.shape # B, C, T_latent, H/8, W/8
        timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
        # condition_mask: tells DiT which latent frames are reliable conditions (=1) vs to-be-generated (=0).
        # In Video2World, this separates input frames from predicted future frames.
        # Here (action prediction, not generation), we set timestep=0 + condition_mask on real frames
        # so DiT runs a near-clean forward pass for feature extraction, not actual denoising.
        # in_channels = 16 (latents) + 1 (condition_mask) = 17
        # Shape: [B, 1, T_latent, H_latent, W_latent]
        condition_mask = latents.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        for i, n_cond in enumerate(cond_frame_counts):
            # Map pixel-frame count to latent-frame count
            n_cond_latent = (n_cond - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[i, :, :n_cond_latent] = 1.0

        # padding_mask: concat_padding_mask=True adds 1 more channel → 18 total
        # Shape: [1, 1, H_orig, W_orig] — all zeros = no padding
        # Will be resized to latent spatial dims by the transformer
        padding_mask = latents.new_zeros(1, 1, h_lat, w_lat)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        """Forward pass through the DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        Returns an output object with .hidden_states for compatibility.
        """
        is_wm = kwargs.pop("_is_wm_input", False)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        return_dict = kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)

        # Clear feature buffer
        self._intermediate_features.clear()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            dit_output = self.transformer(
                hidden_states=kwargs["hidden_states"],
                timestep=kwargs["timestep"],
                encoder_hidden_states=kwargs["encoder_hidden_states"],
                condition_mask=kwargs.get("condition_mask", None),
                padding_mask=kwargs.get("padding_mask", None),
            )

        # Build hidden_states tuple from captured intermediate features
        # Also reshape from [B, C, T, H, W] to [B, N_tokens, hidden_dim]
        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                # [B, C, T, H, W] -> [B, T*H*W, C]
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)

        # If no hooks fired (shouldn't happen), use transformer output
        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        # Build compatible output object
        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss

        return _WMOutput(hidden_states_tuple=tuple(extracted))

    def generate(self, **kwargs):
        """Video generation (for world-model imagination / planning).

        This builds the full Cosmos2VideoToWorldPipeline on-the-fly.
        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        """
        from diffusers import Cosmos2VideoToWorldPipeline

        pipe = Cosmos2VideoToWorldPipeline(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            safety_checker=None,
        )
        return pipe(**kwargs)
