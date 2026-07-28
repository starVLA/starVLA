from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from starVLA.model.modules.vlm.openpi_transformers.gemma import modeling_gemma

from deployment.model_server.tools.image_tools import resize_with_pad, to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.modules.action_model.OpenPI_ActionHead import (
    GemmaDims,
    OpenPI0ActionHead,
    OpenPIActionHeadBase,
    make_att_2d_masks,
    make_att_4d_masks,
)
from starVLA.model.modules.vlm.OpenPIPaliGemma import OpenPIPaliGemma
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclass
class Observation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    state: torch.Tensor


@dataclass
class PI0DefaultConfig:
    name: str = "PI0"
    precision: Literal["bfloat16", "float32"] = "bfloat16"

    action_dim: int = 32
    action_horizon: int = 50
    max_state_dim: int = 32
    discrete_state_input: bool = False
    max_token_len: int = 200
    num_inference_steps: int = 10

    image_resolution: list[int] = field(default_factory=lambda: [224, 224])
    image_keys: list[str] = field(default_factory=lambda: list(IMAGE_KEYS))

    tokenizer: dict = field(default_factory=lambda: {"model_path": "", "pad_state_value": -2.0})
    paligemma: dict = field(default_factory=lambda: {"model_name_or_path": ""})
    paligemma_config: dict = field(
        default_factory=lambda: {
            "width": 2048,
            "depth": 18,
            "mlp_dim": 16384,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )
    action_expert_config: dict = field(
        default_factory=lambda: {
            "width": 1024,
            "depth": 18,
            "mlp_dim": 4096,
            "num_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
        }
    )


def _as_dims(config) -> GemmaDims:
    return GemmaDims(
        width=int(config.width),
        depth=int(config.depth),
        mlp_dim=int(config.mlp_dim),
        num_heads=int(config.num_heads),
        num_kv_heads=int(config.num_kv_heads),
        head_dim=int(config.head_dim),
    )


class LazyPaliGemmaTokenizer:
    def __init__(self, max_len: int, model_path: str = ""):
        self.max_len = int(max_len)
        self.model_path = str(model_path or "")
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if not self.model_path:
            raise FileNotFoundError(
                "PI0/PI05 tokenizer.model_path is empty. Set framework.tokenizer.model_path "
                "to a local paligemma_tokenizer.model path for raw-text examples."
            )
        import sentencepiece

        with open(self.model_path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        return self._tokenizer

    def tokenize(self, prompt: str, state: np.ndarray | torch.Tensor | None = None):
        tokenizer = self._load()
        cleaned_text = str(prompt).strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            state_np = np.asarray(state, dtype=np.float32).reshape(-1)
            discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state.tolist()))
            prompt_text = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = tokenizer.encode(prompt_text, add_bos=True)
        else:
            tokens = tokenizer.encode(cleaned_text, add_bos=True) + tokenizer.encode("\n")

        token_len = len(tokens)
        if token_len < self.max_len:
            padding = [0] * (self.max_len - token_len)
            mask = [True] * token_len + [False] * len(padding)
            tokens = tokens + padding
        else:
            if token_len > self.max_len:
                logger.warning("PI tokenizer length %s exceeds max_token_len=%s; truncating", token_len, self.max_len)
            tokens = tokens[: self.max_len]
            mask = [True] * self.max_len
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=bool)


def _gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor | None):
    return x + y if gate is None else x + y * gate


def _norm(layernorm, hidden_states: torch.Tensor, cond: torch.Tensor | None):
    try:
        out = layernorm(hidden_states, cond=cond)
    except TypeError as exc:
        if cond is not None:
            dense = getattr(layernorm, "dense", None)
            if dense is None:
                raise RuntimeError(
                    "PI05 requires Gemma adaRMS support, but this layernorm has no dense module."
                ) from exc
            dtype = hidden_states.dtype
            eps = getattr(layernorm, "variance_epsilon", getattr(layernorm, "eps", 1e-6))
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            out = hidden_states * torch.rsqrt(variance + eps)
            scale, shift, gate = dense(cond.to(dense.weight.dtype))[:, None, :].chunk(3, dim=-1)
            return (out * (1 + scale) + shift).to(dtype), gate.to(dtype)
        out = layernorm(hidden_states)
    if isinstance(out, tuple):
        return out
    return out, None


@FRAMEWORK_REGISTRY.register("PI0")
@FRAMEWORK_REGISTRY.register("Pi0")
class PI0(baseframework):
    default_config_cls = PI0DefaultConfig
    action_head_cls: type[OpenPIActionHeadBase] = OpenPI0ActionHead
    model_name = "PI0"
    discrete_state_input = False

    def __init__(self, config=None, **kwargs):
        super().__init__()
        self._init_runtime_config(config)
        self._build_tokenizer()
        self._build_model_modules()
        self.to_precision(self.config.framework.precision)
        torch.set_float32_matmul_precision("highest")

    def _init_runtime_config(self, config):
        self.config = merge_framework_config(self.default_config_cls, config)
        self.config.framework.name = self.model_name
        self.discrete_state_input = bool(self.config.framework.discrete_state_input)

        self.action_dim = int(self.config.framework.action_dim)
        self.action_horizon = int(self.config.framework.action_horizon)
        self.max_state_dim = int(self.config.framework.max_state_dim)
        self.num_inference_steps = int(self.config.framework.num_inference_steps)
        self.image_resolution = tuple(int(x) for x in self.config.framework.image_resolution)
        self.image_keys = tuple(self.config.framework.image_keys)
        self.state_pad_value = float(self.config.framework.tokenizer.pad_state_value)

    def _build_tokenizer(self):
        self.tokenizer = LazyPaliGemmaTokenizer(
            max_len=int(self.config.framework.max_token_len),
            model_path=str(self.config.framework.tokenizer.model_path),
        )

    def _build_model_modules(self):
        vlm_config = _as_dims(self.config.framework.paligemma_config)
        expert_config = _as_dims(self.config.framework.action_expert_config)
        self.vlm = OpenPIPaliGemma(
            vlm_config,
            use_adarms=False,
            model_name_or_path=str(self.config.framework.paligemma.model_name_or_path),
        )
        self.action_head = self.action_head_cls(
            expert_config,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            max_state_dim=self.max_state_dim,
        )

    def to_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Unsupported PI precision: {precision}")
        self.to(dtype=torch.bfloat16)
        keep_fp32 = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, param in self.named_parameters():
            if any(selector in name for selector in keep_fp32):
                param.data = param.data.to(dtype=torch.float32)

    def _pad_array_2d(self, arr: np.ndarray, target_time: int, target_dim: int, pad_value: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] > target_time or arr.shape[1] > target_dim:
            raise ValueError(f"Input shape {arr.shape} exceeds target {(target_time, target_dim)}")
        out = np.full((target_time, target_dim), pad_value, dtype=np.float32)
        out[: arr.shape[0], : arr.shape[1]] = arr
        return out

    def _pil_to_tensor(self, image) -> torch.Tensor:
        image_np = np.asarray(image.convert("RGB")).copy()
        image_np = resize_with_pad(image_np[None], self.image_resolution[0], self.image_resolution[1])[0]
        return torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float() / 255.0 * 2.0 - 1.0

    def _prepare_examples(self, examples: list[dict], include_actions: bool):
        images, masks, states, actions, prompts = [], [], [], [], []
        for example in examples:
            raw_images = to_pil_preserve(example["image"])
            if isinstance(raw_images, tuple):
                raw_images = list(raw_images)
            elif not isinstance(raw_images, list):
                raw_images = [raw_images]
            sample_images, sample_masks = [], []
            provided_mask = example.get("image_mask")
            provided_mask = list(provided_mask) if provided_mask is not None else None
            for idx, _key in enumerate(self.image_keys):
                has_image = idx < len(raw_images)
                if has_image:
                    image = raw_images[idx]
                else:
                    image = raw_images[0].copy()
                    image.paste(0, (0, 0, image.size[0], image.size[1]))
                active = bool(provided_mask[idx]) if provided_mask is not None and idx < len(provided_mask) else has_image
                sample_images.append(self._pil_to_tensor(image))
                sample_masks.append(active)
            images.append(torch.stack(sample_images))
            masks.append(sample_masks)
            prompts.append(str(example.get("lang", example.get("language", ""))))

            state_value = example.get("state")
            if state_value is None:
                fill = self.state_pad_value if self.config.framework.discrete_state_input else 0.0
                state_np = np.full((1, self.max_state_dim), fill, dtype=np.float32)
            else:
                state_np = np.asarray(state_value, dtype=np.float32)
                if state_np.ndim == 2 and state_np.shape[0] > 1:
                    state_np = state_np[-1:]
                state_np = self._pad_array_2d(
                    state_np,
                    1 if state_np.ndim == 1 else state_np.shape[0],
                    self.max_state_dim,
                    self.state_pad_value if self.config.framework.discrete_state_input else 0.0,
                )
            states.append(torch.from_numpy(state_np))
            if include_actions:
                action_np = self._pad_array_2d(np.asarray(example["action"], dtype=np.float32), self.action_horizon, self.action_dim, 0.0)
                actions.append(torch.from_numpy(action_np))

        batch = {
            "image": torch.stack(images),
            "image_mask": torch.as_tensor(np.asarray(masks), dtype=torch.bool),
            "state": torch.stack(states),
            "lang": prompts,
        }
        if include_actions:
            batch["action"] = torch.stack(actions)
        return batch

    def _build_observation_from_batch(self, batch):
        device = self.action_head.action_in_proj.weight.device
        images = {key: batch["image"][:, idx].to(device=device, dtype=torch.float32) for idx, key in enumerate(self.image_keys)}
        masks = {key: batch["image_mask"][:, idx].to(device=device, dtype=torch.bool) for idx, key in enumerate(self.image_keys)}
        tokens, token_masks = [], []
        for idx, prompt in enumerate(batch["lang"]):
            token_state = batch["state"][idx].squeeze(0) if self.config.framework.discrete_state_input else None
            token, mask = self.tokenizer.tokenize(prompt, token_state)
            tokens.append(token)
            token_masks.append(mask)
        return Observation(
            images=images,
            image_masks=masks,
            tokenized_prompt=torch.as_tensor(np.asarray(tokens), device=device),
            tokenized_prompt_mask=torch.as_tensor(np.asarray(token_masks), device=device),
            state=batch["state"].to(device=device, dtype=torch.float32),
        )

    def _preprocess_observation(self, observation: Observation):
        return (
            [observation.images[key] for key in self.image_keys],
            [observation.image_masks[key] for key in self.image_keys],
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        embs, pad_masks, att_masks = [], [], []
        for image, image_mask in zip(images, img_masks, strict=True):
            image_emb = self.vlm.embed_image(image)
            bsize, num_img_embs = image_emb.shape[:2]
            embs.append(image_emb)
            pad_masks.append(image_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
        lang_emb = self.vlm.embed_language_tokens(lang_tokens)
        embs.append(lang_emb * math.sqrt(lang_emb.shape[-1]))
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :]
        return embs, pad_masks, att_masks.expand(pad_masks.shape[0], -1)

    def forward_with_action_expert(
        self,
        *,
        prefix_embs: torch.Tensor,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
        adarms_cond: torch.Tensor | None,
    ):
        streams = [prefix_embs, suffix_embs]
        conds = [None, adarms_cond]
        models = [self.vlm.language_model, self.action_head.model]
        for layer_idx in range(self.vlm.config.text_config.num_hidden_layers):
            streams = self.forward_shared_gemma_layer(layer_idx, models, streams, conds, attention_mask, position_ids)
        suffix_out, _ = _norm(models[1].norm, streams[1], conds[1])
        return suffix_out

    def forward_shared_gemma_layer(self, layer_idx, models, streams, conds, attention_mask, position_ids):
        query_states, key_states, value_states, gates = [], [], [], []
        for idx, hidden_states in enumerate(streams):
            layer = models[idx].layers[layer_idx]
            hidden_states, gate = _norm(layer.input_layernorm, hidden_states, conds[idx])
            gates.append(gate)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
            value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

        query_states = torch.cat(query_states, dim=2)
        key_states = torch.cat(key_states, dim=2)
        value_states = torch.cat(value_states, dim=2)
        dummy = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = self.vlm.language_model.rotary_emb(dummy, position_ids)
        query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, key_states, cos, sin, unsqueeze_dim=1
        )
        att_output, _ = modeling_gemma.eager_attention_forward(
            self.vlm.language_model.layers[layer_idx].self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            self.vlm.language_model.layers[layer_idx].self_attn.scaling,
        )
        head_dim = self.vlm.language_model.layers[layer_idx].self_attn.head_dim
        num_heads = self.vlm.language_model.config.num_attention_heads
        att_output = att_output.reshape(query_states.shape[0], -1, num_heads * head_dim)

        next_streams = []
        start = 0
        for idx, hidden_states in enumerate(streams):
            layer = models[idx].layers[layer_idx]
            end = start + hidden_states.shape[1]
            layer_att = att_output[:, start:end]
            if layer_att.dtype != layer.self_attn.o_proj.weight.dtype:
                layer_att = layer_att.to(layer.self_attn.o_proj.weight.dtype)
            out = layer.self_attn.o_proj(layer_att)
            out = _gated_residual(hidden_states, out, gates[idx])
            residual = out.clone()
            out, gate = _norm(layer.post_attention_layernorm, out, conds[idx])
            if out.dtype != layer.mlp.up_proj.weight.dtype:
                out = out.to(layer.mlp.up_proj.weight.dtype)
            out = layer.mlp(out)
            next_streams.append(_gated_residual(residual, out, gate))
            start = end
        return next_streams

    def forward_from_observation(self, observation: Observation, actions: torch.Tensor, noise=None, time=None, return_debug=False):
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        x_t, u_t, noise, time = self.action_head.prepare_flow_targets(actions, noise=noise, time=time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.action_head.embed_action_suffix(state.squeeze(1), x_t, time)
        if self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        attention_mask = make_att_4d_masks(make_att_2d_masks(pad_masks, att_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        suffix_out = self.forward_with_action_expert(
            prefix_embs=prefix_embs,
            suffix_embs=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            adarms_cond=adarms_cond,
        )
        v_t = self.action_head.decode_action_velocity(suffix_out)
        out = {"v_t": v_t, "action_loss": F.mse_loss(u_t, v_t, reduction="none").mean()}
        if return_debug:
            out.update({"u_t": u_t, "x_t": x_t, "time": time, "noise": noise})
        return out

    def forward(self, examples: list[dict] = None, **kwargs):
        '''
        Training forward for PI0 flow-matching action prediction.
        '''
        batch = self._prepare_examples(examples, include_actions=True)
        observation = self._build_observation_from_batch(batch)
        actions = batch["action"].to(device=self.action_head.action_in_proj.weight.device, dtype=torch.float32)
        return self.forward_from_observation(
            observation,
            actions,
            noise=kwargs.get("noise"),
            time=kwargs.get("time"),
            return_debug=bool(kwargs.get("return_debug", False)),
        )

    @torch.inference_mode()
    def predict_action(self, examples: list[dict] | dict = None, **kwargs):
        '''
        Prediction forward for PI0 flow-matching action prediction
        '''
        if not isinstance(examples, list):
            examples = [examples]
        batch = self._prepare_examples(examples, include_actions=False)
        observation = self._build_observation_from_batch(batch)
        actions = self.sample_actions(
            observation,
            noise=kwargs.get("noise"),
            num_steps=int(kwargs.get("num_steps", kwargs.get("num_inference_steps", self.num_inference_steps))),
        )
        return {"normalized_actions": actions.cpu().numpy()}

    def sample_actions(self, observation: Observation, noise=None, num_steps: int = 10):
        device = self.action_head.action_in_proj.weight.device
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks_4d = make_att_4d_masks(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.vlm.language_model.config._attn_implementation = "eager"
        prefix_output = self.vlm.forward_prefix(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
        )
        past_key_values = prefix_output.past_key_values
        bsize = state.shape[0]
        shape = (bsize, self.action_horizon, self.action_dim)
        x_t = self.action_head.sample_action_noise(shape, device) if noise is None else torch.as_tensor(noise, device=device, dtype=torch.float32)
        if x_t.ndim == 2:
            x_t = x_t[None]
        dt = torch.tensor(-1.0 / max(num_steps, 1), dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state.squeeze(1), prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_embs, suffix_pad_masks, suffix_att_masks, _adarms_cond = self.action_head.embed_action_suffix(state, x_t, timestep)
        suffix_len = suffix_pad_masks.shape[1]
        bsize, prefix_len = prefix_pad_masks.shape
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention_mask = make_att_4d_masks(torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2))
        position_ids = torch.sum(prefix_pad_masks, dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1
        suffix_embs = suffix_embs.to(torch.bfloat16) if self.vlm.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16 else suffix_embs
        return self.action_head(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            suffix_embs=suffix_embs,
            adarms_cond=_adarms_cond,
        )

    def _is_expected_openpi_missing_key(self, key: str) -> bool:
        if not getattr(self.action_head, "use_adarms", False):
            return False
        prefix = "action_head.gemma_expert.model."
        if key == prefix + "norm.weight":
            return True
        if not key.startswith(prefix + "layers."):
            return False
        return key.endswith(".input_layernorm.weight") or key.endswith(".post_attention_layernorm.weight")

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
        path = Path(checkpoint_path)
        if path.is_dir():
            for name in ("model.safetensors", "pytorch_model.bin", "model.pt", "checkpoint.pt", "ckpt.pt"):
                candidate = path / name
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"No supported PI checkpoint file found under {path}")
        return path

    @classmethod
    def _load_checkpoint_state_dict(cls, checkpoint_path: str | Path):
        checkpoint_path = cls._resolve_checkpoint_path(checkpoint_path)
        if str(checkpoint_path).endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict):
                for key in ("state_dict", "model_state_dict", "module", "model"):
                    if key in state_dict and isinstance(state_dict[key], dict):
                        state_dict = state_dict[key]
                        break
        return state_dict

    def _normalize_state_dict(self, state_dict: dict):
        current = self.state_dict()
        normalized = {}
        for key, value in state_dict.items():
            clean_key = key
            changed = True
            while changed:
                changed = False
                for prefix in ("model.", "module.", "_orig_mod."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        changed = True
            if clean_key.startswith("paligemma_with_expert.paligemma.model."):
                clean_key = clean_key.replace(
                    "paligemma_with_expert.paligemma.model.",
                    "vlm.model.model.",
                    1,
                )
            elif clean_key.startswith("paligemma_with_expert.paligemma."):
                clean_key = clean_key.replace("paligemma_with_expert.paligemma.", "vlm.model.", 1)
            elif clean_key.startswith("paligemma_with_expert.gemma_expert."):
                clean_key = clean_key.replace("paligemma_with_expert.gemma_expert.", "action_head.gemma_expert.", 1)
            elif clean_key.startswith("action_in_proj."):
                clean_key = clean_key.replace("action_in_proj.", "action_head.action_in_proj.", 1)
            elif clean_key.startswith("action_out_proj."):
                clean_key = clean_key.replace("action_out_proj.", "action_head.action_out_proj.", 1)
            elif clean_key.startswith("state_proj."):
                clean_key = clean_key.replace("state_proj.", "action_head.state_proj.", 1)
            elif clean_key.startswith("action_time_mlp_in."):
                clean_key = clean_key.replace("action_time_mlp_in.", "action_head.action_time_mlp_in.", 1)
            elif clean_key.startswith("action_time_mlp_out."):
                clean_key = clean_key.replace("action_time_mlp_out.", "action_head.action_time_mlp_out.", 1)
            elif clean_key.startswith("time_mlp_in."):
                clean_key = clean_key.replace("time_mlp_in.", "action_head.time_mlp_in.", 1)
            elif clean_key.startswith("time_mlp_out."):
                clean_key = clean_key.replace("time_mlp_out.", "action_head.time_mlp_out.", 1)
            normalized[clean_key] = value
        embed_key = "vlm.model.model.language_model.embed_tokens.weight"
        lm_head_key = "vlm.model.lm_head.weight"
        if embed_key not in normalized and lm_head_key in normalized and embed_key in current:
            if normalized[lm_head_key].shape == current[embed_key].shape:
                normalized[embed_key] = normalized[lm_head_key]
        return normalized

    def load_pretrained_checkpoint(self, checkpoint_path: str | Path, reload_modules: str | None = None):
        state_dict = self._normalize_state_dict(self._load_checkpoint_state_dict(checkpoint_path))
        if reload_modules:
            for path in [p.strip() for p in reload_modules.split(",") if p.strip()]:
                module = self
                for part in path.split("."):
                    module = getattr(module, part)
                prefix = path + "."
                sub_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                module.load_state_dict(sub_state, strict=False)
            return self

        current = self.state_dict()
        filtered, shape_mismatch = {}, []
        for key, value in state_dict.items():
            if key in current and getattr(value, "shape", None) == current[key].shape:
                filtered[key] = value
            elif key in current:
                shape_mismatch.append((key, tuple(value.shape), tuple(current[key].shape)))
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        expected_missing = [key for key in missing if self._is_expected_openpi_missing_key(key)]
        real_missing = [key for key in missing if key not in expected_missing]
        logger.info(
            "%s checkpoint load: checkpoint=%s loaded=%s model=%s missing=%s expected_missing=%s unexpected=%s shape_mismatch=%s",
            type(self).__name__,
            len(state_dict),
            len(filtered),
            len(current),
            len(real_missing),
            len(expected_missing),
            len(unexpected),
            len(shape_mismatch),
        )
        if real_missing:
            logger.info("%s first missing keys: %s", type(self).__name__, real_missing[:10])
        if shape_mismatch:
            logger.info("%s first shape mismatches: %s", type(self).__name__, shape_mismatch[:10])
        return self
