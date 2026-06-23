# Copyright 2026 starVLA community. All rights reserved.
"""QwenVAR: autoregressive LIBERO action-token policy using VAR Stage 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_tokenizer import VARTokenTextCodec, load_frozen_var_action_tokenizer
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)
IGNORE_INDEX = -100


@dataclass
class QwenVARDefaultConfig:
    """Default framework config for QwenVAR."""

    name: str = "QwenVAR"
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct-VARAction",
            "attn_implementation": "flash_attention_2",
        }
    )
    stage1_tokenizer: dict = field(
        default_factory=lambda: {
            "artifact": "playground/Checkpoints/var_stage1_pi05_libero/best_recon.ckpt",
            "stage1_config": "examples/LIBERO/train_files/train_var_stage1_pi05_libero.yaml",
            "freeze": True,
            "token_cache": None,
        }
    )
    action_token_text: dict = field(
        default_factory=lambda: {
            "prefix": "<var_action_",
            "suffix": ">",
        }
    )
    labeling: dict = field(
        default_factory=lambda: {
            # Match QwenFast: ignore everything before the first action token,
            # then train the assistant action-token answer as normal LM text.
            # Set to "action_tokens_only" for the older stricter VAR masking.
            "mask_strategy": "qwenfast",
        }
    )
    generation: dict = field(
        default_factory=lambda: {
            # Match QwenFast by default. Constrained decoding is useful as a
            # later trick, but it is not part of the FAST baseline recipe.
            "constrain_to_action_tokens": False,
            "do_sample": False,
            "max_length": 2048,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenVAR")
class QwenVAR(baseframework):
    """Qwen-VL policy that predicts frozen VAR Stage 1 action tokens."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenVARDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        stage1_path = self.config.framework.stage1_tokenizer.get("artifact", None) or self.config.framework.stage1_tokenizer.get("checkpoint", None)
        if stage1_path is None:
            raise ValueError("QwenVAR requires framework.stage1_tokenizer.artifact or .checkpoint.")
        artifact = load_frozen_var_action_tokenizer(stage1_path, device="cpu")
        self.stage1_artifact_id = artifact.artifact_id
        self.action_spec = artifact.action_spec
        self.stage1_tokenizer = artifact.tokenizer
        self.stage1_tokenizer.eval()
        for param in self.stage1_tokenizer.parameters():
            param.requires_grad_(False)

        token_cfg = self.config.framework.action_token_text
        self.var_token_codec = VARTokenTextCodec(
            codebook_size=self.stage1_tokenizer.codebook_size,
            prefix=str(token_cfg.get("prefix", "<var_action_")),
            suffix=str(token_cfg.get("suffix", ">")),
        )
        self.token_dim = int(self.stage1_tokenizer.token_dim)
        self._action_token_id_set: set[int] | None = None
        self._action_token_id_bounds: tuple[int, int] | None = None

    def _ensure_action_tokens(self) -> set[int]:
        if self._action_token_id_set is not None:
            return self._action_token_id_set
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        token_ids = []
        missing = []
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token in self.var_token_codec.all_token_strings():
            token_id = int(tokenizer.convert_tokens_to_ids(token))
            if token_id < 0 or (unk_id is not None and token_id == unk_id):
                missing.append(token)
            token_ids.append(token_id)
        if missing:
            raise ValueError(
                "QwenVAR action special tokens are missing from the tokenizer. "
                f"First missing tokens: {missing[:5]}"
            )
        self._action_token_id_set = set(token_ids)
        return self._action_token_id_set

    def _action_token_id_range(self) -> tuple[int, int]:
        """Return the contiguous tokenizer-id range for VAR action tokens."""
        if self._action_token_id_bounds is None:
            self._action_token_id_bounds = self.var_token_codec.tokenizer_id_range(
                self.qwen_vl_interface.processor.tokenizer
            )
        return self._action_token_id_bounds

    def _action_token_ids_for_generation(self) -> list[int]:
        return sorted(self._ensure_action_tokens())

    def _mask_labels_after_first_action(self, labels: torch.Tensor) -> torch.Tensor:
        """QwenFast-compatible SFT labels: keep tokens from first action onward."""
        action_token_min, action_token_max = self._action_token_id_range()
        for row in range(labels.shape[0]):
            seq = labels[row]
            mask = (seq >= action_token_min) & (seq <= action_token_max)
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                seq[:] = IGNORE_INDEX
            else:
                seq[: int(indices[0].item())] = IGNORE_INDEX
        return labels

    def _mask_labels_action_tokens_only(self, labels: torch.Tensor) -> torch.Tensor:
        """Legacy VAR labels: keep only VAR action special-token positions."""
        action_token_ids = self._ensure_action_tokens()
        token_positions = torch.arange(labels.shape[1], device=labels.device)
        for row in range(labels.shape[0]):
            seq = labels[row]
            mask = torch.zeros_like(seq, dtype=torch.bool)
            for token_id in action_token_ids:
                mask |= seq == int(token_id)
            indices = torch.nonzero(mask, as_tuple=False).flatten()
            if indices.numel() == 0:
                seq[:] = IGNORE_INDEX
            else:
                first_action = int(indices[0].item())
                seq[:first_action] = IGNORE_INDEX
                seq[~mask & (token_positions >= first_action)] = IGNORE_INDEX
        return labels

    def _build_qwenvl_inputs(self, *, images: list, instructions: list[str], solutions: list[str] | None = None):
        messages = []
        for imgs, instruction in zip(images, instructions, strict=True):
            content = [{"type": "image", "image": img} for img in imgs]
            prompt = instruction
            if "CoT_prompt" in self.config.datasets.vla_data:
                prompt = str(self.config.datasets.vla_data.get("CoT_prompt", "")).replace("{instruction}", instruction)
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[len(messages)]}]})
            messages.append(msg)

        batch_inputs = self.qwen_vl_interface.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            mask_strategy = str(self.config.framework.get("labeling", {}).get("mask_strategy", "qwenfast"))
            if mask_strategy == "qwenfast":
                labels = self._mask_labels_after_first_action(labels)
            elif mask_strategy == "action_tokens_only":
                labels = self._mask_labels_action_tokens_only(labels)
            else:
                raise ValueError(f"Unsupported QwenVAR label mask strategy: {mask_strategy!r}")
            labels[labels == self.qwen_vl_interface.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels
        return batch_inputs.to(self.qwen_vl_interface.model.device)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        action_tokens = [example["action_tokens"].detach().cpu().tolist() for example in examples]
        token_text = [self.var_token_codec.ids_to_text(tokens) for tokens in action_tokens]
        qwen_inputs = self._build_qwenvl_inputs(images=batch_images, instructions=instructions, solutions=token_text)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
        )
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss.detach()).all().item():
            trainable_param = next(param for param in self.qwen_vl_interface.parameters() if param.requires_grad)
            loss = trainable_param.float().sum() * 0.0
        return {"action_loss": loss}

    def _generated_ids_to_var_tokens(self, generated_ids: torch.LongTensor) -> tuple[torch.LongTensor, list[dict[str, Any]]]:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        action_token_min, action_token_max = self._action_token_id_range()
        rows = []
        diagnostics = []
        for row_tensor in generated_ids.detach().cpu():
            mask = (row_tensor >= action_token_min) & (row_tensor <= action_token_max)
            action_tokenizer_ids = row_tensor[mask].tolist()
            if action_tokenizer_ids:
                ids = [int(token_id) - action_token_min for token_id in action_tokenizer_ids]
                ids = ids[: self.token_dim]
            else:
                ids = self.var_token_codec.ids_from_tokenizer_ids(row_tensor.tolist(), tokenizer, expected_len=self.token_dim)
            valid_len = len(ids)
            if valid_len < self.token_dim:
                ids = ids + [0] * (self.token_dim - valid_len)
            rows.append(ids[: self.token_dim])
            diagnostics.append({"valid_token_count": valid_len, "padded": valid_len < self.token_dim})
        return torch.as_tensor(rows, dtype=torch.long), diagnostics

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict[str, np.ndarray]:
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        qwen_inputs = self._build_qwenvl_inputs(images=batch_images, instructions=instructions)
        constrain_to_action_tokens = bool(
            kwargs.get(
                "constrain_to_action_tokens",
                self.config.framework.get("generation", {}).get("constrain_to_action_tokens", False),
            )
        )
        max_new_tokens = int(kwargs.get("max_new_tokens", self.token_dim))
        generation_kwargs: dict[str, Any] = {
            "do_sample": bool(kwargs.get("do_sample", self.config.framework.get("generation", {}).get("do_sample", False))),
        }
        if constrain_to_action_tokens:
            action_token_ids = self._action_token_ids_for_generation()

            def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> list[int]:
                return action_token_ids

            generation_kwargs["max_new_tokens"] = self.token_dim
            generation_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
        else:
            generation_kwargs["max_length"] = int(
                kwargs.get("max_length", self.config.framework.get("generation", {}).get("max_length", 2048))
            )
            if "max_new_tokens" in kwargs:
                generation_kwargs["max_new_tokens"] = max_new_tokens

        generated_ids = self.qwen_vl_interface.model.generate(
            **qwen_inputs,
            **generation_kwargs,
        )
        input_token_len = qwen_inputs["input_ids"].shape[1]
        prompt_token_ids, prompt_diagnostics = self._generated_ids_to_var_tokens(generated_ids[:, :input_token_len])
        token_ids, diagnostics = self._generated_ids_to_var_tokens(generated_ids[:, input_token_len:])
        for diag, prompt_diag in zip(diagnostics, prompt_diagnostics, strict=True):
            diag.update(
                {
                    "constrain_to_action_tokens": constrain_to_action_tokens,
                    "max_new_tokens": generation_kwargs.get("max_new_tokens"),
                    "do_sample": generation_kwargs.get("do_sample"),
                    "input_token_len": int(input_token_len),
                    "generated_token_len": int(generated_ids.shape[1]),
                    "new_token_len": int(generated_ids.shape[1] - input_token_len),
                    "prompt_valid_token_count": int(prompt_diag["valid_token_count"]),
                }
            )
        token_ids = token_ids.to(next(self.stage1_tokenizer.parameters()).device)
        normalized_actions = self.stage1_tokenizer.decode(token_ids).detach().cpu().float().numpy()
        return {
            "normalized_actions": normalized_actions,
            "action_tokens": token_ids.detach().cpu().numpy(),
            "generation_diagnostics": diagnostics,
        }
