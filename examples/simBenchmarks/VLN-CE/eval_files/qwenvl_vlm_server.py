# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Qwen-VL websocket server for VLN-CE evaluation.

The server reuses the lightweight websocket protocol under
``deployment/model_server/tools``. A client can send either

1. ``{"examples": [{"image": [np.ndarray, ...], "lang": "..."}]}``, or
2. ``{"images": [np.ndarray, ...], "instruction": "..."}``, or
3. ``{"messages": [...]}`` already in Qwen-VL chat-template format.

It returns ``{"outputs": [text, ...], "output": text}``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen3VLForConditionalGeneration

try:
    from transformers import Qwen3VLMoeForConditionalGeneration
except ImportError:  # transformers without Qwen3 MoE support
    Qwen3VLMoeForConditionalGeneration = None

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

LOGGER = logging.getLogger(__name__)


class QwenVLVLNServer:
    def __init__(
        self,
        ckpt_path: str,
        attn_implementation: str = "flash_attention_2",
        dtype: str = "bf16",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        num_video_frames: int = 8,
        trust_remote_code: bool = False,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.max_new_tokens = max_new_tokens
        self.num_video_frames = num_video_frames
        self.metadata = {
            "model_type": "qwenvl_vlnce",
            "ckpt_path": ckpt_path,
            "max_new_tokens": max_new_tokens,
            "num_video_frames": num_video_frames,
        }

        model_cls = self._select_model_class(ckpt_path)
        torch_dtype = self._resolve_dtype(dtype)
        load_kwargs: dict[str, Any] = {
            "attn_implementation": attn_implementation,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype

        LOGGER.info("Loading %s from %s", model_cls.__name__, ckpt_path)
        try:
            self.model = model_cls.from_pretrained(ckpt_path, **load_kwargs)
        except TypeError:
            if "dtype" not in load_kwargs:
                raise
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            self.model = model_cls.from_pretrained(ckpt_path, **load_kwargs)

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(ckpt_path, trust_remote_code=trust_remote_code)
        LOGGER.info("Model loaded. device=%s", getattr(self.model, "device", "unknown"))

    @staticmethod
    def _select_model_class(ckpt_path: str):
        name = Path(ckpt_path.rstrip("/")).name.lower()
        full = ckpt_path.lower()
        if "qwen3" in full:
            if ("a3b" in name or "a22b" in name or "moe" in name) and Qwen3VLMoeForConditionalGeneration is not None:
                return Qwen3VLMoeForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        if "qwen2.5" in full or "qwen2_5" in full:
            return Qwen2_5_VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration

    @staticmethod
    def _resolve_dtype(dtype: str):
        dtype = dtype.lower()
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32", "none"}:
            return None
        raise ValueError(f"Unsupported dtype: {dtype}")

    def predict_action(self, **payload) -> dict[str, Any]:
        max_new_tokens = int(payload.pop("max_new_tokens", self.max_new_tokens))

        if "messages" in payload:
            outputs = [self._generate(self._normalize_messages(payload["messages"]), max_new_tokens=max_new_tokens)]
        else:
            examples = payload.get("examples")
            if examples is None:
                examples = [payload]
            outputs = [self._infer_example(example, max_new_tokens=max_new_tokens) for example in examples]

        return {"outputs": outputs, "output": outputs[0] if outputs else ""}

    def _infer_example(self, example: dict[str, Any], max_new_tokens: int) -> str:
        images = example.get("image", example.get("images", []))
        if images is None:
            images = []
        if not isinstance(images, (list, tuple)):
            images = [images]
        pil_images = [self._to_pil_image(image) for image in images]

        question = example.get("question") or example.get("prompt")
        instruction = example.get("lang") or example.get("instruction") or example.get("task") or ""
        messages = self._build_messages(pil_images, instruction=instruction, question=question)
        return self._generate(messages, max_new_tokens=max_new_tokens)

    def _generate(self, messages: list[dict[str, Any]], max_new_tokens: int) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs = self._collect_image_inputs(messages)
        processor_kwargs = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs

        LOGGER.info("Prompt: %s", text)

        inputs = self.processor(**processor_kwargs).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()

    def _collect_image_inputs(self, messages: list[dict[str, Any]]) -> list[Image.Image]:
        image_inputs = []
        for message in messages:
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    image_inputs.append(self._to_pil_image(item["image"]))
        return image_inputs

    def _build_messages(
        self,
        images: list[Image.Image],
        instruction: str,
        question: str | None = None,
    ) -> list[dict[str, Any]]:
        if question is None:
            history_images = "<image>\n" * max(len(images) - 1, 0)
            QUESTION = (
                "Imagine you are an autonomous robot performing a vision-language navigation task.\n"
                "You are given a sequence of historical observations {history_images} "
                "and the current observation <image>\n\n"
                "Your goal is: \"{instruction}\".\n\n"
                "Based on the history and current view, analyze the environment and "
                "decide the best next action to safely reach the goal.\n\n"
            )
            question = QUESTION.format(history_images=history_images, instruction=instruction)

        image_pool = [{"type": "image", "image": image} for image in images]
        content: list[dict[str, Any]] = []

        if "<image>" not in question:
            content.extend(image_pool)
            content.append({"type": "text", "text": question})
            return [{"role": "user", "content": content}]

        for segment in re.split(r"(<image>|<video>)", question):
            if segment == "<image>":
                if not image_pool:
                    raise ValueError("Number of <image> placeholders exceeds the number of provided images")
                content.append(image_pool.pop(0))
            elif segment == "<video>":
                raise ValueError("<video> placeholders are not supported by this simple image-list server")
            elif segment.strip():
                content.append({"type": "text", "text": segment.strip()})

        content.extend(image_pool)
        return [{"role": "user", "content": content}]

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for message in messages:
            content = []
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "image":
                    content.append({**item, "image": self._to_pil_image(item["image"])})
                else:
                    content.append(item)
            normalized.append({**message, "content": content})
        return normalized

    @staticmethod
    def _to_pil_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict) and "image" in image:
            return QwenVLVLNServer._to_pil_image(image["image"])
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image).convert("RGB")

        arr = np.asarray(image)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.size > 0 and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve a Qwen-VL model for VLN-CE evaluation over websocket.")
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6694)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32", "none"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_video_frames", type=int, default=8)
    parser.add_argument("--idle_timeout", type=int, default=-1, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    wrapper = QwenVLVLNServer(
        ckpt_path=args.ckpt_path,
        attn_implementation=args.attn_implementation,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        num_video_frames=args.num_video_frames,
        trust_remote_code=args.trust_remote_code,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    LOGGER.info("Creating Qwen-VL server (host: %s, ip: %s, bind: %s:%s)", hostname, local_ip, args.host, args.port)
    server = WebsocketPolicyServer(
        policy=wrapper,
        host=args.host,
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    LOGGER.info("server running ... metadata=%s", wrapper.metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
