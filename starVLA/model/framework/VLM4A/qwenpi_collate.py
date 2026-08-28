"""DataLoader-side preprocessing for QwenPI (datasets.vla_data.preprocess_in_collate).

QwenPI-specific by design: it encodes QwenPI's prompt construction, its
action/state batch layout, and the Qwen3.5 processor semantics. Other
frameworks opt in to the same mechanism by overriding
``baseframework.build_collate`` with their own collate.
"""

import logging

logger = logging.getLogger(__name__)


class QwenPIPreprocessCollate:
    """Runs the QwenVL processor inside DataLoader workers instead of forward.

    On the main process the processor (chat template + image resize + packing)
    costs ~50 ms (bs=16) to ~100 ms (bs=32) per step with the GPU fully idle;
    inside workers it overlaps with training. Output tokens are identical to
    the in-forward path (same processor, same message construction; the same
    ``obs_image_size`` resize the prediction paths apply is applied here).

    ``pad_to`` > 0 pads every batch to that fixed length (manually, after a
    single processor pass) so downstream shape-keyed kernel caches see one
    sequence length. Skipped for batch_size 1 — the current Qwen3.5
    linear-attention layers only apply the padding mask when batch > 1
    (huggingface/transformers#46773), so singleton left-pads would leak into
    the recurrent state. Over-long batches keep their dynamic length after a
    one-time warning.
    """

    def __init__(self, base_vlm, cot_prompt=None, pad_to=0, obs_image_size=None):
        self.base_vlm = base_vlm
        self.cot_prompt = cot_prompt
        self.pad_to = int(pad_to or 0)
        self.obs_image_size = obs_image_size
        self._warned = set()
        # Built eagerly in the parent so fork-started workers share it
        # copy-on-write; __getstate__ drops it so spawn workers rebuild.
        self._processor = self._build_processor()

    def _build_processor(self):
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.base_vlm)
        processor.tokenizer.padding_side = "left"
        return processor

    def __getstate__(self):
        return {**self.__dict__, "_processor": None}

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def __call__(self, batch):
        import numpy as np
        import torch

        if self._processor is None:  # spawn-started worker
            self._processor = self._build_processor()
        from deployment.model_server.tools.image_tools import to_pil_preserve

        from starVLA.model.modules.vlm.qwenvl_messages import build_qwenvl_messages

        images = [to_pil_preserve(ex["image"]) for ex in batch]
        if self.obs_image_size:
            # Same resize the raw prediction paths apply before the processor.
            from starVLA.training.trainer_utils.trainer_tools import resize_images

            images = resize_images(images, target_size=self.obs_image_size)
        messages = build_qwenvl_messages(images, [ex["lang"] for ex in batch], self.cot_prompt)

        # Single processor pass (dynamic pad-to-longest); the fixed length is
        # applied by hand below so over-long batches are never processed twice.
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, padding=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        out = dict(inputs)
        seq_len = out["input_ids"].shape[1]
        if self.pad_to and len(batch) == 1:
            self._warn_once("singleton", "collate_pad_to skipped for batch_size 1 (padding-mask gate in the backbone)")
        elif self.pad_to and seq_len > self.pad_to:
            self._warn_once(
                "too_long", f"collate_pad_to={self.pad_to} is shorter than a sample (length {seq_len}); "
                "this batch keeps its dynamic length"
            )
        elif self.pad_to and seq_len < self.pad_to:
            pad = self.pad_to - seq_len
            fills = {"input_ids": self._processor.tokenizer.pad_token_id, "attention_mask": 0, "mm_token_type_ids": 0}
            for key, fill in fills.items():
                if key in out:  # left-pad, matching the tokenizer's padding_side
                    out[key] = torch.nn.functional.pad(out[key], (pad, 0), value=fill)
        if "pixel_values" in out:
            # Qwen3.5 consumes pixels under bf16 autocast (the first op casts
            # them anyway); shipping bf16 halves the worker-to-main IPC volume.
            out["pixel_values"] = out["pixel_values"].to(torch.bfloat16)
        out["action"] = torch.from_numpy(np.array([ex["action"] for ex in batch]))
        if "state" in batch[0]:
            out["state"] = torch.from_numpy(np.array([ex["state"] for ex in batch]))
        return out
