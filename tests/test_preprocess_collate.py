"""Tests for the DataLoader preprocessing collate (preprocess_in_collate).

Most tests need a local Qwen-VL checkpoint with processor files; set
STARVLA_TEST_VLM to its directory (they skip otherwise, e.g. in CI without
model weights). test_forward_dispatch_* run everywhere (no weights needed).
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from starVLA.model.framework.VLM4A.qwenpi_collate import QwenPIPreprocessCollate

MODEL_DIR = os.environ.get("STARVLA_TEST_VLM", "")
needs_model = pytest.mark.skipif(not os.path.isdir(MODEL_DIR), reason="STARVLA_TEST_VLM not set")

COT = "Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format."


def make_batch(n=3):
    rng = np.random.default_rng(0)
    return [
        {
            "image": [Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2)],
            "lang": ["turn on the stove", "put the bowl on the plate", "open the top drawer and put the bowl inside"][i % 3],
            "action": rng.standard_normal((16, 7)).astype(np.float32),
            "state": rng.standard_normal((1, 7)).astype(np.float32),
        }
        for i in range(n)
    ]


@needs_model
def test_collate_matches_in_forward_path():
    """Collate output must be token-exact vs the REAL in-forward path
    (_QWen3_5_VL_Interface.build_qwenvl_inputs run unbound with a stub self)."""
    from omegaconf import OmegaConf

    from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_VL_Interface
    from transformers import AutoProcessor

    batch = make_batch()
    out = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT)(batch)

    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    proc.tokenizer.padding_side = "left"
    stub = SimpleNamespace(
        processor=proc,
        config=OmegaConf.create({"datasets": {"vla_data": {"CoT_prompt": COT}}}),
        model=SimpleNamespace(device="cpu"),
    )
    ref = _QWen3_5_VL_Interface.build_qwenvl_inputs(
        stub, images=[ex["image"] for ex in batch], instructions=[ex["lang"] for ex in batch]
    )
    for key in ("input_ids", "attention_mask", "image_grid_thw"):
        assert torch.equal(out[key], ref[key]), key
    # pixels ship as bf16 (the model casts them there anyway under autocast)
    assert torch.equal(out["pixel_values"], ref["pixel_values"].to(torch.bfloat16))
    assert torch.equal(out["action"], torch.from_numpy(np.stack([ex["action"] for ex in batch])))
    assert torch.equal(out["state"], torch.from_numpy(np.stack([ex["state"] for ex in batch])))


@needs_model
def test_collate_pad_to_fixed_length():
    batch = make_batch()
    free = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT)(batch)
    fixed = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT, pad_to=192)(batch)
    assert fixed["input_ids"].shape[1] == 192 > free["input_ids"].shape[1]
    assert torch.equal(fixed["attention_mask"].sum(1), free["attention_mask"].sum(1))
    n = free["input_ids"].shape[1]
    assert torch.equal(fixed["input_ids"][:, -n:], free["input_ids"][:, -n:])  # left-padded


@needs_model
def test_collate_pad_to_matches_processor_maxlen():
    """Manual left-padding must be byte-identical to the processor's own
    max_length padding (guards the hand-pad against tokenizer drift)."""
    from starVLA.model.modules.vlm.qwenvl_messages import build_qwenvl_messages
    from transformers import AutoProcessor

    batch = make_batch()
    out = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT, pad_to=192)(batch)
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    proc.tokenizer.padding_side = "left"
    ref = proc.apply_chat_template(
        build_qwenvl_messages([ex["image"] for ex in batch], [ex["lang"] for ex in batch], COT),
        tokenize=True, padding="max_length", max_length=192,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    for key in ("input_ids", "attention_mask"):
        assert torch.equal(out[key], ref[key]), key


@needs_model
def test_collate_pad_to_too_short_falls_back():
    batch = make_batch()
    ref = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT)(batch)
    short = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT, pad_to=8)
    out = short(batch)
    assert "too_long" in short._warned  # warned once, batch keeps dynamic length
    assert torch.equal(out["input_ids"], ref["input_ids"])


@needs_model
def test_singleton_batch_skips_fixed_pad():
    """Qwen3.5's linear-attention mask gate ignores padding when B == 1
    (transformers #46773), so fixed padding must be skipped for singletons."""
    batch = make_batch(n=1)
    c = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT, pad_to=192)
    out = c(batch)
    free = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT)(batch)
    assert out["input_ids"].shape == free["input_ids"].shape  # no pads added
    assert "singleton" in c._warned


@needs_model
def test_text_only_batch_has_no_pixel_values():
    batch = make_batch()
    for ex in batch:
        ex["image"] = []
    out = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT)(batch)  # must not raise
    assert "input_ids" in out and "pixel_values" not in out


@needs_model
def test_build_collate_backend_gate():
    from omegaconf import OmegaConf

    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

    ok = OmegaConf.create({"framework": {"qwenvl": {"base_vlm": MODEL_DIR}},
                           "datasets": {"vla_data": {"CoT_prompt": COT, "collate_pad_to": 192}}})
    assert MODEL_DIR.endswith("Qwen3.5-4B") and Qwen_PI.build_collate(ok) is not None
    for unverified in ("/models/gemma-4-9b", "/models/Qwen2.5-VL-7B"):
        bad = OmegaConf.create({"framework": {"qwenvl": {"base_vlm": unverified}},
                                "datasets": {"vla_data": {}}})
        assert Qwen_PI.build_collate(bad) is None  # unverified backend -> no collate


@needs_model
def test_obs_image_size_resize_applied():
    """The collate applies the same obs_image_size resize the raw prediction
    paths apply, so prepared and raw predictions see identical pixels."""
    batch = make_batch(n=2)
    # note: sizes below the processor's min_pixels get upscaled back by
    # smart_resize, so use a larger target to make the resize observable.
    big = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT, obs_image_size=(448, 448))(batch)
    full = QwenPIPreprocessCollate(MODEL_DIR, cot_prompt=COT)(batch)
    assert big["image_grid_thw"][:, 1:].prod(1).sum() > full["image_grid_thw"][:, 1:].prod(1).sum()


def _stub_qwenpi(record):
    """Weights-free Qwen_PI with stubbed encoder to test forward dispatch."""
    from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI

    m = object.__new__(Qwen_PI)
    param = torch.nn.Parameter(torch.zeros(1))
    m.qwen_vl_interface = SimpleNamespace(model=SimpleNamespace(parameters=lambda: iter([param])))

    def encode(batch_images=None, instructions=None, qwen_inputs=None):
        record["qwen_inputs"] = qwen_inputs
        record["raw"] = batch_images is not None
        raise RuntimeError("stop-after-dispatch")

    m._encode_vl_hidden_states = encode
    return m


def test_forward_dispatch_preprocessed_dict():
    record = {}
    m = _stub_qwenpi(record)
    batch = {
        "input_ids": torch.ones(2, 5, dtype=torch.long),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
        "action": torch.zeros(2, 16, 7),
        "state": torch.zeros(2, 1, 7),
    }
    with pytest.raises(RuntimeError, match="stop-after-dispatch"):
        m.forward(batch)
    assert record["qwen_inputs"] is not None and not record.get("raw")
    # action/state must be popped before the VLM sees the batch
    assert set(record["qwen_inputs"]) == {"input_ids", "attention_mask"}


def test_forward_dispatch_raw_examples():
    record = {}
    m = _stub_qwenpi(record)
    examples = [{"image": [], "lang": "x", "action": np.zeros((16, 7))}]
    with pytest.raises(RuntimeError, match="stop-after-dispatch"):
        m.forward(examples)
    assert record.get("raw") is True and record["qwen_inputs"] is None
