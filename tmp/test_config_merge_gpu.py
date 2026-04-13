"""Quick GPU test: verify QwenOFT build + forward with merged default config."""
import sys
sys.path.insert(0, "/home/jye624/Projcets/starVLA")

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.base_framework import build_framework

# ── Test QwenOFT with minimal YAML (most params from defaults) ──
cfg = OmegaConf.create({
    "framework": {
        "name": "QwenOFT",
        "qwenvl": {
            # Use existing symlinked model that has full weights
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        },
    },
    "datasets": {
        "vla_data": {
            "default_image_resolution": [3, 224, 224],
        },
    },
})

print("Building QwenOFT framework (using merged defaults)...")
model = build_framework(cfg)
print(f"  Model class: {type(model).__name__}")
print(f"  chunk_len = {model.chunk_len}")
print(f"  action_model = {type(model.action_model).__name__}")
print(f"  future_action_window_size = {model.future_action_window_size}")
print(f"  past_action_window_size = {model.past_action_window_size}")

# ── Quick forward ──
image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
sample = {
    "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
    "image": [image],
    "lang": "Pick up the red block.",
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device = {device}")
model = model.to(device)

out = model([sample])
print(f"  action_loss = {out['action_loss'].item():.4f}")

# ── Predict action ──
pred = model.predict_action([sample])
print(f"  predicted action shape = {pred['normalized_actions'].shape}")

print("\n✅ QwenOFT GPU test PASSED")
