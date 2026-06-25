# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
MiniCPM-GR00T Framework
Direct port of QwenGR00T to the MiniCPM-V 4.6 backbone (`openbmb/MiniCPM-V-4.6`).

See `MiniCPMPI.py` for the rationale: VLM swap is handled by the dispatcher in
`starVLA/model/modules/vlm/__init__.py`, so this file just re-registers under a new
framework name. Override forward / predict_action here only if MiniCPM-V needs
GR00T-specific surgery.
"""
from typing import Optional

from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("MiniCPMGR00T")
class MiniCPM_GR00T(Qwen_GR00T):
    """
    MiniCPM-V 4.6 + last-hidden-state cross-attention DiT (GR00T head).

    QwenGR00T overwrites `cross_attention_dim` from
    `qwen_vl_interface.model.config.hidden_size`, which MiniCPM-V exposes as 1024.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        backbone_hidden = self.qwen_vl_interface.model.config.hidden_size
        assert backbone_hidden == 1024, (
            f"[MiniCPMGR00T] unexpected backbone hidden_size={backbone_hidden}; "
            "check `framework.qwenvl.base_vlm` and DiT cross_attention_dim alignment."
        )


if __name__ == "__main__":
    import argparse

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="openbmb/MiniCPM-V-4.6",
    )
    parser.add_argument("--attn", type=str, default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "MiniCPMGR00T",
                "qwenvl": {
                    "base_vlm": args.model_id,
                    "attn_implementation": args.attn,
                    "vl_hidden_dim": 1024,
                },
                "dino": {"dino_backbone": "dinov2_vits14"},
                "action_model": {
                    "action_model_type": "DiT-B",
                    "action_hidden_dim": 1024,
                    "hidden_size": 1024,
                    "add_pos_embed": True,
                    "max_seq_len": 1024,
                    "action_dim": 7,
                    "state_dim": 7,
                    "future_action_window_size": 7,
                    "action_horizon": 8,
                    "past_action_window_size": 0,
                    "repeated_diffusion_steps": 8,
                    "noise_beta_alpha": 1.5,
                    "noise_beta_beta": 1.0,
                    "noise_s": 0.999,
                    "num_timestep_buckets": 1000,
                    "num_inference_timesteps": 4,
                    "num_target_vision_tokens": 32,
                    "diffusion_model_cfg": {
                        "cross_attention_dim": 1024,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
                "reduce_in_full_precision": True,
            },
            "trainer": {"repeated_diffusion_steps": 2},
            "datasets": {"vla_data": {"obs_image_size": None}},
        }
    )

    model = MiniCPM_GR00T(cfg)
    print(f"[MiniCPMGR00T] backbone hidden_size = {model.qwen_vl_interface.model.config.hidden_size}")

    img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [img],
        "lang": "Test instruction for MiniCPM-GR00T smoke test.",
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        forward_output = model([sample])
        print(f"[MiniCPMGR00T] action_loss = {forward_output['action_loss'].item():.6f}")

        predict_output = model.predict_action(examples=[sample])
        actions = predict_output["normalized_actions"]
        print(f"[MiniCPMGR00T] predicted actions shape = {actions.shape}")
    print("[MiniCPMGR00T] OK")
