# Copyright 2026 starVLA / gemma-vla community.
# Licensed under the MIT License.
"""
Gemma4-PI Framework
A direct port of QwenPI to the Gemma 4 VLM backbone (`google/gemma-4-E2B-it`).

The actual VLM swap happens in `starVLA/model/modules/vlm/__init__.py::get_vlm_model`,
which routes any `framework.qwenvl.base_vlm` containing "gemma-4" to
`_Gemma4_VL_Interface`. Because that interface mirrors `_QWen3_VL_Interface`, the body
of this framework is identical to QwenPI — we simply re-register under a new name so
configs can ask for `framework.name=Gemma4PI`.

If Gemma 4 ever needs framework-level surgery (extra projector, custom hidden-state
slicing, etc.), override `forward` / `predict_action` here rather than touching QwenPI.
"""
from typing import Optional

from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.tools import FRAMEWORK_REGISTRY


@FRAMEWORK_REGISTRY.register("Gemma4PI")
class Gemma4_PI(Qwen_PI):
    """
    Gemma 4 + layer-wise flow-matching DiT action head.

    Notes:
        - DiT `cross_attention_dim` must equal Gemma 4 E2B's text_config.hidden_size = 1536.
          The constructor of `Qwen_PI` reads `qwen_vl_interface.model.config.hidden_size`,
          which `_Gemma4_VL_Interface` aligns to 1536 at load time, so the existing
          alignment logic in QwenPI works without modification.
        - PI consumes the last N hidden states for layer-wise cross-attention. Gemma 4 has
          35 text layers, so any reasonable DiT block count (≤35) fits.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        # Hard assertion against the documented Gemma4-E2B hidden size to fail loud
        # if anyone wires in a wrong checkpoint or a future variant.
        backbone_hidden = self.qwen_vl_interface.model.config.hidden_size
        assert backbone_hidden in (1536, 2048, 2560), (
            f"[Gemma4PI] unexpected backbone hidden_size={backbone_hidden}; "
            f"check `framework.qwenvl.base_vlm` and DiT cross_attention_dim alignment."
        )


if __name__ == "__main__":
    import argparse
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-4-E2B-it",
    )
    parser.add_argument("--attn", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "Gemma4PI",
                "qwenvl": {
                    "base_vlm": args.model_id,
                    "attn_implementation": args.attn,
                    "drop_audio_tower": True,
                },
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
                        "cross_attention_dim": 1536,
                        "dropout": 0.2,
                        "final_dropout": True,
                        "interleave_self_attention": True,
                        "norm_type": "ada_norm",
                        "num_layers": 16,
                        "output_dim": 1024,
                        "positional_embeddings": None,
                    },
                },
            },
            "trainer": {"repeated_diffusion_steps": 2},
            "datasets": {"vla_data": {"image_size": None}},
        }
    )

    model = Gemma4_PI(cfg)
    print(model)
    print(f"[Gemma4PI] backbone hidden_size = {model.qwen_vl_interface.model.config.hidden_size}")

    # Tiny inputs so the smoke test fits on a single A6000 (49GB) without gradient checkpointing.
    img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [img],  # one view, not two
        "lang": "Test instruction for Gemma4-PI smoke test.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    batch = [sample]  # batch size 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Smoke test only; no gradients needed.
    with torch.no_grad():
        forward_output = model(batch)
        print(f"[Gemma4PI] action_loss = {forward_output['action_loss'].item():.6f}")

        predict_output = model.predict_action([sample])
        actions = predict_output["normalized_actions"]
        print(f"[Gemma4PI] predicted actions shape = {actions.shape}")
    print("[Gemma4PI] OK")
