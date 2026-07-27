#!/usr/bin/env python
"""Export a starVLA MiniCPMRobotManip fine-tune checkpoint into an HF model dir.

The trainer saves a *framework* state_dict whose keys are prefixed with
``model.`` (e.g. ``model.vlm.*`` / ``model.action_head.*``). The released
``MiniCPMV_VLA`` class (loaded via ``AutoModel.from_pretrained(..., trust_remote_code=True)``)
expects keys without that prefix (``vlm.*`` / ``action_head.*``).

This script strips the prefix and writes a self-contained HF directory
(released code/config/tokenizer + the fine-tuned ``model.safetensors``) that any
evaluation harness can load directly:

    python export_checkpoint.py \
        --ckpt playground/Checkpoints/mrm_libero_ft8/checkpoints/steps_2000_pytorch_model.pt \
        --base /path/to/MiniCPM-RobotManip \
        --out  playground/Exported/mrm_libero_ft8_step2000
"""

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

_COPY_FILES = (
    "config.json",
    "configuration_minicpm_vla.py",
    "modeling_minicpm_vla.py",
    "action_head.py",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="framework .pt/.safetensors checkpoint")
    ap.add_argument("--base", required=True, help="released MiniCPM-RobotManip dir (code + config + tokenizer)")
    ap.add_argument("--out", required=True, help="output HF model dir")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) copy released code/config/tokenizer
    base = Path(args.base)
    for f in _COPY_FILES:
        src = base / f
        if src.exists():
            shutil.copy2(src, out / f)

    # 2) load framework state_dict, strip the leading "model." prefix
    ckpt = Path(args.ckpt)
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file

        sd = load_file(str(ckpt))
    else:
        sd = torch.load(str(ckpt), map_location="cpu")
    sd = sd.get("state_dict", sd)

    new_sd, dropped = {}, 0
    for k, v in sd.items():
        nk = k[len("model.") :] if k.startswith("model.") else k
        if nk.startswith(("vlm.", "action_head.")):
            new_sd[nk] = v.contiguous()
        else:
            dropped += 1
    print(f"kept {len(new_sd)} tensors, dropped {dropped} non-model keys")

    save_file(new_sd, str(out / "model.safetensors"), metadata={"format": "pt"})
    print(f"exported HF model dir -> {out}")
    print("load with: AutoModel.from_pretrained(out, trust_remote_code=True)")


if __name__ == "__main__":
    main()
