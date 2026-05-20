"""Training entry point for QwenGR00T_MoE_LoRA.

Pre-imports GTY MoE modules and the WMH MoE+LoRA framework registration before
delegating to starVLA.training.train_starvla.main().
"""

import os
import sys

_LOCAL_TRAIN_FILES = os.path.dirname(os.path.abspath(__file__))
if _LOCAL_TRAIN_FILES not in sys.path:
    sys.path.insert(0, _LOCAL_TRAIN_FILES)

_GTY_ROOT = os.environ.get(
    "GTY_ROOT",
    "/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/members/GTY",
)
_GTY_TRAIN_FILES = os.path.join(_GTY_ROOT, "train_files")
if _GTY_TRAIN_FILES not in sys.path:
    sys.path.insert(0, _GTY_TRAIN_FILES)

from moe_lora.qwen_gr00t_moe_lora import QwenGR00T_MoE_LoRA  # noqa: F401,E402

from starVLA.training.train_starvla import main as _train_main  # noqa: E402


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    _train_main(cfg)
