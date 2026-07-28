# EgoVLA (VILA) backbone pieces vendored into starVLA.
from .projector import EgoVLAMMProjector, DownSampleBlock

# LLaVA / VILA multimodal token constants (EgoVLA `llava/constants.py`).
IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"

__all__ = ["EgoVLAMMProjector", "DownSampleBlock", "IMAGE_TOKEN_INDEX", "IGNORE_INDEX", "DEFAULT_IMAGE_TOKEN"]
