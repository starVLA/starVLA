# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""MetaWorld env-side adapter (thin client).

Mirrors the LIBERO ``ModelClient`` interface but tailored for MetaWorld MT50:
  - Single camera view (``corner2``, preprocessed: ROT180 + center-crop + resize 224).
  - 4-D continuous action (xyz + gripper), no gripper binarization.
  - Action chunk caching: refreshed every ``action_chunk_size`` steps.

The websocket *server* returns already-unnormalized actions (see
``deployment/model_server/policy_wrapper.py``).
"""

from typing import Optional, Sequence

import cv2
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


# Image preprocessing constants — must match MetaWorld training data pipeline.
CAMERA_NAME = "corner2"
APPLY_ROT_180 = True
APPLY_CENTER_CROP = True
CROP_KEEP_RATIO = 2 / 3
IMG_SIZE = (224, 224)


def preprocess_metaworld_image(rgb: np.ndarray) -> np.ndarray:
    """Render output -> ROT180 -> center_crop(2/3) -> resize 224x224 -> RGB uint8 HWC."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

    if APPLY_ROT_180:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)

    if APPLY_CENTER_CROP and (0.0 < CROP_KEEP_RATIO < 1.0):
        h, w = rgb.shape[:2]
        new_h = max(1, int(round(h * CROP_KEEP_RATIO)))
        new_w = max(1, int(round(w * CROP_KEEP_RATIO)))
        y0 = (h - new_h) // 2
        x0 = (w - new_w) // 2
        rgb = rgb[y0 : y0 + new_h, x0 : x0 + new_w, :].copy()

    if IMG_SIZE is not None:
        rgb = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)

    return np.ascontiguousarray(rgb)


class ModelClient:
    """MetaWorld evaluation client that talks to the starVLA policy server."""

    def __init__(
        self,
        unnorm_key: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 10095,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
    ) -> None:
        # Connect & receive handshake metadata (action_chunk_size, etc.)
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self._server_metadata = meta

        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps

        print(
            f"*** MetaWorld client: unnorm_key={unnorm_key}, "
            f"action_chunk_size={self.action_chunk_size}, "
            f"server_meta={meta} ***"
        )

        self.task_description: Optional[str] = None
        # Cached unnormalized action chunk; refreshed every `action_chunk_size` steps.
        self.raw_actions: Optional[np.ndarray] = None

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.raw_actions = None

    def step(self, image: np.ndarray, prompt: str, step: int = 0) -> np.ndarray:
        """One env step.

        Args:
            image: Preprocessed RGB uint8 HWC image (224x224x3).
            prompt: Task language instruction.
            step: Env step counter; used for chunk caching.

        Returns:
            4-D action ``np.ndarray`` of shape ``(4,)`` (xyz + gripper).
        """
        if prompt != self.task_description:
            self.reset(prompt)

        # Refresh chunk if needed.
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            example = {"image": [image], "lang": prompt}
            vla_input = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                actions_batch = response["data"]["actions"]  # (B, T, D)
            except KeyError:
                raise KeyError(
                    f"Key 'actions' not found in response: "
                    f"keys={list(response.get('data', {}).keys())}"
                )
            self.raw_actions = np.asarray(actions_batch)[0]  # (T, D)

        action = self.raw_actions[step % self.action_chunk_size]
        return np.asarray(action[:4], dtype=np.float32)
