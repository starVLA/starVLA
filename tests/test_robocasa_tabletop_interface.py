"""Tests for the Robocasa_tabletop eval client (PolicyWarper).

Covers the state-input toggle and request plumbing from
``examples/Robocasa_tabletop/eval_files/model2robocasa_interface.py``:

- ``send_state=False`` must omit the ``state`` key from the request so that
  prompt-augmenting frameworks (e.g. QwenOFT) see the same prompt format as
  during state-less training (issue #355).
- The default keeps the current behavior (sin/cos-encoded state included),
  which the Qwen3-VL-GR00T checkpoint relies on.

These tests mock the websocket client only; no policy server or robocasa
simulator is required.
"""

import unittest
from unittest import mock

import numpy as np

from examples.Robocasa_tabletop.eval_files import model2robocasa_interface as m2r


def _fake_observations(batch_size: int = 1, image_hw: tuple = (256, 256)) -> dict:
    """Build observations shaped like GrootRoboCasaEnv GR1 output."""
    h, w = image_hw
    return {
        "annotation.human.coarse_action": ("pick the milk",) * batch_size,
        "video.ego_view": np.zeros((batch_size, 1, h, w, 3), dtype=np.uint8),
        "state.left_arm": np.zeros((batch_size, 1, 7), dtype=np.float32),
        "state.right_arm": np.zeros((batch_size, 1, 7), dtype=np.float32),
        "state.left_hand": np.zeros((batch_size, 1, 6), dtype=np.float32),
        "state.right_hand": np.zeros((batch_size, 1, 6), dtype=np.float32),
        "state.waist": np.zeros((batch_size, 1, 3), dtype=np.float32),
    }


class _FakeClient:
    """Stands in for WebsocketClientPolicy and records every request."""

    def __init__(self, *args, **kwargs):
        self.requests = []

    def get_server_metadata(self) -> dict:
        return {"env": "test"}

    def predict_action(self, vla_input: dict) -> dict:
        self.requests.append(vla_input)
        batch = len(vla_input["examples"])
        return {"data": {"actions": np.zeros((batch, 16, 29)).tolist()}}


class PolicyWarperRequestTest(unittest.TestCase):
    def _make_warper(self, **kwargs) -> "m2r.PolicyWarper":
        with mock.patch.object(m2r, "WebsocketClientPolicy", _FakeClient):
            return m2r.PolicyWarper(policy_ckpt_path="unused", **kwargs)

    def test_send_state_false_omits_state_key(self):
        warper = self._make_warper(send_state=False)
        warper.step(_fake_observations())
        example = warper.client.requests[-1]["examples"][0]
        self.assertNotIn("state", example)

    def test_send_state_default_keeps_sincos_state(self):
        warper = self._make_warper()
        warper.step(_fake_observations())
        example = warper.client.requests[-1]["examples"][0]
        self.assertIn("state", example)
        self.assertEqual(np.asarray(example["state"]).shape, (1, 58))

    def test_unnorm_key_forwarded_in_request(self):
        warper = self._make_warper(unnorm_key="gr1", send_state=False)
        warper.step(_fake_observations())
        self.assertEqual(warper.client.requests[-1]["unnorm_key"], "gr1")

    def test_images_resized_to_224(self):
        warper = self._make_warper(send_state=False)
        warper.step(_fake_observations(image_hw=(256, 256)))
        example = warper.client.requests[-1]["examples"][0]
        self.assertEqual(np.asarray(example["image"][0]).shape[:2], (224, 224))


if __name__ == "__main__":
    unittest.main()
