"""Tests for LIBERO eval client request shaping."""

import unittest
from unittest import mock
import sys
import types
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
examples_pkg = types.ModuleType("examples")
examples_pkg.__path__ = [str(repo_root / "examples")]
sys.modules["examples"] = examples_pkg

from examples.simBenchmarks.LIBERO.eval_files import model2libero_interface as m2l


class _FakeClient:
    metadata = {}
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.requests = []
        _FakeClient.last_instance = self

    def get_server_metadata(self) -> dict:
        return dict(self.metadata)

    def predict_action(self, vla_input: dict) -> dict:
        self.requests.append(vla_input)
        return {"data": {"actions": np.zeros((1, 8, 7), dtype=np.float32)}}


def _make_example() -> dict:
    primary = np.full((32, 32, 3), 11, dtype=np.uint8)
    wrist = np.full((32, 32, 3), 22, dtype=np.uint8)
    return {"image": [primary, wrist], "lang": "put the bowl on the plate"}


class LiberoModelClientRequestTest(unittest.TestCase):
    def _make_client(self, metadata: dict) -> m2l.ModelClient:
        _FakeClient.metadata = metadata
        with mock.patch.object(m2l, "WebsocketClientPolicy", _FakeClient):
            return m2l.ModelClient(action_ensemble=False, image_size=(32, 32))

    def test_single_view_video_metadata_sends_primary_image_only(self):
        client = self._make_client({"action_chunk_size": 8, "vla_video_keys": ["video.primary_image"]})

        client.step(_make_example(), step=0)

        sent_images = _FakeClient.last_instance.requests[-1]["examples"][0]["image"]
        self.assertEqual(len(sent_images), 1)
        np.testing.assert_array_equal(sent_images[0], _make_example()["image"][0])

    def test_two_view_video_metadata_preserves_requested_order(self):
        client = self._make_client(
            {"action_chunk_size": 8, "vla_video_keys": ["video.wrist_image", "video.primary_image"]}
        )

        client.step(_make_example(), step=0)

        sent_images = _FakeClient.last_instance.requests[-1]["examples"][0]["image"]
        self.assertEqual(len(sent_images), 2)

        np.testing.assert_array_equal(sent_images[0], _make_example()["image"][1])
        np.testing.assert_array_equal(sent_images[1], _make_example()["image"][0])

    def test_video_metadata_takes_precedence_over_raw_vla_data_obs(self):
        client = self._make_client(
            {
                "action_chunk_size": 8,
                "vla_video_keys": ["video.primary_image", "video.wrist_image"],
                "vla_data_obs": ["image_0"],
            }
        )

        client.step(_make_example(), step=0)

        sent_images = _FakeClient.last_instance.requests[-1]["examples"][0]["image"]
        self.assertEqual(len(sent_images), 2)
        np.testing.assert_array_equal(sent_images[0], _make_example()["image"][0])
        np.testing.assert_array_equal(sent_images[1], _make_example()["image"][1])

    def test_raw_vla_data_obs_fallback_still_supports_single_view_metadata(self):
        client = self._make_client({"action_chunk_size": 8, "vla_data_obs": ["image_0"]})

        client.step(_make_example(), step=0)

        sent_images = _FakeClient.last_instance.requests[-1]["examples"][0]["image"]
        self.assertEqual(len(sent_images), 1)
        np.testing.assert_array_equal(sent_images[0], _make_example()["image"][0])

    def test_missing_checkpoint_metadata_keeps_existing_two_view_behavior(self):
        client = self._make_client({"action_chunk_size": 8})

        client.step(_make_example(), step=0)

        sent_images = _FakeClient.last_instance.requests[-1]["examples"][0]["image"]
        self.assertEqual(len(sent_images), 2)


if __name__ == "__main__":
    unittest.main()
