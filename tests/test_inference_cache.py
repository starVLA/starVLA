import unittest
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


def _load_inference_cache_module():
    module_path = Path(__file__).resolve().parents[1] / "starVLA" / "model" / "framework" / "inference_cache.py"
    spec = importlib.util.spec_from_file_location("starvla_inference_cache", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_inference_cache = _load_inference_cache_module()
SessionInferenceCache = _inference_cache.SessionInferenceCache
build_multimodal_cache_key = _inference_cache.build_multimodal_cache_key
estimate_cache_value_bytes = _inference_cache.estimate_cache_value_bytes


class _DummyPolicy:
    def __init__(self) -> None:
        self.clear_calls = []
        self.predict_calls = []

    def clear_inference_cache(self, session_id=None) -> None:
        self.clear_calls.append(session_id)

    def predict_action(self, **kwargs):
        self.predict_calls.append(kwargs)
        return {"normalized_actions": np.zeros((1, 1, 7), dtype=np.float32)}


class InferenceCacheTests(unittest.TestCase):
    def test_cache_key_changes_with_instruction_and_pixels(self):
        image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), mode="RGB")
        altered_image = Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8), mode="RGB")

        base_key = build_multimodal_cache_key(images=[[image]], instructions=["pick up block"])
        same_key = build_multimodal_cache_key(images=[[image.copy()]], instructions=["pick up block"])
        other_instruction_key = build_multimodal_cache_key(images=[[image]], instructions=["open drawer"])
        other_image_key = build_multimodal_cache_key(images=[[altered_image]], instructions=["pick up block"])

        self.assertEqual(base_key, same_key)
        self.assertNotEqual(base_key, other_instruction_key)
        self.assertNotEqual(base_key, other_image_key)

    def test_session_cache_tracks_hits_and_misses_per_session(self):
        cache = SessionInferenceCache()
        cache.put("session-a", "key-a", np.zeros((2, 3), dtype=np.float32))

        cached_value, hit = cache.get("session-a", "key-a")
        self.assertTrue(hit)
        self.assertEqual((2, 3), cached_value.shape)

        _, hit = cache.get("session-a", "key-b")
        self.assertFalse(hit)
        cache.put("session-b", "key-b", np.zeros((1, 2), dtype=np.float16))
        _, hit = cache.get("session-b", "key-b")
        self.assertTrue(hit)

        self.assertEqual(
            {"hits": 1, "misses": 1, "cache_entries": 1, "cache_bytes": 24},
            cache.stats("session-a"),
        )
        self.assertEqual(
            {"hits": 1, "misses": 0, "cache_entries": 1, "cache_bytes": 4},
            cache.stats("session-b"),
        )

    def test_clear_resets_session_entry_and_stats(self):
        cache = SessionInferenceCache()
        cache.put("session-a", "key-a", np.zeros((2, 3), dtype=np.float32))
        cache.get("session-a", "key-a")
        cache.get("session-a", "missing")

        cache.clear("session-a")

        self.assertEqual({"hits": 0, "misses": 0, "cache_entries": 0, "cache_bytes": 0}, cache.stats("session-a"))
        _, hit = cache.get("session-a", "key-a")
        self.assertFalse(hit)
        self.assertEqual({"hits": 0, "misses": 1, "cache_entries": 0, "cache_bytes": 0}, cache.stats("session-a"))

    def test_estimate_cache_value_bytes_handles_nested_tensors(self):
        nested_value = {
            "a": np.zeros((2, 2), dtype=np.float32),
            "b": [np.zeros((1,), dtype=np.float16), np.zeros((3,), dtype=np.uint8)],
        }

        self.assertEqual(16 + 2 + 3, estimate_cache_value_bytes(nested_value))

    def test_websocket_server_reset_and_infer_are_session_scoped(self):
        policy = _DummyPolicy()
        server = WebsocketPolicyServer(policy=policy, metadata={"env": "test"})

        reset_response = server._route_message({"type": "reset", "request_id": "r1"}, session_id="session-a")
        self.assertTrue(reset_response["ok"])
        self.assertEqual(["session-a"], policy.clear_calls)

        infer_response = server._route_message(
            {
                "type": "infer",
                "request_id": "r2",
                "payload": {"examples": [{"lang": "pick", "image": []}], "use_vlm_cache": True},
            },
            session_id="session-a",
        )
        self.assertTrue(infer_response["ok"])
        self.assertEqual("session-a", policy.predict_calls[-1]["cache_session_id"])
        self.assertTrue(policy.predict_calls[-1]["use_vlm_cache"])
        self.assertIn("metrics", infer_response)
        self.assertGreaterEqual(infer_response["metrics"]["predict_action_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
