"""CPU-only tests for RoboTwin receding-horizon action execution."""

import unittest
from unittest import mock

import numpy as np

from examples.simBenchmarks.Robotwin.eval_files import model2robotwin_interface as robotwin


class _FakeClient:
    """Return identifiable 50-step chunks and record policy requests."""

    def __init__(self, *args, **kwargs):
        self.requests = []

    def get_server_metadata(self) -> dict:
        return {"action_chunk_size": 50}

    def predict_action(self, vla_input: dict) -> dict:
        chunk_id = len(self.requests)
        self.requests.append(vla_input)
        steps = np.arange(50, dtype=np.float32) + chunk_id * 100
        actions = np.repeat(steps[:, None], 14, axis=1)
        return {"data": {"actions": actions[None].tolist()}}


def _example() -> dict:
    return {
        "lang": "pick up the object",
        "image": [np.zeros((8, 8, 3), dtype=np.uint8)],
        "state": np.zeros(14, dtype=np.float32),
    }


class RobotwinExecutionHorizonTest(unittest.TestCase):
    def _make_client(self, execution_horizon=None) -> "robotwin.ModelClient":
        with mock.patch.object(robotwin, "WebsocketClientPolicy", _FakeClient):
            client = robotwin.ModelClient(policy_ckpt_path="unused", execution_horizon=execution_horizon)
        client._resize_image = lambda image: image
        return client

    def test_replans_after_configured_execution_horizon(self):
        client = self._make_client(execution_horizon=8)

        markers = [client.step(_example(), step=step)[0] for step in range(17)]

        self.assertEqual(len(client.client.requests), 3)
        np.testing.assert_array_equal(markers[:8], np.arange(8))
        np.testing.assert_array_equal(markers[8:16], np.arange(100, 108))
        self.assertEqual(markers[16], 200)

    def test_default_executes_the_full_prediction_chunk(self):
        client = self._make_client()

        first = client.step(_example(), step=0)[0]
        last = client.step(_example(), step=49)[0]
        next_chunk = client.step(_example(), step=50)[0]

        self.assertEqual(client.execution_horizon, 50)
        self.assertEqual(len(client.client.requests), 2)
        self.assertEqual((first, last, next_chunk), (0, 49, 100))

    def test_invalid_execution_horizons_are_rejected(self):
        for execution_horizon in (0, 51, True):
            with self.subTest(execution_horizon=execution_horizon):
                with self.assertRaisesRegex(ValueError, "execution_horizon"):
                    self._make_client(execution_horizon=execution_horizon)

    def test_get_model_forwards_execution_horizon(self):
        with mock.patch.object(robotwin, "ModelClient") as model_client:
            robotwin.get_model({"policy_ckpt_path": "checkpoint.pt", "execution_horizon": 16})

        self.assertEqual(model_client.call_args.kwargs["execution_horizon"], 16)

    def test_reset_replans_from_the_start_of_a_new_chunk(self):
        client = self._make_client(execution_horizon=8)
        client.step(_example(), step=3)

        client.reset("pick up the object")
        first_after_reset = client.step(_example(), step=4)[0]

        self.assertEqual(len(client.client.requests), 2)
        self.assertEqual(first_after_reset, 100)


if __name__ == "__main__":
    unittest.main()
