"""Protocol tests for the GR00T ZMQ compatibility server.

Verifies that ``deployment/model_server/server_policy_gr00t_zmq.py``'s building
blocks (``Gr00tCompatPolicy`` + ``ZmqGr00tPolicyServer``) are wire-compatible
with the Isaac-GR00T N1.6 ``PolicyClient`` contract — specifically the client
embedded in GR00T-WBC-Bridge's ``gr00t_n1_wbc_bridge_deploy.py`` (default
``--server_codec custom``):

- msgpack + ``__ndarray_class__``/``as_npy`` ndarray codec,
- ``{"endpoint": ..., "data": {"observation": ..., "options": None}}`` requests,
- ``get_action`` returning ``(action_dict, info)`` where each action value is
  shaped ``(1, T, dim)`` and the bridge consumes ``actions[key][0]``,
- errors reported as ``{"error": ...}`` without wedging the REQ/REP loop.

The policy wrapper is stubbed (no GPU / checkpoint needed); the ZMQ transport
and the observation/action conversion run for real.
"""

import io
import threading
import unittest

import msgpack
import numpy as np
import zmq

from deployment.model_server.gr00t_obs_adapter import Gr00tCompatPolicy
from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer


# ---------------------------------------------------------------------------
# Bridge-side codec, copied verbatim from gr00t_n1_wbc_bridge_deploy.py so the
# test exercises the exact bytes the bridge would send/receive.
# ---------------------------------------------------------------------------

def _bridge_encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    raise TypeError(f"Cannot encode type {type(obj)}")


def _bridge_decode(obj):
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


class BridgeLikeClient:
    """Minimal replica of GR00TN1Client (the WBC bridge's ZMQ client)."""

    def __init__(self, port: int):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, 5000)
        self._sock.connect(f"tcp://127.0.0.1:{port}")

    def call(self, endpoint: str, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._sock.send(msgpack.packb(request, default=_bridge_encode, strict_types=False))
        return msgpack.unpackb(self._sock.recv(), object_hook=_bridge_decode, raw=False)

    def get_action(self, observation: dict):
        result = self.call("get_action", {"observation": observation, "options": None})
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])
        return result[0] if isinstance(result, list) else result

    def close(self):
        self._sock.close(linger=0)
        self._ctx.term()


# ---------------------------------------------------------------------------
# Stubbed starVLA wrapper with the decoupled-WBC G1 contract (dex3 hands).
# ---------------------------------------------------------------------------

STATE_KEYS = [
    "state.left_leg", "state.right_leg", "state.waist",
    "state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand",
]
STATE_DIMS = {
    "state.left_leg": 6, "state.right_leg": 6, "state.waist": 3,
    "state.left_arm": 7, "state.right_arm": 7,
    "state.left_hand": 7, "state.right_hand": 7,
}
ACTION_KEYS = [
    "action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand",
    "action.waist", "action.base_height_command", "action.navigate_command",
]
ACTION_DIMS = {
    "action.left_arm": 7, "action.right_arm": 7,
    "action.left_hand": 7, "action.right_hand": 7,
    "action.waist": 3, "action.base_height_command": 1, "action.navigate_command": 3,
}
ACTION_DIM = sum(ACTION_DIMS.values())  # 35
CHUNK = 30


class _StubProcessor:
    unnorm_key = "unitree_g1_wbc"
    state_keys = STATE_KEYS
    state_key_dims = STATE_DIMS
    action_keys = ACTION_KEYS
    action_key_dims = ACTION_DIMS


class _StubWrapper:
    """Duck-typed PolicyServerWrapper: records requests, returns a chunk whose
    flat dim d holds the constant value d (so the split is verifiable)."""

    def __init__(self):
        self.requests = []

    def get_norm_processor(self, unnorm_key=None):
        return _StubProcessor()

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        self.requests.append({"examples": examples, "unnorm_key": unnorm_key})
        chunk = np.tile(np.arange(ACTION_DIM, dtype=np.float32), (1, CHUNK, 1))
        return {"actions": chunk}


def _bridge_observation(n_hist: int = 1) -> dict:
    """Observation shaped exactly like GR00TBridgeNode._build_obs output."""
    state = {}
    for i, full_key in enumerate(STATE_KEYS):
        subkey = full_key.split(".", 1)[-1]
        dim = STATE_DIMS[full_key]
        # History frames hold garbage (-1); ONLY the latest frame holds the
        # group's marker value (100 + i), so ordering AND frame selection are
        # both asserted by the flattened result.
        arr = np.full((1, n_hist, dim), -1.0, dtype=np.float32)
        arr[0, -1, :] = 100.0 + i
        state[subkey] = arr
    video = np.zeros((1, 1, 224, 224, 3), dtype=np.uint8)
    video[0, 0, 0, 0, 0] = 42
    return {
        "video": {"ego_view": video},
        "state": state,
        "language": {"annotation.human.task_description": [["pick up the apple"]]},
    }


class Gr00tZmqCompatServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wrapper = _StubWrapper()
        policy = Gr00tCompatPolicy(cls.wrapper, fallback_instruction="noop")
        cls.server = ZmqGr00tPolicyServer.__new__(ZmqGr00tPolicyServer)
        # Bind to a random free port (bypass fixed-port __init__ binding).
        cls.server._policy = policy
        cls.server.running = False
        cls.server._ctx = zmq.Context()
        cls.server._sock = cls.server._ctx.socket(zmq.REP)
        cls.port = cls.server._sock.bind_to_random_port("tcp://127.0.0.1")
        from deployment.model_server.tools.zmq_policy_server import EndpointHandler

        cls.server._endpoints = {
            "ping": EndpointHandler(cls.server._ping, requires_input=False),
            "kill": EndpointHandler(cls.server._kill, requires_input=False),
            "get_action": EndpointHandler(policy.get_action),
            "reset": EndpointHandler(policy.reset),
            "get_modality_config": EndpointHandler(policy.get_modality_config, requires_input=False),
        }
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        cls.client = BridgeLikeClient(cls.port)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.client.call("kill")
        finally:
            cls.client.close()
            cls.thread.join(timeout=5)

    def test_ping_and_reset(self):
        self.assertEqual(self.client.call("ping").get("status"), "ok")
        self.assertTrue(self.client.call("reset", {"options": None}).get("ok"))

    def test_modality_config_reports_contract(self):
        contract = self.client.call("get_modality_config")
        self.assertEqual(contract["action_keys"], ACTION_KEYS)
        self.assertEqual(contract["state_keys"], STATE_KEYS)

    def test_get_action_matches_bridge_contract(self):
        actions = self.client.get_action(_bridge_observation(n_hist=2))

        # Every key the bridge's n16_unitree_g1 profile requires, at (1, T, dim);
        # the bridge consumes actions[key][0] -> (T, dim).
        expected_dims = {k.split(".", 1)[-1]: d for k, d in ACTION_DIMS.items()}
        cursor = 0
        for full_key in ACTION_KEYS:
            subkey = full_key.split(".", 1)[-1]
            dim = expected_dims[subkey]
            self.assertIn(subkey, actions)
            per_step = np.asarray(actions[subkey][0], dtype=np.float32)
            self.assertEqual(per_step.shape, (CHUNK, dim))
            # Flat dim d carries value d -> verifies split offsets.
            np.testing.assert_array_equal(per_step[0], np.arange(cursor, cursor + dim))
            cursor += dim

        # Server-side example the framework saw: flat state in DataConfig
        # order, built from the LATEST history frame of each group.
        example = self.wrapper.requests[-1]["examples"][0]
        expected_state = np.concatenate(
            [np.full(STATE_DIMS[k], 100.0 + i, dtype=np.float32) for i, k in enumerate(STATE_KEYS)]
        )
        np.testing.assert_array_equal(example["state"][0], expected_state)
        self.assertEqual(example["state"].shape, (1, expected_state.size))
        self.assertEqual(example["lang"], "pick up the apple")
        self.assertEqual(len(example["image"]), 1)
        self.assertEqual(example["image"][0].shape, (224, 224, 3))
        self.assertEqual(example["image"][0][0, 0, 0], 42)

    def test_missing_state_key_is_reported_not_fatal(self):
        obs = _bridge_observation()
        del obs["state"]["waist"]
        with self.assertRaisesRegex(RuntimeError, "waist"):
            self.client.get_action(obs)
        # REQ/REP loop must survive the error.
        self.assertEqual(self.client.call("ping").get("status"), "ok")


if __name__ == "__main__":
    unittest.main()
