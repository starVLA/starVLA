"""Protocol tests for the policy-server smoke-test client.

Verifies that ``deployment/model_server/tools/debug_server_policy.py`` speaks the
contract that ``WebsocketClientPolicy`` and ``WebsocketPolicyServer`` actually
implement:

- the request is a single ``predict_action`` message carrying
  ``{"examples": [{"image": [...], "lang": ...}], "unnorm_key": ...}``,
- images default to the server's ``training_obs_image_size``,
- a policy-side failure, or a response without a usable action chunk, makes the
  smoke test *fail* rather than log success.

The policy is stubbed (no GPU / checkpoint needed); the websocket transport, the
msgpack codec and the server's routing run for real.
"""

import asyncio
import logging
import threading
import unittest

import numpy as np
import websockets.asyncio.server

from deployment.model_server.tools import debug_server_policy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

CHUNK = 8
ACTION_DIM = 7
TRAINING_IMAGE_SIZE = [96, 128]  # deliberately not the 224x224 fallback
UNNORM_KEY = "stub_mix"

METADATA = {
    "env": "starvla_policy_server",
    "ckpt_path": "/stub",
    "action_chunk_size": CHUNK,
    "available_unnorm_keys": [UNNORM_KEY],
    "default_unnorm_key": UNNORM_KEY,
    "training_obs_image_size": TRAINING_IMAGE_SIZE,
}


class _StubPolicy:
    """Duck-typed PolicyServerWrapper: records requests, returns a fixed chunk."""

    def __init__(self):
        self.requests = []
        self.raise_on_next = False
        self.response = None  # override the returned dict when set

    def predict_action(self, examples, unnorm_key=None, **kwargs):
        self.requests.append({"examples": examples, "unnorm_key": unnorm_key, "kwargs": kwargs})
        if self.raise_on_next:
            raise RuntimeError("stub policy failure")
        if self.response is not None:
            return self.response
        return {"actions": np.zeros((len(examples), CHUNK, ACTION_DIM), dtype=np.float32)}


class DebugServerPolicyTest(unittest.TestCase):
    """Drives the real `WebsocketPolicyServer._handler` over a real socket.

    The handler is served directly rather than through `serve_forever()`, which
    never returns, so the port is bound on 0 (no free-port race) and released at
    teardown. `tests/test_gr00t_zmq_compat_server.py` drives its ZMQ server the
    same way.
    """

    @classmethod
    def setUpClass(cls):
        cls.policy = _StubPolicy()
        server = WebsocketPolicyServer(cls.policy, host="127.0.0.1", port=0, metadata=METADATA)
        cls.loop = asyncio.new_event_loop()
        cls.ready = threading.Event()
        cls.thread = threading.Thread(target=cls._serve, args=(server,), daemon=True)
        cls.thread.start()
        if not cls.ready.wait(timeout=15):
            raise RuntimeError("stub policy server never started listening")

    @classmethod
    def _serve(cls, server):
        asyncio.set_event_loop(cls.loop)
        try:
            cls.loop.run_until_complete(cls._serve_until_stopped(server))
        finally:
            cls.loop.close()

    @classmethod
    async def _serve_until_stopped(cls, server):
        cls.stop = asyncio.Event()
        async with websockets.asyncio.server.serve(
            server._handler, "127.0.0.1", 0, compression=None, max_size=None
        ) as ws_server:
            cls.port = ws_server.sockets[0].getsockname()[1]
            cls.ready.set()
            await cls.stop.wait()

    @classmethod
    def tearDownClass(cls):
        cls.loop.call_soon_threadsafe(cls.stop.set)
        cls.thread.join(timeout=5)

    def setUp(self):
        self.policy.requests.clear()
        self.policy.raise_on_next = False
        self.policy.response = None

    def _run_smoke_test(self, *extra_argv):
        debug_server_policy._main(["--host", "127.0.0.1", "--port", str(self.port), *extra_argv])

    def test_smoke_test_sends_the_documented_payload(self):
        self._run_smoke_test("--num_images", "2", "--instruction", "pick up the red block")

        self.assertEqual(len(self.policy.requests), 1)
        request = self.policy.requests[0]
        self.assertEqual(request["unnorm_key"], UNNORM_KEY)

        examples = request["examples"]
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["lang"], "pick up the red block")
        images = examples[0]["image"]
        self.assertEqual(len(images), 2)
        for image in images:
            # Image size defaults to the server's training_obs_image_size.
            self.assertEqual(np.asarray(image).shape, (*TRAINING_IMAGE_SIZE, 3))
            self.assertEqual(np.asarray(image).dtype, np.uint8)

    def test_image_size_override_is_honoured(self):
        self._run_smoke_test("--image_size", "64", "80")

        images = self.policy.requests[0]["examples"][0]["image"]
        self.assertEqual(np.asarray(images[0]).shape, (64, 80, 3))

    def test_policy_failure_fails_the_smoke_test(self):
        self.policy.raise_on_next = True
        with self.assertRaisesRegex(RuntimeError, "stub policy failure"):
            self._run_smoke_test()

    def test_response_without_actions_fails_the_smoke_test(self):
        self.policy.response = {"normalized_actions": np.zeros((1, CHUNK, ACTION_DIM), dtype=np.float32)}
        with self.assertRaisesRegex(RuntimeError, "no 'actions'"):
            self._run_smoke_test()

    def test_response_with_a_mis_shaped_chunk_fails_the_smoke_test(self):
        self.policy.response = {"actions": np.zeros((CHUNK, ACTION_DIM), dtype=np.float32)}
        with self.assertRaisesRegex(RuntimeError, r"\(B, T, D\)"):
            self._run_smoke_test()

    def test_chunk_length_disagreeing_with_the_handshake_fails_the_smoke_test(self):
        # Eval clients index the cached chunk by the handshake's action_chunk_size,
        # so a disagreement breaks them at rollout time rather than here.
        self.policy.response = {"actions": np.zeros((1, CHUNK - 1, ACTION_DIM), dtype=np.float32)}
        with self.assertRaisesRegex(RuntimeError, "action_chunk_size"):
            self._run_smoke_test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    unittest.main()
