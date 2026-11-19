import contextlib
import threading
import unittest
from unittest import mock

import numpy as np
import websockets.exceptions
import websockets.sync.server

from deployment.model_server.tools import msgpack_numpy, websocket_policy_client
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


@contextlib.contextmanager
def local_server(handler):
    with websockets.sync.server.serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.socket.getsockname()[1]
        finally:
            server.shutdown()
            thread.join(timeout=5)


def waiting_client(port):
    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._uri = f"ws://127.0.0.1:{port}"
    client._api_key = None
    return client


class WebsocketPolicyClientTest(unittest.TestCase):
    def test_metadata_wait_obeys_startup_timeout_and_closes_connection(self):
        release_metadata = threading.Event()
        disconnected = threading.Event()

        def handler(connection):
            try:
                # Release eventually even on the buggy client, so the regression cannot hang.
                release_metadata.wait(timeout=1)
                connection.send(msgpack_numpy.packb({"ready": True}))
                connection.recv(timeout=2)
            except websockets.exceptions.ConnectionClosed:
                disconnected.set()
            except TimeoutError:
                pass

        with local_server(handler) as port:
            client = waiting_client(port)
            connection = None
            try:
                with self.assertRaises(TimeoutError):
                    connection, _ = client._wait_for_server(timeout=0.1)
            finally:
                release_metadata.set()
                if connection is not None:
                    connection.close()
            self.assertTrue(disconnected.wait(timeout=2))

    def test_connection_attempt_respects_remaining_budget(self):
        now = [0.0]

        def advance(seconds):
            now[0] += seconds

        with (
            mock.patch.object(websocket_policy_client.time, "time", side_effect=lambda: now[0]),
            mock.patch.object(websocket_policy_client.time, "monotonic", side_effect=lambda: now[0]),
            mock.patch.object(websocket_policy_client.time, "sleep", side_effect=advance) as sleep,
            mock.patch.object(
                websocket_policy_client.websockets.sync.client,
                "connect",
                side_effect=ConnectionRefusedError,
            ) as connect,
        ):
            with self.assertRaises(TimeoutError):
                waiting_client(1)._wait_for_server(timeout=0.25)

        self.assertLessEqual(connect.call_args.kwargs["open_timeout"], 0.25)
        self.assertLessEqual(sleep.call_args.args[0], 0.25)

    def test_metadata_uses_budget_remaining_after_handshake(self):
        now = [0.0]
        connection = mock.Mock()
        connection.recv.return_value = msgpack_numpy.packb({"ready": True})

        def connect(*args, **kwargs):
            now[0] += 0.75
            return connection

        with (
            mock.patch.object(websocket_policy_client.time, "monotonic", side_effect=lambda: now[0]),
            mock.patch.object(websocket_policy_client.websockets.sync.client, "connect", side_effect=connect),
        ):
            received_connection, metadata = waiting_client(1)._wait_for_server(timeout=1)

        connection.recv.assert_called_once_with(timeout=0.25)
        connection.close.assert_not_called()
        self.assertIs(received_connection, connection)
        self.assertEqual(metadata, {"ready": True})

    def test_invalid_metadata_closes_connection_and_preserves_error(self):
        connection = mock.Mock()
        connection.recv.return_value = b"\xc1"  # Reserved/invalid msgpack byte.
        with mock.patch.object(websocket_policy_client.websockets.sync.client, "connect", return_value=connection):
            with self.assertRaises(ValueError):
                waiting_client(1)._wait_for_server(timeout=1)
        connection.close.assert_called_once_with()

    def test_connection_refused_retries_and_returns_metadata(self):
        now = [0.0]
        connection = mock.Mock()
        connection.recv.return_value = msgpack_numpy.packb({"ready": True})

        def advance(seconds):
            now[0] += seconds

        with (
            mock.patch.object(websocket_policy_client.time, "monotonic", side_effect=lambda: now[0]),
            mock.patch.object(websocket_policy_client.time, "sleep", side_effect=advance),
            mock.patch.object(
                websocket_policy_client.websockets.sync.client,
                "connect",
                side_effect=[ConnectionRefusedError, connection],
            ) as connect,
        ):
            received_connection, metadata = waiting_client(1)._wait_for_server(timeout=3)

        self.assertEqual(connect.call_count, 2)
        connection.recv.assert_called_once_with(timeout=1)
        self.assertIs(received_connection, connection)
        self.assertEqual(metadata, {"ready": True})

    def test_metadata_and_prediction_round_trip(self):
        metadata = {"action_chunk_size": 2}
        actions = np.zeros((1, 2, 7), dtype=np.float32)
        policy = mock.Mock()
        policy.predict_action.return_value = {"actions": actions}
        server = WebsocketPolicyServer(policy, metadata=metadata)

        def handler(connection):
            connection.send(msgpack_numpy.packb(server._metadata))
            query = msgpack_numpy.unpackb(connection.recv())
            connection.send(msgpack_numpy.packb(server._route_message(query)))

        with local_server(handler) as port:
            client = WebsocketClientPolicy(port=port)
            try:
                self.assertEqual(client.get_server_metadata(), metadata)
                result = client.predict_action({"request_id": "smoke", "payload": {"examples": []}})
                self.assertTrue(result["ok"])
                self.assertEqual(result["request_id"], "smoke")
                np.testing.assert_array_equal(result["data"]["actions"], actions)
                policy.predict_action.assert_called_once_with(examples=[])
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
