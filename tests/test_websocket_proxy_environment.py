import contextlib
import os
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import websockets.exceptions
import websockets.sync.server

from deployment.model_server.tools import msgpack_numpy, websocket_policy_client
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

PROXY_KEYS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")


@contextlib.contextmanager
def serving(server):
    with server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)


def http_server(marker):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(marker)))
            self.end_headers()
            self.wfile.write(marker)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler)


def metadata_handler(connection):
    connection.send(msgpack_numpy.packb({"ready": True}))
    try:
        connection.recv(timeout=2)
    except (websockets.exceptions.ConnectionClosed, TimeoutError):
        pass


class WebsocketProxyEnvironmentTest(unittest.TestCase):
    def test_client_keeps_http_proxy_routing_and_environment(self):
        with (
            serving(http_server(b"proxy")) as proxy,
            serving(http_server(b"origin")) as origin,
            serving(websockets.sync.server.serve(metadata_handler, "127.0.0.1", 0)) as websocket_server,
        ):
            proxy_url = f"http://127.0.0.1:{proxy.server_port}"
            proxy_environment = {key: proxy_url for key in PROXY_KEYS}
            proxy_environment.update(NO_PROXY="", no_proxy="")
            url = f"http://127.0.0.1:{origin.server_port}/asset"

            def fetch():
                with urllib.request.build_opener().open(url, timeout=2) as response:
                    return response.read()

            with mock.patch.dict(os.environ, proxy_environment, clear=True):
                self.assertEqual(fetch(), b"proxy")
                client = WebsocketClientPolicy(port=websocket_server.socket.getsockname()[1])
                try:
                    self.assertEqual(client.get_server_metadata(), {"ready": True})
                    self.assertEqual(fetch(), b"proxy")
                    self.assertEqual(dict(os.environ), proxy_environment)
                finally:
                    client.close()

    def test_failed_connection_also_preserves_environment(self):
        proxy_environment = {key: "http://127.0.0.1:9" for key in PROXY_KEYS}
        with (
            mock.patch.dict(os.environ, proxy_environment, clear=True),
            mock.patch.object(
                websocket_policy_client.websockets.sync.client,
                "connect",
                side_effect=websockets.exceptions.InvalidHandshake("invalid peer"),
            ),
        ):
            with self.assertRaises(websockets.exceptions.InvalidHandshake):
                WebsocketClientPolicy()
            self.assertEqual(dict(os.environ), proxy_environment)

    def test_connection_without_proxy_configuration(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            serving(websockets.sync.server.serve(metadata_handler, "127.0.0.1", 0)) as server,
        ):
            client = WebsocketClientPolicy(port=server.socket.getsockname()[1])
            try:
                self.assertEqual(client.get_server_metadata(), {"ready": True})
                self.assertEqual(dict(os.environ), {})
            finally:
                client.close()


if __name__ == "__main__":
    unittest.main()
