# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import os
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client
from typing_extensions import override

from . import msgpack_numpy

# =============================================================================
# TRAIN / TEST CONSISTENCY REMINDER (shown at every eval entry point)
# -----------------------------------------------------------------------------
# Every eval benchmark under `examples/` connects to the policy server through
# this client, so this banner is emitted once per eval run. Embodied policies
# are extremely sensitive to the gap between how observations are built during
# TRAINING versus INFERENCE. A silent mismatch will NOT raise an error, it will
# only quietly degrade the success rate.
# =============================================================================
_CONSISTENCY_REMINDER = (
    "\n"
    "============================================================\n"
    "  [TRAIN/TEST CONSISTENCY CHECK] read before trusting results\n"
    "------------------------------------------------------------\n"
    "  Make sure the EVAL observation matches TRAINING for:\n"
    "    - state    : whether proprioceptive state is fed (use_state)\n"
    "                 and its dimension / ordering\n"
    "    - img size : resize / crop resolution (e.g. 224x224)\n"
    "    - img count: how many camera views are fed to the model\n"
    "    - img order: the ordering of those camera views\n"
    "    - horizon  : action chunk size / action horizon\n"
    "  A mismatch on ANY of these silently lowers the success rate.\n"
    "  Cross-check the values below against your training config.\n"
    "============================================================"
)


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "127.0.0.1", port: Optional[int] = 10093, api_key: Optional[str] = None) -> None:
        # 0.0.0.0 cannot be used as a connection target, here default 127.0.0.1
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

        # Remind the user to keep the eval-time observation pipeline aligned with
        # training, and echo the server metadata so the values can be verified.
        logging.warning(_CONSISTENCY_REMINDER)
        logging.warning("[TRAIN/TEST CONSISTENCY CHECK] server metadata: %s", self._server_metadata)

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self, timeout: float = 300) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        start_time = time.time()

        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to server within {timeout} seconds")

            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=150,
                    ping_interval=None,
                    ping_timeout=60,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for server {self._uri} ...")
                time.sleep(2)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass

    @override
    def predict_action(self, query_info: Dict) -> Dict:
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)
