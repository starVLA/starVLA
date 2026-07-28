# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

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
    "  The client will NOT infer or reorder camera views for you.\n"
    "============================================================"
)


def _as_image_sequence(value: Any) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _image_hw(image: Any) -> Optional[Tuple[int, int]]:
    shape = getattr(image, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) >= 2:
        return int(size[1]), int(size[0])
    return None


def _expected_image_hw(metadata: Dict) -> Optional[Tuple[int, int]]:
    size = metadata.get("training_obs_image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return int(size[0]), int(size[1])
    return None


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
        self._did_log_eval_observation_contract = False
        self._did_warn_eval_observation_mismatch = False

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
        self._check_eval_observation_contract(query_info)
        data = self._packer.pack(query_info)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def _check_eval_observation_contract(self, query_info: Dict) -> None:
        examples = query_info.get("examples")
        if not isinstance(examples, list) or not examples:
            return

        expected_hw = _expected_image_hw(self._server_metadata)
        image_counts: List[int] = []
        image_shapes: List[List[Optional[Tuple[int, int]]]] = []

        for idx, example in enumerate(examples):
            if not isinstance(example, dict):
                continue
            images = _as_image_sequence(example.get("image"))
            if images is None:
                logging.warning(
                    "[TRAIN/TEST CONSISTENCY CHECK] example %d has no `image` key. "
                    "Verify this is intended for the checkpoint metadata=%s",
                    idx,
                    self._server_metadata,
                )
                continue
            shapes = [_image_hw(img) for img in images]
            image_counts.append(len(images))
            image_shapes.append(shapes)

            if expected_hw is not None:
                for image_idx, hw in enumerate(shapes):
                    if hw is not None and hw != expected_hw:
                        self._warn_eval_observation_mismatch_once(
                            "[TRAIN/TEST CONSISTENCY CHECK] eval image size mismatch: "
                            f"example={idx}, image_index={image_idx}, got={hw}, "
                            f"training_obs_image_size={expected_hw}, metadata={self._server_metadata}. "
                            "Resize/crop explicitly in the benchmark interface before calling predict_action."
                        )

        if image_counts and len(set(image_counts)) > 1:
            self._warn_eval_observation_mismatch_once(
                "[TRAIN/TEST CONSISTENCY CHECK] inconsistent image counts across eval batch: "
                f"image_counts={image_counts}. Each example should use the same explicit camera contract."
            )

        if not self._did_log_eval_observation_contract and image_counts:
            self._did_log_eval_observation_contract = True
            logging.info(
                "[TRAIN/TEST CONSISTENCY CHECK] eval request image_count=%s image_shapes=%s "
                "server_metadata=%s. The benchmark interface is responsible for camera order; "
                "verify this order manually against the checkpoint training setup.",
                image_counts[0],
                image_shapes[0] if image_shapes else None,
                self._server_metadata,
            )

    def _warn_eval_observation_mismatch_once(self, message: str) -> None:
        if self._did_warn_eval_observation_mismatch:
            return
        self._did_warn_eval_observation_mismatch = True
        logging.warning(message)
