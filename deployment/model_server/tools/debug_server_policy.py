"""
Debug / smoke-test client for deployment/model_server/server_policy.py.

Purpose:
  - Establish a WebSocket connection to the policy server and echo its handshake metadata.
  - Send one synthetic `predict_action` request to verify end-to-end transport
    (msgpack serialization + server routing + policy call + response shape).

Usage example:
  python -m deployment.model_server.tools.debug_server_policy --host 127.0.0.1 --port 10093

Notes:
  - The observation is synthetic: it validates the interface, not the policy. The
    returned actions are meaningless, only their shape is checked.
  - Any interface or transport failure raises, so the script exits non-zero and can
    be used as a gate in launch scripts.
  - Adjust `--num_images` / `--image_size` to match the camera contract the
    checkpoint was trained with; the server does not reorder or infer camera views.
    The default of one view matches `examples/eval_protocol.md`; the LIBERO client
    sends two (primary + wrist), so pass `--num_images 2` for those checkpoints.
  - The example carries no `state`, matching `eval_libero.py`. A framework that
    reads `example["state"]` unconditionally will fail here.
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy, _expected_image_hw


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WebSocket policy client smoke test (msgpack protocol)")
    ap.add_argument("--host", default="127.0.0.1", help="server hostname/IP (do not use 0.0.0.0)")
    ap.add_argument("--port", type=int, default=10093, help="server port")
    ap.add_argument("--api_key", default="", help="optional: API key for authentication")
    ap.add_argument(
        "--unnorm_key",
        default=None,
        help="dataset key for un-normalization; default: taken from the server handshake metadata",
    )
    ap.add_argument("--instruction", default="pick up the red block", help="task instruction sent as `lang`")
    ap.add_argument("--num_images", type=int, default=1, help="number of camera views in the synthetic example")
    ap.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="synthetic image size; default: `training_obs_image_size` from the server, else 224 224",
    )
    ap.add_argument("--log_level", default="INFO")
    return ap


def _resolve_unnorm_key(metadata: Dict, explicit: Optional[str]) -> Optional[str]:
    """Pick the un-normalization key to send, or None to let the server decide.

    `PolicyServerWrapper.predict_action` owns validation: on a multi-dataset
    checkpoint it rejects a missing key and names the valid ones.
    """
    if explicit:
        return explicit
    default_key = metadata.get("default_unnorm_key")
    if default_key:
        return default_key
    available = metadata.get("available_unnorm_keys") or []
    return available[0] if len(available) == 1 else None


def _resolve_image_size(metadata: Dict, explicit: Optional[List[int]]) -> Tuple[int, int]:
    if explicit:
        return int(explicit[0]), int(explicit[1])
    return _expected_image_hw(metadata) or (224, 224)


def _build_query(metadata: Dict, args: argparse.Namespace) -> Dict:
    """Build one `predict_action` request matching the eval-client contract.

    The payload is forwarded to `PolicyServerWrapper.predict_action(**payload)`;
    `examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py` is the
    reference client for its shape.
    """
    height, width = _resolve_image_size(metadata, args.image_size)
    images = [
        np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)  # (H, W, C), uint8 0-255
        for _ in range(args.num_images)
    ]
    query_info = {"examples": [{"image": images, "lang": args.instruction}]}
    unnorm_key = _resolve_unnorm_key(metadata, args.unnorm_key)
    if unnorm_key is not None:
        query_info["unnorm_key"] = unnorm_key
    logging.info(
        "Request: num_images=%d, image_size=%s, unnorm_key=%s, lang=%r",
        args.num_images,
        (height, width),
        unnorm_key,
        args.instruction,
    )
    return query_info


def _check_response(response: Dict, metadata: Dict) -> np.ndarray:
    """Raise unless the server returned a well-formed action chunk."""
    if not isinstance(response, dict):
        raise RuntimeError(f"malformed response, expected a dict: {response!r}")
    if response.get("status") != "ok":
        raise RuntimeError(f"server reported an error: {response.get('error', response)}")

    data = response.get("data")
    if not isinstance(data, dict) or "actions" not in data:
        raise RuntimeError(
            f"response has no 'actions'; data keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
        )

    actions = np.asarray(data["actions"])
    if actions.ndim != 3:
        raise RuntimeError(f"expected actions of shape (B, T, D), got {actions.shape}")

    # Eval clients cache a chunk and index it as `raw_actions[step % action_chunk_size]`
    # (model2libero_interface.py:159, model2metaworld_interface.py:120), reading the size
    # from this handshake. A short chunk raises IndexError mid-episode and a long one
    # silently drops actions, so a disagreement here is a failure, not a warning.
    chunk_size = metadata.get("action_chunk_size")
    if chunk_size is not None and actions.shape[1] != int(chunk_size):
        raise RuntimeError(
            f"action chunk length {actions.shape[1]} does not match the server's action_chunk_size {chunk_size}"
        )
    return actions


def _main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    client = WebsocketClientPolicy(host=args.host, port=args.port, api_key=(args.api_key or None))
    try:
        metadata = client.get_server_metadata()
        logging.info("Connected. Server metadata: %s", metadata)

        response = client.predict_action(_build_query(metadata, args))
        actions = _check_response(response, metadata)
        logging.info("Received actions with shape %s (B, T, D), dtype=%s", actions.shape, actions.dtype)
    finally:
        client.close()

    logging.info("Smoke test passed.")


if __name__ == "__main__":
    _main()
