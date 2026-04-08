"""
Debug / smoke-test client for deployment/model_server/server_policy.py.

Purpose:
  - Establish a WebSocket connection to the policy server.
  - Optionally run a simple inference request to verify end-to-end transport.

Usage example:
  python -m deployment.model_server.tools.debug_server_policy --host 127.0.0.1 --port 10093 --test infer
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WebSocket policy client smoke test (msgpack protocol)")
    ap.add_argument("--host", default="127.0.0.1", help="server hostname/IP (do not use 0.0.0.0)")
    ap.add_argument("--port", type=int, default=10093, help="server port")
    ap.add_argument("--api_key", default="", help="optional API key")
    ap.add_argument("--image", default="assets/starVLA_LOGO.png", help="local RGB image used for the smoke test")
    ap.add_argument("--instruction", default="pick up the red block", help="instruction text for smoke-test inference")
    ap.add_argument("--test", choices=["connect", "infer"], default="infer", help="connect only, or run one inference")
    ap.add_argument("--log_level", default="INFO")
    return ap


def _load_image(image_path: str) -> np.ndarray:
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = (_WORKSPACE_ROOT / resolved_path).resolve()
    image = Image.open(resolved_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    client = WebsocketClientPolicy(host=args.host, port=args.port, api_key=(args.api_key or None))
    logging.info("Connected. Server metadata: %s", client.get_server_metadata())

    if args.test == "infer":
        try:
            image_primary_np = _load_image(args.image)
            request = {
                "type": "infer",
                "request_id": "smoke-test",
                "examples": [{
                    "image": [image_primary_np],
                    "lang": args.instruction,
                }],
                "return_cache_info": True,
            }
            infer_ret = client.infer(request)
            logging.info("Infer resp: %s", infer_ret)
        except Exception as exc:
            logging.error("Infer error (transport may still be fine): %s", exc)

    client.close()
    logging.info("Smoke test done.")


if __name__ == "__main__":
    _main()
