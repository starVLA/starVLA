"""
Debug / smoke-test client for deployment/model_server/server_policy.py.

Purpose:
  - Establish a WebSocket connection to the policy server.
  - Initialize device on the server side.
  - Optionally run a very simple inference request to verify end-to-end transport
    (serialization + server handling).

Usage example:
  python debug_server_policy.py --host 127.0.0.1 --port 10093 --device cuda --test infer

Notes:
  - The random observation is synthetic and only meant to validate the interface.
  - Adjust keys (e.g. 'images', 'task_description') to match the server's expected schema.
"""

import argparse
import logging
import numpy as np


from tools.websocket_policy_client import WebsocketClientPolicy


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WebSocket policy client smoke test (msgpack protocol)")
    ap.add_argument("--host", default="127.0.0.1", help="server hostname/IP (do not use 0.0.0.0)")
    ap.add_argument("--port", type=int, default=10093, help="server port")
    ap.add_argument("--api_key", default="", help="optional: API key for authentication")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="initialize device")
    ap.add_argument(
        "--test", choices=["init", "infer"], default="infer", help="test mode: only initialize, or try simple inference"
    )
    ap.add_argument("--log_level", default="INFO")
    return ap


def _main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    client = WebsocketClientPolicy(host=args.host, port=args.port, api_key=(args.api_key or None))
    logging.info("Connected. Server metadata: %s", client.get_server_metadata())

    # 1) device initialization
    init_ret = client.init_device(args.device)  # here to set some things on the server
    logging.info("Init device resp: %s", init_ret)

    # 2) optional: try one simple inference
    if args.test == "infer":
        try:
            # build observation aligned with model API
            H, W = 224, 224
            img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            wrist_img = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
            state = np.zeros((7,), dtype=np.float32)  # [x,y,z, ax,ay,az, gripper]

            observation = {  # key to align with model API
                "request_id": "smoke-test",
                "observation.primary": np.expand_dims(img, axis=0),  # (1,H,W,C), uint8 0-255
                "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (1,H,W,C)
                "observation.state": np.expand_dims(state, axis=0),  # (1,7), float32
                "instruction": ["debug: pick up the red block"],  # single element list
            }

            image_path = "assets/table.jpeg"
            # read image as PIL
            from PIL import Image
            image_primary = Image.open(image_path).convert("RGB")
            # Convert PIL -> numpy uint8 (H,W,3)
            image_primary_np = np.asarray(image_primary, dtype=np.uint8)

            instruction_lang="pick up the red block"
            obs = {
                "request_id": "smoke-test",
                "batch_images": [[image_primary_np]],
                "instructions": [instruction_lang],  # assume batch task description
            }

            infer_ret = client.infer(obs)
            logging.info("Infer resp: %s", infer_ret)
        except Exception as e:
            logging.error("Infer error (this still proves transport OK): %s", e)

    client.close()
    logging.info("Smoke test done.")


if __name__ == "__main__":
    _main()
