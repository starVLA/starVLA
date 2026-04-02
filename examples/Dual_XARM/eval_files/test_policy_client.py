import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def load_or_random_image(path: str | None, image_size: int = 224) -> np.ndarray:
    if path is None:
        return np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    return np.asarray(img, dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual_XARM policy inference smoke test")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5694)
    parser.add_argument("--instruction", type=str, default="Pick up the box and place it to the target area.")
    parser.add_argument("--head_image", type=str, default=None)
    parser.add_argument("--left_wrist_image", type=str, default=None)
    parser.add_argument("--right_wrist_image", type=str, default=None)
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    args = parser.parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)

    images = [
        load_or_random_image(args.head_image),
        load_or_random_image(args.left_wrist_image),
        load_or_random_image(args.right_wrist_image),
    ]

    example = {
        "image": images,
        "lang": args.instruction,
    }

    payload = {
        "examples": [example],
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": args.num_ddim_steps,
    }

    result = client.predict_action(payload)
    normalized_actions = np.asarray(result["data"]["normalized_actions"])
    print("normalized_actions shape:", normalized_actions.shape)
    print("first step action:", normalized_actions[0, 0].tolist())


if __name__ == "__main__":
    main()
