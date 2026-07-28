"""
StarVLA Policy Server Inference Example — Real Code + Pseudocode

This script shows how to connect to a StarVLA policy server over WebSocket,
receive model-predicted actions, unnormalize them, and execute them on a robot arm.

⚠️ Note: the current implementation is based on a Franka arm with a 7D action space:
    [x, y, z, roll, pitch, yaw, gripper]
For other robot arms, the action dimensionality and semantics may differ, for example:
    - a dual-arm system may use 14 dimensions (7 per arm)
    - joint-space control may use N joint angles + gripper
    - the gripper index and meaning may also be different
Please adjust `action_dim`, gripper indices, and unnormalization logic to match your robot.

Real code sections (directly reusable):
    - WebSocket client connection and communication
    - request construction and response parsing
    - action unnormalization (`unnormalize_actions`)

Pseudocode sections (replace for your robot):
    - camera image acquisition
    - robot environment creation, `reset`, and `step`
"""

import json
from typing import Dict, List

import numpy as np

# ============================================================
# ✅ Real code: WebSocket client
# ============================================================
# WebsocketClientPolicy uses msgpack_numpy serialization and communicates with the server over WebSocket.
# Source: starVLA/deployment/model_server/tools/websocket_policy_client.py
from websocketclient import WebsocketClientPolicy


# ============================================================
# ✅ Real code: action unnormalization
# ============================================================
def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert normalized model outputs in [-1, 1] back to the real action space.

    Args:
        normalized_actions: normalized actions, shape [T, action_dim], range [-1, 1]
        action_norm_stats: normalization statistics containing:
            - "min": np.ndarray, per-dimension minimum values
            - "max": np.ndarray, per-dimension maximum values
            - "mask": np.ndarray (bool), which dimensions are normalized

    Returns:
        actions: unnormalized actions, shape [T, action_dim]

    Formula:
        action = 0.5 * (normalized + 1) * (max - min) + min
        where the gripper dimension is first binarized: < 0.5 → -1, >= 0.5 → 1
    """
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high = np.array(action_norm_stats["max"])
    action_low = np.array(action_norm_stats["min"])

    normalized_actions = np.clip(normalized_actions, -1, 1)

    # Apply binary thresholding to the gripper dimension (index=6).
    # ⚠️ In the current Franka 7D action space, the gripper is at index=6.
    # Other robots may use different action sizes and gripper indices.
    if normalized_actions.shape[-1] >= 7:
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, -1, 1)

    # Linear unnormalization, only for dimensions with mask=True.
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


# ============================================================
# ✅ Real code: request construction and response parsing
# ============================================================
def build_request(images: List[np.ndarray], task_instruction: str) -> dict:
    """
    Construct a request for the StarVLA policy server.

    Args:
        images: multi-view camera images, each with shape (H, W, 3), dtype uint8
        task_instruction: natural-language task instruction, e.g. "Pick up the red cup"

    Returns:
        request_data: dict matching the server API
    """
    request_data = {
        "examples": [
            {
                "image": images,  # List[np.ndarray], converted to PIL Images on the server side
                "lang": task_instruction,
            }
        ]
    }
    return request_data


def parse_response(result: dict) -> np.ndarray:
    """
    Parse the policy server response and extract the action chunk.

    Args:
        result: dict returned by the server, formatted as:
            {"data": {"normalized_actions": np.ndarray}, "status": "ok"}

    Returns:
        action_chunk: shape [T, action_dim], where T is the predicted horizon
    """
    data = result.get("data", result)

    # Try several common output keys.
    for key in ["normalized_actions", "actions", "action"]:
        if key in data:
            actions = data[key]
            if isinstance(actions, list):
                actions = np.array(actions)
            # Normalize to [T, action_dim].
            if len(actions.shape) == 3:
                actions = actions[0]  # [B, T, D] -> [T, D]
            elif len(actions.shape) == 1:
                actions = actions.reshape(1, -1)  # [D] -> [1, D]
            return actions

    raise KeyError(f"Could not extract actions from response. Available keys: {list(data.keys())}")


# ============================================================
# ✅ Real code: load action normalization statistics
# ============================================================
def load_action_norm_stats(json_path: str, embodiment_key: str = "franka") -> Dict[str, np.ndarray]:
    """
    Load normalization statistics from `dataset_statistics.json`.

    Example JSON format:
    {
        "franka": {
            "action": {
                "min": [...],
                "max": [...],
                "mask": [true, true, ...]
            }
        }
    }
    """
    with open(json_path, "r") as f:
        stats_data = json.load(f)

    if embodiment_key in stats_data:
        stats_data = stats_data[embodiment_key]
    if "action" in stats_data:
        stats_data = stats_data["action"]

    norm_stats = {
        "min": np.array(stats_data.get("min", stats_data.get("low", []))),
        "max": np.array(stats_data.get("max", stats_data.get("high", []))),
    }
    if "mask" in stats_data:
        norm_stats["mask"] = np.array(stats_data["mask"], dtype=bool)

    return norm_stats


# ============================================================
# === Pseudocode: implement the following functions for your robot ===
# ============================================================


def capture_images_from_cameras() -> List[np.ndarray]:
    """
    [Pseudocode] Capture multi-view images from cameras.

    TODO: implement this based on your camera hardware, for example:
        - RealSense: use the pyrealsense2 SDK
        - USB cameras: use OpenCV VideoCapture
        - Other devices: use the corresponding SDK

    Returns:
        images: List[np.ndarray], each with shape (H, W, 3), dtype uint8, in RGB format
    """
    # --- Example pseudocode ---
    # import pyrealsense2 as rs
    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()
    # image = np.asanyarray(color_frame.get_data())  # BGR
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB
    # image = cv2.resize(image, (224, 224))
    # return [image_wrist, image_base]  # multi-view
    raise NotImplementedError("Please implement camera image acquisition")


class YourRobotEnv:
    """
    [Pseudocode] Robot arm environment interface.

    You need to implement the following methods to send 7D actions to your robot:
        - reset(): move the robot to its initial pose and return the initial observation
        - step(action): execute a 7D action [x, y, z, roll, pitch, yaw, gripper]
        - get_obs(): return the current observation (images + state)

    Action definition (Franka 7D action space):
        action[0:3] - position delta (x, y, z), Cartesian coordinates, in meters
        action[3:6] - orientation delta (roll, pitch, yaw), Euler angles, in radians
        action[6]   - gripper control (-1: close, 1: open)

    ⚠️ Other robots may use different action sizes and meanings. Adjust accordingly.

    `env.step()` should internally handle both:
        1. pose control: convert action[0:6] into a target pose and send it to the controller
        2. gripper control: open/close the gripper based on action[6]
    """

    def reset(self):
        """Reset the robot to its initial pose."""
        # TODO: send a reset command to the robot
        # TODO: wait until the robot reaches the initial pose
        # TODO: return the initial observation
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """
        Execute one action step (pose + gripper handled together).

        Args:
            action: np.ndarray, shape (7,),
                [x, y, z, roll, pitch, yaw, gripper]

        Returns:
            obs: dict, observation (including images and state)
            reward: float
            done: bool
            truncated: bool
            info: dict
        """
        # TODO: Example implementation:
        #
        # 1. Parse the action
        # pose_delta = action[0:6]   # pose delta
        # gripper_cmd = action[6]    # gripper: -1=close, 1=open
        #
        # 2. Compute the target pose
        # target_pose = current_pose + pose_delta * action_scale
        # target_pose = clip_to_safety_box(target_pose)
        #
        # 3. Send the pose command
        # robot.move_to(target_pose)
        #
        # 4. Send the gripper command
        # if gripper_cmd >= 0.9:
        #     robot.open_gripper()
        # elif gripper_cmd <= -0.9:
        #     robot.close_gripper()
        #
        # 5. Get the observation
        # obs = self.get_obs()
        # return obs, reward, done, truncated, info
        raise NotImplementedError

    def get_obs(self) -> dict:
        """Get the current observation."""
        # TODO: return a dict containing images and state
        # return {
        #     "images": capture_images_from_cameras(),
        #     "state": robot.get_state(),
        # }
        raise NotImplementedError


# ============================================================
# Main inference loop
# ============================================================
def main():
    # ------ Configuration ------
    policy_host = "127.0.0.1"
    policy_port = 5694
    task_instruction = "Pick up the pink cube and place it into the black box."
    action_stats_path = "/path/to/dataset_statistics.json"
    max_episodes = 10
    max_steps_per_episode = 500

    # ------ ✅ Real code: load normalization statistics ------
    action_norm_stats = load_action_norm_stats(action_stats_path, embodiment_key="franka")
    print(f"Action min: {action_norm_stats['min']}")
    print(f"Action max: {action_norm_stats['max']}")

    # ------ ✅ Real code: connect to the Policy Server ------
    client = WebsocketClientPolicy(host=policy_host, port=policy_port)
    print(f"Connected to Policy Server: {policy_host}:{policy_port}")

    # ------ [Pseudocode] Create the robot environment ------
    env = YourRobotEnv()  # TODO: replace with your robot environment
    obs = env.reset()

    # ------ Main inference loop ------
    for episode in range(max_episodes):
        obs = env.reset()
        print(f"\n--- Episode {episode + 1}/{max_episodes} ---")

        step_count = 0
        done = False

        while step_count < max_steps_per_episode and not done:

            # Step 1: [Pseudocode] Read images from the observation
            images = obs["images"]  # List[np.ndarray], (H, W, 3), uint8

            # Step 2: ✅ Real code - build the request and call the policy server
            request = build_request(images, task_instruction)
            result = client.predict_action(request)

            # Step 3: ✅ Real code - parse the response and get the normalized action chunk
            normalized_action_chunk = parse_response(result)  # [T, 7]

            # Step 4: ✅ Real code - unnormalize the actions
            action_chunk = unnormalize_actions(normalized_action_chunk, action_norm_stats)
            # action_chunk: [T, 7], each row = [x, y, z, roll, pitch, yaw, gripper]

            # Step 5: Execute the action chunk step by step
            for action in action_chunk:
                # action is a 7D vector: [x, y, z, roll, pitch, yaw, gripper]
                # env.step() should internally handle both pose and gripper control
                obs, reward, done, truncated, info = env.step(action)  # [Pseudocode]
                step_count += 1

                if done or truncated:
                    break

            if done or truncated:
                break

        print(f"Episode {episode + 1} finished, steps: {step_count}")

    # Close the connection
    client.close()
    print("Inference complete")


if __name__ == "__main__":
    main()
