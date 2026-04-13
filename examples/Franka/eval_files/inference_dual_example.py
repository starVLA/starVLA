"""
StarVLA Policy Server Dual-Arm Inference Example — Real Code + Pseudocode

This script shows how to connect to a StarVLA policy server over WebSocket,
receive model-predicted dual-arm actions, unnormalize them, and execute them on a dual-arm robot.

⚠️ Note: the current implementation is based on a dual-arm Franka setup with a 14D action space:
    [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l,
     x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r]
    where action[0:7] corresponds to the left arm and action[7:14] to the right arm.

For other dual-arm robots, the action dimensionality, arm ordering, and gripper indices may differ.
Please adjust `action_dim`, gripper indices, and unnormalization logic to match your robot.

For single-arm robots (7D), see `inference_single_example.py`.

Real code sections (directly reusable):
    - WebSocket client connection and communication
    - request construction and response parsing
    - action unnormalization (`unnormalize_actions`), supporting 7D single-arm and 14D dual-arm actions

Pseudocode sections (replace for your robot):
    - camera image acquisition
    - dual-arm robot environment creation, `reset`, and `step`
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
# ✅ Real code: action unnormalization (supports 7D single-arm and 14D dual-arm)
# ============================================================
def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Convert normalized model outputs in [-1, 1] back to the real action space.
    Supports both single-arm (7D) and dual-arm (14D) action spaces.

    Args:
        normalized_actions: normalized actions, shape [T, action_dim], range [-1, 1]
            - single-arm: action_dim=7, gripper at index=6
            - dual-arm: action_dim=14, grippers at index=6 (left arm) and index=13 (right arm)
        action_norm_stats: normalization statistics containing:
            - "min": np.ndarray, per-dimension minimum values
            - "max": np.ndarray, per-dimension maximum values
            - "mask": np.ndarray (bool), which dimensions are normalized

    Returns:
        actions: unnormalized actions, shape [T, action_dim]

    Formula:
        action = 0.5 * (normalized + 1) * (max - min) + min
        where the gripper dimensions are first binarized: < 0.5 → -1, >= 0.5 → 1
    """
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high = np.array(action_norm_stats["max"])
    action_low = np.array(action_norm_stats["min"])

    normalized_actions = np.clip(normalized_actions, -1, 1)

    # Apply binary thresholding to the gripper dimensions.
    action_dim = normalized_actions.shape[-1]
    if action_dim == 14:
        # Dual-arm: left gripper at index=6, right gripper at index=13.
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, -1, 1)
        normalized_actions[:, 13] = np.where(normalized_actions[:, 13] < 0.5, -1, 1)
    elif action_dim >= 7:
        # Single-arm: gripper at index=6.
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
        task_instruction: natural-language task instruction

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
def load_action_norm_stats(json_path: str, embodiment_key: str = "new_embodiment") -> Dict[str, np.ndarray]:
    """
    Load normalization statistics from `dataset_statistics.json`.

    Example JSON format for a dual-arm 14D setup:
    {
        "new_embodiment": {
            "action": {
                "min": [14 minimum values],
                "max": [14 maximum values],
                "mask": [true, true, ..., true]  // 14 entries
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
# === Pseudocode: implement the following functions for your dual-arm robot ===
# ============================================================


def capture_images_from_cameras() -> List[np.ndarray]:
    """
    [Pseudocode] Capture multi-view images from cameras.

    Dual-arm setups usually require more camera views, for example:
        - a left wrist camera and a right wrist camera
        - a global base camera

    TODO: implement this based on your camera hardware

    Returns:
        images: List[np.ndarray], each with shape (H, W, 3), dtype uint8, in RGB format
    """
    # --- Example pseudocode ---
    # return [image_left_wrist, image_right_wrist, image_base]
    raise NotImplementedError("Please implement camera image acquisition")


class YourDualArmRobotEnv:
    """
    [Pseudocode] Dual-arm robot environment interface.

    You need to implement the following methods to send 14D actions to a dual-arm robot:
        - reset(): move both arms to their initial poses and return the initial observation
        - step(action): execute a 14D action
        - get_obs(): return the current observation (images + state)

    Dual-arm action definition (Franka dual-arm 14D action space):
        action[0:3]   - left-arm position delta (x, y, z)
        action[3:6]   - left-arm orientation delta (roll, pitch, yaw)
        action[6]     - left gripper control (-1: close, 1: open)
        action[7:10]  - right-arm position delta (x, y, z)
        action[10:13] - right-arm orientation delta (roll, pitch, yaw)
        action[13]    - right gripper control (-1: close, 1: open)

    ⚠️ Other dual-arm robots may use different action sizes, arm ordering, and gripper indices.

    `env.step()` should internally handle both:
        1. left-arm pose control + left gripper control
        2. right-arm pose control + right gripper control
    """

    def reset(self):
        """Reset both arms to their initial poses."""
        # TODO: send a reset command for both arms
        # TODO: wait until both arms reach their initial poses
        # TODO: return the initial observation
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """
        Execute one action step (dual-arm pose + grippers handled together).

        Args:
            action: np.ndarray, shape (14,)
                [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l,
                 x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r]

        Returns:
            obs: dict, observation (including images and state)
            reward: float
            done: bool
            truncated: bool
            info: dict
        """
        # TODO: Example implementation:
        #
        # 1. Parse the dual-arm action
        # left_pose_delta = action[0:6]
        # left_gripper_cmd = action[6]
        # right_pose_delta = action[7:13]
        # right_gripper_cmd = action[13]
        #
        # 2. Left-arm pose control
        # left_target = left_current_pose + left_pose_delta * action_scale
        # left_target = clip_to_safety_box(left_target)
        # left_robot.move_to(left_target)
        #
        # 3. Left gripper control
        # if left_gripper_cmd >= 0.9:
        #     left_robot.open_gripper()
        # elif left_gripper_cmd <= -0.9:
        #     left_robot.close_gripper()
        #
        # 4. Right-arm pose control
        # right_target = right_current_pose + right_pose_delta * action_scale
        # right_target = clip_to_safety_box(right_target)
        # right_robot.move_to(right_target)
        #
        # 5. Right gripper control
        # if right_gripper_cmd >= 0.9:
        #     right_robot.open_gripper()
        # elif right_gripper_cmd <= -0.9:
        #     right_robot.close_gripper()
        #
        # 6. Get the observation
        # obs = self.get_obs()
        # return obs, reward, done, truncated, info
        raise NotImplementedError

    def get_obs(self) -> dict:
        """Get the current observation."""
        # TODO: return a dict containing images and state
        # return {
        #     "images": capture_images_from_cameras(),
        #     "state": {
        #         "left_tcp_pose": ...,
        #         "right_tcp_pose": ...,
        #         ...
        #     },
        # }
        raise NotImplementedError


# ============================================================
# Main inference loop (dual-arm)
# ============================================================
def main():
    # ------ Configuration ------
    policy_host = "127.0.0.1"
    policy_port = 5694
    task_instruction = "Two robotic arms are working together to perform a handover task."
    action_stats_path = "/path/to/dataset_statistics.json"
    max_episodes = 10
    max_steps_per_episode = 500

    # ------ ✅ Real code: load normalization statistics (dual-arm 14D) ------
    action_norm_stats = load_action_norm_stats(action_stats_path, embodiment_key="new_embodiment")
    print(f"Action dim: {len(action_norm_stats['min'])}")  # should be 14
    print(f"Action min: {action_norm_stats['min']}")
    print(f"Action max: {action_norm_stats['max']}")

    # ------ ✅ Real code: connect to the Policy Server ------
    client = WebsocketClientPolicy(host=policy_host, port=policy_port)
    print(f"Connected to Policy Server: {policy_host}:{policy_port}")

    # ------ [Pseudocode] Create the dual-arm robot environment ------
    env = YourDualArmRobotEnv()  # TODO: replace with your dual-arm robot environment
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
            normalized_action_chunk = parse_response(result)  # [T, 14]

            # Step 4: ✅ Real code - unnormalize the actions
            action_chunk = unnormalize_actions(normalized_action_chunk, action_norm_stats)
            # action_chunk: [T, 14], each row = [left-arm 7D, right-arm 7D]

            # Step 5: Execute the action chunk step by step
            for action in action_chunk:
                # action is a 14D vector:
                # [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l,
                #  x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r]
                # env.step() should internally handle dual-arm pose and gripper control together
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
