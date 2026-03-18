"""
StarVLA Policy Server 推理示例 — 混合真实代码 + 伪代码

本脚本展示如何连接 StarVLA policy server（通过 WebSocket），
获取模型预测的动作，反归一化后控制机械臂执行。

⚠️ 注意：当前实现基于 Franka 机械臂，动作空间为 7 维:
  [x, y, z, roll, pitch, yaw, gripper]
对于其他机械臂，动作维度和含义可能不同，例如:
  - 双臂机械臂可能是 14 维 (每臂 7 维)
  - 关节空间控制可能是 N 个关节角 + gripper
  - 夹爪维度的索引和语义也可能不同
请根据你的机械臂实际情况调整 action_dim、夹爪索引和反归一化逻辑。

真实代码部分（可直接运行）：
  - WebSocket 客户端连接与通信
  - 请求构造与响应解析
  - 动作反归一化 (unnormalize_actions)

伪代码部分（需根据你的机械臂替换）：
  - 相机图像获取
  - 机械臂环境创建、reset、step
"""

import numpy as np
import json
import time
from typing import List, Dict

# ============================================================
# ✅ 真实代码：WebSocket 客户端
# ============================================================
# WebsocketClientPolicy 使用 msgpack_numpy 序列化，通过 WebSocket 与 server 通信
# 源码位于: starVLA/deployment/model_server/tools/websocket_policy_client.py
from websocketclient import WebsocketClientPolicy


# ============================================================
# ✅ 真实代码：动作反归一化
# ============================================================
def unnormalize_actions(normalized_actions: np.ndarray, 
                        action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """
    将模型输出的归一化动作 [-1, 1] 转换回真实动作空间。

    Args:
        normalized_actions: 归一化动作, shape [T, action_dim], 值域 [-1, 1]
        action_norm_stats: 归一化统计信息，包含:
            - "min": np.ndarray, 各维度最小值
            - "max": np.ndarray, 各维度最大值
            - "mask": np.ndarray (bool), 哪些维度参与归一化

    Returns:
        actions: 反归一化后的动作, shape [T, action_dim]

    公式:
        action = 0.5 * (normalized + 1) * (max - min) + min
        其中 gripper 维度先做二值化: < 0.5 → -1, >= 0.5 → 1
    """
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high = np.array(action_norm_stats["max"])
    action_low = np.array(action_norm_stats["min"])

    normalized_actions = np.clip(normalized_actions, -1, 1)

    # 夹爪维度 (index=6) 做二值化阈值处理
    # ⚠️ 当前为 Franka 7D 动作空间，夹爪在 index=6
    # 其他机械臂的夹爪索引和维数可能不同，请相应修改
    if normalized_actions.shape[-1] >= 7:
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, -1, 1)

    # 线性反归一化（仅对 mask=True 的维度）
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


# ============================================================
# ✅ 真实代码：构造请求 & 解析响应
# ============================================================
def build_request(images: List[np.ndarray], task_instruction: str) -> dict:
    """
    构造发送给 StarVLA policy server 的请求。

    Args:
        images: 多视角相机图像列表, 每张 shape (H, W, 3), dtype uint8
        task_instruction: 自然语言任务指令, 如 "Pick up the red cup"

    Returns:
        request_data: 符合 server 接口的 dict
    """
    request_data = {
        "examples": [{
            "image": images,       # List[np.ndarray], server 端会转为 PIL Image
            "lang": task_instruction,
        }]
    }
    return request_data


def parse_response(result: dict) -> np.ndarray:
    """
    解析 policy server 返回的结果，提取 action chunk。

    Args:
        result: server 返回的 dict，格式:
            {"data": {"normalized_actions": np.ndarray}, "status": "ok"}

    Returns:
        action_chunk: shape [T, action_dim], T 为模型预测的时间步数
    """
    data = result.get("data", result)

    # 尝试多种常见 key
    for key in ["normalized_actions", "actions", "action"]:
        if key in data:
            actions = data[key]
            if isinstance(actions, list):
                actions = np.array(actions)
            # 统一为 [T, action_dim]
            if len(actions.shape) == 3:
                actions = actions[0]       # [B, T, D] → [T, D]
            elif len(actions.shape) == 1:
                actions = actions.reshape(1, -1)  # [D] → [1, D]
            return actions

    raise KeyError(f"无法从响应中提取动作，可用 keys: {list(data.keys())}")


# ============================================================
# ✅ 真实代码：加载 action 归一化统计
# ============================================================
def load_action_norm_stats(json_path: str, embodiment_key: str = "franka") -> Dict[str, np.ndarray]:
    """
    从 dataset_statistics.json 加载归一化统计信息。

    JSON 格式示例:
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
    with open(json_path, 'r') as f:
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
# === 伪代码：以下函数需要根据你的机械臂具体实现 ===
# ============================================================

def capture_images_from_cameras() -> List[np.ndarray]:
    """
    [伪代码] 从相机获取多视角图像。

    TODO: 根据你的相机硬件实现，例如:
      - RealSense: 使用 pyrealsense2 SDK
      - USB 摄像头: 使用 OpenCV VideoCapture
      - 其他: 使用对应 SDK

    Returns:
        images: List[np.ndarray], 每张 shape (H, W, 3), dtype uint8, RGB 格式
    """
    # --- 示例伪代码 ---
    # import pyrealsense2 as rs
    # frames = pipeline.wait_for_frames()
    # color_frame = frames.get_color_frame()
    # image = np.asanyarray(color_frame.get_data())  # BGR
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转 RGB
    # image = cv2.resize(image, (224, 224))
    # return [image_wrist, image_base]  # 多视角
    raise NotImplementedError("请实现相机图像获取")


class YourRobotEnv:
    """
    [伪代码] 机械臂环境接口。

    你需要实现以下方法，将 7D 动作统一发送给机械臂：
      - reset(): 重置机械臂到初始位姿，返回初始观测
      - step(action): 执行 7D 动作 [x,y,z,roll,pitch,yaw,gripper]
      - get_obs(): 获取当前观测（含图像和状态）

    action 说明 (Franka 7D 动作空间):
      action[0:3] — 位置增量 (x, y, z)，笛卡尔坐标，单位: 米
      action[3:6] — 姿态增量 (roll, pitch, yaw)，欧拉角，单位: 弧度
      action[6]   — 夹爪控制 (-1: 关闭, 1: 打开)

    ⚠️ 其他机械臂的动作维度和含义可能不同，请根据实际情况调整。

    env.step() 需在内部同时处理:
      1. 位姿控制: 将 action[0:6] 转换为目标位姿并发送给机械臂控制器
      2. 夹爪控制: 根据 action[6] 的值控制夹爪开合
    """

    def reset(self):
        """重置机械臂到初始位姿"""
        # TODO: 发送 reset 指令给机械臂
        # TODO: 等待机械臂到达初始位姿
        # TODO: 获取并返回初始观测
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """
        执行一步动作（位姿 + 夹爪统一执行）。

        Args:
            action: np.ndarray, shape (7,)
                [x, y, z, roll, pitch, yaw, gripper]

        Returns:
            obs: dict, 观测（含图像和状态）
            reward: float
            done: bool
            truncated: bool
            info: dict
        """
        # TODO: 实现示例:
        #
        # 1. 解析动作
        # pose_delta = action[0:6]   # 位姿增量
        # gripper_cmd = action[6]    # 夹爪: -1=关, 1=开
        #
        # 2. 计算目标位姿
        # target_pose = current_pose + pose_delta * action_scale
        # target_pose = clip_to_safety_box(target_pose)
        #
        # 3. 发送位姿命令
        # robot.move_to(target_pose)
        #
        # 4. 发送夹爪命令
        # if gripper_cmd >= 0.9:
        #     robot.open_gripper()
        # elif gripper_cmd <= -0.9:
        #     robot.close_gripper()
        #
        # 5. 获取观测
        # obs = self.get_obs()
        # return obs, reward, done, truncated, info
        raise NotImplementedError

    def get_obs(self) -> dict:
        """获取当前观测"""
        # TODO: 返回包含图像和状态的 dict
        # return {
        #     "images": capture_images_from_cameras(),
        #     "state": robot.get_state(),
        # }
        raise NotImplementedError


# ============================================================
# 主推理循环
# ============================================================
def main():
    # ------ 配置参数 ------
    policy_host = "127.0.0.1"
    policy_port = 5694
    task_instruction = "Pick up the pink cube and place it into the black box."
    action_stats_path = "/path/to/dataset_statistics.json"
    max_episodes = 10
    max_steps_per_episode = 500

    # ------ ✅ 真实代码：加载归一化统计 ------
    action_norm_stats = load_action_norm_stats(action_stats_path, embodiment_key="franka")
    print(f"Action min: {action_norm_stats['min']}")
    print(f"Action max: {action_norm_stats['max']}")

    # ------ ✅ 真实代码：连接 Policy Server ------
    client = WebsocketClientPolicy(host=policy_host, port=policy_port)
    print(f"已连接 Policy Server: {policy_host}:{policy_port}")

    # ------ [伪代码] 创建机械臂环境 ------
    env = YourRobotEnv()  # TODO: 替换为你的机械臂环境
    obs = env.reset()

    # ------ 推理主循环 ------
    for episode in range(max_episodes):
        obs = env.reset()
        print(f"\n--- Episode {episode + 1}/{max_episodes} ---")

        step_count = 0
        done = False

        while step_count < max_steps_per_episode and not done:

            # Step 1: [伪代码] 从观测中获取图像
            images = obs["images"]  # List[np.ndarray], (H, W, 3), uint8

            # Step 2: ✅ 真实代码 — 构造请求并调用 policy server
            request = build_request(images, task_instruction)
            result = client.predict_action(request)

            # Step 3: ✅ 真实代码 — 解析响应，获取归一化动作 chunk
            normalized_action_chunk = parse_response(result)  # [T, 7]

            # Step 4: ✅ 真实代码 — 反归一化
            action_chunk = unnormalize_actions(normalized_action_chunk, action_norm_stats)
            # action_chunk: [T, 7], 每行 = [x, y, z, roll, pitch, yaw, gripper]

            # Step 5: 逐步执行 action chunk
            for action in action_chunk:
                # action 是 7D 向量: [x, y, z, roll, pitch, yaw, gripper]
                # env.step() 内部统一处理位姿控制和夹爪控制
                obs, reward, done, truncated, info = env.step(action)  # [伪代码]
                step_count += 1

                if done or truncated:
                    break

            if done or truncated:
                break

        print(f"Episode {episode + 1} 完成, steps: {step_count}")

    # 关闭连接
    client.close()
    print("推理完成")


if __name__ == "__main__":
    main()
