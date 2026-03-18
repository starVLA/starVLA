# 使用 Franka 数据训练 StarVLA 模型

## 第一部分：转换数据到 Lerobot 2.1 格式

1. 将你的 Franka 数据转换为 `lerobot 2.1` 格式。

## 第二部分：注册训练数据到 StarVLA
### 1. 配置 Modality 文件
在你的数据目录下创建 `modality.json`：

```
your_data/meta/modality.json
```

### 2. 注册数据集

在 StarVLA 的 `dataloader` 中注册你的数据集：

```
starVLA/dataloader/gr00t_lerobot/data_config.py
```

### 3. 配置访问路径

创建或修改训练配置文件，例如：

```
examples/Franka/train_files/starvla_cotrain_franka_single.yaml
```

配置内容示例：

```yaml
vla_data:
  dataset_py: lerobot_datasets
  data_root_dir: playground/Datasets/franka_pick_and_place_lerobot
  data_mix: smartmore_franka_eef_joints
```

配置mixure： starVLA/dataloader/gr00t_lerobot/mixtures.py
添加新dataset的mixure

### 4. 验证 DataLoader

使用以下命令检查 DataLoader 是否正确：

```bash
python starVLA/dataloader/lerobot_datasets.py \
    --config_yaml examples/Franka/train_files/starvla_cotrain_franka_single.yaml
```

## 第三部分：模型训练准备

### 1. 配置模型参数

在 YAML 配置文件中设置模型参数，例如：

```yaml
action_model:
  action_dim: 7
  state_dim: 18
```

### 2. 验证模型 Forward

在开始训练前，可以验证模型是否可以正常 forward：

```bash
python starVLA/model/framework/QwenOFT.py \
    --config_yaml examples/Franka/train_files/starvla_cotrain_franka_single.yaml
```

### 3. 开始训练

训练脚本示例：

```bash
bash path/to/your/train_script.sh
```

训练完成后，你可以在配置的输出目录中找到训练结果和模型权重。


# Franka 机械臂部署 StarVLA 模型

本节说明如何使用 StarVLA policy server 控制机械臂，包括如何适配到你自己的机械臂。

完整示例代码：
- **单臂** (7D): `eval_files/inference_single_example.py`
- **双臂** (14D): `eval_files/inference_dual_example.py`

## 整体架构

```
┌──────────────────┐         WebSocket (msgpack_numpy)         ┌─────────────────────┐
│   机械臂客户端    │  ──────────────────────────────────────▶  │  StarVLA Policy     │
│                  │                                           │  Server (GPU)       │
│  1. 相机采集图像  │  ◀──────────────────────────────────────  │                     │
│  2. 发送图像+指令 │     归一化动作 [B, T, action_dim]          │  模型推理 → 输出     │
│  3. 反归一化动作  │                                           │  normalized_actions │
│  4. env.step()   │                                           └─────────────────────┘
│     执行动作      │
│  (位姿+夹爪)     │         action_dim: 单臂=7, 双臂=14
└──────────────────┘
```

**核心流程：**
1. 客户端从相机获取图像（多视角，`np.ndarray`, `uint8`, `(H, W, 3)`）
2. 通过 WebSocket 发送图像 + 语言指令给 server
3. Server 返回归一化动作 `normalized_actions`，shape `[B, T, action_dim]`（单臂 action_dim=7，双臂 action_dim=14）
4. 客户端反归一化后得到真实动作
5. 调用 `env.step(action)` 统一执行位姿控制和夹爪控制

## 1. 启动 Policy Server

```bash
# 在 starVLA_franka 根目录下
bash examples/Franka/eval_files/run_policy_server.sh
```

`run_policy_server.sh` 核心内容：
```bash
export PYTHONPATH=$(pwd):${PYTHONPATH}
CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
    --ckpt_path your_checkpoint.pt \
    --port 5694 \
    --use_bf16
```

Server 启动后会在指定端口监听 WebSocket 连接，等待客户端发送推理请求。

## 2. 客户端推理

可直接参考 `eval_files/inference_single_example.py` 运行，或基于以下步骤集成到你的系统：

### 2.1 连接 Server

```python
from websocketclient import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=5694)
```

`WebsocketClientPolicy` 使用 `msgpack_numpy` 序列化（支持直接传输 numpy 数组），连接后即可调用 `predict_action()`。

### 2.2 获取图像

从你的相机获取多视角图像，格式要求：
- **类型**: `np.ndarray`
- **形状**: `(H, W, 3)`，推荐 `(224, 224, 3)`
- **数据类型**: `uint8`，RGB 格式
- **数量**: 与训练时使用的视角数一致（如 1 个腕部相机 + 1 个基座相机 = 2 张图像）

```python
# 伪代码：根据你的相机 SDK 实现
images = [camera_wrist.capture(), camera_base.capture()]  # List[np.ndarray]
```

### 2.3 构造请求 & 调用

```python
request_data = {
    "examples": [{
        "image": images,           # List[np.ndarray], 多视角图像
        "lang": "Pick up the red cup and place it on the plate.",
    }]
}

result = client.predict_action(request_data)
```

### 2.4 解析响应 & 反归一化

```python
# 服务器返回格式
# result = {"data": {"normalized_actions": np.ndarray}, "status": "ok"}
# normalized_actions shape: [B, T, action_dim]，B=1, T=预测步数, action_dim=7

normalized_actions = result["data"]["normalized_actions"][0]  # [T, 7]

# 反归一化: [-1, 1] → 真实动作空间
# 公式: action = 0.5 * (normalized + 1) * (max - min) + min
# 其中 gripper 维度先做二值化: < 0.5 → -1 (关), >= 0.5 → 1 (开)
actions = unnormalize_actions(normalized_actions, action_norm_stats)
```

### 2.5 执行动作

模型输出一个 **action chunk**（T 步动作序列），逐步执行：

```python
for action in actions:
    # action: [x, y, z, roll, pitch, yaw, gripper]
    # env.step() 内部统一处理位姿控制和夹爪控制
    obs, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break
```

## 3. 动作空间说明

> ⚠️ **当前实现基于 Franka 机械臂**。对于其他机械臂，动作维度和含义可能不同（例如关节空间 N+1 维等）。请根据你的机械臂实际情况调整 `action_dim`、夹爪索引和反归一化逻辑。

### 单臂 (7D)

Franka 单臂的模型输出为 **7 维动作向量**，位姿和夹爪统一在一个向量中：
```
action = [x, y, z, roll, pitch, yaw, gripper]

• action[0:3] — 位置增量 (x, y, z)，笛卡尔坐标，单位: 米
• action[3:6] — 姿态增量 (roll, pitch, yaw)，欧拉角，单位: 弧度
• action[6]   — 夹爪控制 (-1: 关闭, 1: 打开)
```

### 双臂 (14D)

Franka 双臂的模型输出为 **14 维动作向量**，左臂 7D + 右臂 7D：
```
action = [x_l, y_l, z_l, roll_l, pitch_l, yaw_l, gripper_l,
          x_r, y_r, z_r, roll_r, pitch_r, yaw_r, gripper_r]

• action[0:6]   — 左臂位姿增量 (x, y, z, roll, pitch, yaw)
• action[6]     — 左臂夹爪控制 (-1: 关闭, 1: 打开)
• action[7:13]  — 右臂位姿增量 (x, y, z, roll, pitch, yaw)
• action[13]    — 右臂夹爪控制 (-1: 关闭, 1: 打开)
```

夹爪是动作的一部分，由 `env.step(action)` 统一执行，无需单独处理。

## 4. 动作反归一化

模型输出为 `[-1, 1]` 范围的归一化动作，需要基于训练时的统计信息反归一化。

### `dataset_statistics.json` 格式

**单臂 (7D):**
```json
{
    "franka": {
        "action": {
            "min": [x_min, y_min, z_min, roll_min, pitch_min, yaw_min, gripper_min],
            "max": [x_max, y_max, z_max, roll_max, pitch_max, yaw_max, gripper_max],
            "mask": [true, true, true, true, true, true, true]
        }
    }
}
```

**双臂 (14D):**
```json
{
    "new_embodiment": {
        "action": {
            "min": [左臂7个min值..., 右臂7个min值...],
            "max": [左臂7个max值..., 右臂7个max值...],
            "mask": [true, true, ..., true]
        }
    }
}
```

### 反归一化公式

```python
# 1. clip 到 [-1, 1]
normalized = np.clip(normalized, -1, 1)

# 2. 夹爪二值化
if action_dim == 14:  # 双臂
    normalized[:, 6] = np.where(normalized[:, 6] < 0.5, -1, 1)   # 左臂夹爪
    normalized[:, 13] = np.where(normalized[:, 13] < 0.5, -1, 1)  # 右臂夹爪
else:  # 单臂
    normalized[:, 6] = np.where(normalized[:, 6] < 0.5, -1, 1)

# 3. 线性映射到真实范围 (仅 mask=True 的维度)
action = 0.5 * (normalized + 1) * (max - min) + min
```

## 5. 适配新机械臂

如果你要在一个新的机械臂上使用 StarVLA policy server，需要实现以下接口：

> ⚠️ 以下说明以 Franka 7D 动作空间为例。你的机械臂动作维度可能不同（如关节空间控制、双臂等），需要在训练配置中设置正确的 `action_dim`，并相应调整反归一化逻辑中的夹爪索引。

### 需要实现的内容

| 模块 | 说明 |
|------|------|
| **图像获取** | 从你的相机获取图像，输出 `List[np.ndarray]`，`(H, W, 3)`，`uint8`，RGB。视角数与训练一致 |
| **`env.step(action)`** | 接收动作向量（Franka 为 7D `[x,y,z,roll,pitch,yaw,gripper]`），在内部统一处理位姿发送和夹爪控制。其他机械臂请根据自身动作空间定义调整 |
| **`env.reset()`** | 重置机械臂到初始位姿，返回初始观测 |
| **归一化统计** | 提供你的机械臂对应的 `dataset_statistics.json`（训练时自动生成），维度需与 `action_dim` 一致 |

### 不需要修改的内容

| 模块 | 说明 |
|------|------|
| **Policy Server** | 直接复用，`run_policy_server.sh` 启动即可 |
| **WebSocket 通信** | 直接复用 `WebsocketClientPolicy` |
| **反归一化逻辑** | 直接复用 `unnormalize_actions()`，只需提供对应的统计文件 |
| **请求/响应格式** | 不变：发送 `{image, lang}` → 接收 `{normalized_actions}` |

### `env.step()` 参考实现 — 单臂

```python
def step(self, action: np.ndarray):
    """
    执行 7D 动作: [x, y, z, roll, pitch, yaw, gripper]
    位姿和夹爪在同一个函数中处理。
    """
    pose_delta = action[0:6]    # 位姿增量
    gripper_cmd = action[6]     # 夹爪: -1=关, 1=开

    # 1. 位姿控制
    target_pose = self.current_pose + pose_delta * self.action_scale
    target_pose = np.clip(target_pose, self.pose_limit_low, self.pose_limit_high)
    self.robot.move_to(target_pose)

    # 2. 夹爪控制
    if gripper_cmd >= 0.9:
        self.robot.open_gripper()
    elif gripper_cmd <= -0.9:
        self.robot.close_gripper()

    # 3. 获取观测
    obs = self.get_obs()
    reward = self.compute_reward()
    done = self.check_done()
    return obs, reward, done, False, {}
```

### `env.step()` 参考实现 — 双臂

```python
def step(self, action: np.ndarray):
    """
    执行 14D 动作: [左臂7D, 右臂7D]
    双臂位姿和夹爪在同一个函数中处理。
    """
    # 左臂
    left_pose_delta = action[0:6]
    left_gripper_cmd = action[6]
    # 右臂
    right_pose_delta = action[7:13]
    right_gripper_cmd = action[13]

    # 1. 左臂位姿控制
    left_target = self.left_current_pose + left_pose_delta * self.action_scale
    left_target = np.clip(left_target, self.left_pose_limit_low, self.left_pose_limit_high)
    self.left_robot.move_to(left_target)

    # 2. 左臂夹爪控制
    if left_gripper_cmd >= 0.9:
        self.left_robot.open_gripper()
    elif left_gripper_cmd <= -0.9:
        self.left_robot.close_gripper()

    # 3. 右臂位姿控制
    right_target = self.right_current_pose + right_pose_delta * self.action_scale
    right_target = np.clip(right_target, self.right_pose_limit_low, self.right_pose_limit_high)
    self.right_robot.move_to(right_target)

    # 4. 右臂夹爪控制
    if right_gripper_cmd >= 0.9:
        self.right_robot.open_gripper()
    elif right_gripper_cmd <= -0.9:
        self.right_robot.close_gripper()

    # 5. 获取观测
    obs = self.get_obs()
    reward = self.compute_reward()
    done = self.check_done()
    return obs, reward, done, False, {}
```
