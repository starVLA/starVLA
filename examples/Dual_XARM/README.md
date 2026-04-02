# Dual_XARM (LeRobot, Absolute Cartesian) Quick Start

这个目录是为 Dual_XARM 双臂真机数据准备的训练与推理模板工程。

已按以下前提配置：
- 数据集路径：`/root/datasets/lerobot_datasets_20260311_pick_box_action_cart`
- 数据格式：LeRobot
- state 语义：双臂关节空间（left_arm/right_arm + gripper，合计 14 维）
- action 语义：双臂绝对位姿控制（left_arm/right_arm + gripper，合计 14 维）

## 配置区分规则

当前这份 yaml：
- `examples/Dual_XARM/train_files/starvla_dual_xarm_abs_cart.yaml`
- 对应语义：`state=joint` + `action=abs_ee`

后续新增其他数据格式时，建议新建独立 yaml，并按下面命名：
- `starvla_dual_xarm_state_<state_repr>_action_<action_repr>.yaml`

示例：
- `starvla_dual_xarm_state_joint_action_abs_ee.yaml`
- `starvla_dual_xarm_state_ee_action_delta_ee.yaml`

## 目录结构

```text
examples/Dual_XARM/
  README.md
  train_files/
    modality.json
    starvla_dual_xarm_abs_cart.yaml
    run_dual_xarm_train.sh
  eval_files/
    run_policy_server.sh
    test_policy_client.py
    run_test_inference.sh
```

## 1. 训练前准备

### 1.1 配置 `modality.json`

请确认数据集里存在：

```text
/root/datasets/lerobot_datasets_20260311_pick_box_action_cart/meta/modality.json
```

如果没有，可先复制模板：

```bash
cp examples/Dual_XARM/train_files/modality.json \
  /root/datasets/lerobot_datasets_20260311_pick_box_action_cart/meta/modality.json
```

如果你的真实字段名与模板不一致，请按数据实际键名修改：
- `video.head_camera`, `video.left_wrist_camera`, `video.right_wrist_camera`
- `state.left_arm`, `state.left_gripper`, `state.right_arm`, `state.right_gripper`
- `action.left_arm`, `action.left_gripper`, `action.right_arm`, `action.right_gripper`
- `annotation.human.task_description`

### 1.2 检查 dataloader

在 `starVLA` 环境中执行：

```bash
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/Dual_XARM/train_files/starvla_dual_xarm_abs_cart.yaml
```

### 1.3 检查 framework 前向/推理

```bash
python starVLA/model/framework/QwenOFT.py --config_yaml examples/Dual_XARM/train_files/starvla_dual_xarm_abs_cart.yaml
```

## 2. 启动训练

```bash
bash examples/Dual_XARM/train_files/run_dual_xarm_train.sh
```

你通常需要先改这些参数：
- `examples/Dual_XARM/train_files/run_dual_xarm_train.sh` 中：
  - `base_vlm`
  - `num_processes`
- `examples/Dual_XARM/train_files/starvla_dual_xarm_abs_cart.yaml` 中：
  - `wandb_entity`
  - `wandb_project`
  - `datasets.vla_data.per_device_batch_size`

## 3. 启动推理服务

先改 checkpoint：
- `examples/Dual_XARM/eval_files/run_policy_server.sh` 中的 `YOUR_CKPT` 或 `your_ckpt`

启动服务：

```bash
bash examples/Dual_XARM/eval_files/run_policy_server.sh
```

## 4. 推理测试（客户端）

另开终端执行：

```bash
bash examples/Dual_XARM/eval_files/run_test_inference.sh
```

可选传入三路真实图像：

```bash
HEAD_IMAGE=/path/head.png \
LEFT_WRIST_IMAGE=/path/left.png \
RIGHT_WRIST_IMAGE=/path/right.png \
INSTRUCTION="pick up the box" \
PORT=5694 \
HOST=127.0.0.1 \
bash examples/Dual_XARM/eval_files/run_test_inference.sh
```

若不传图像，脚本会自动用随机图做链路联调。

## 5. 本模板做了哪些代码注册

为了让 `data_mix` 可直接使用，你的仓库已补充：
- `starVLA/starVLA/dataloader/gr00t_lerobot/mixtures.py`
  - 新增 `dual_xarm_pick_box_action_cart_20260311`
- `starVLA/starVLA/dataloader/gr00t_lerobot/data_config.py`
  - 新增 `DualXarmAbsCartDataConfig`
  - 新增 robot type：`dual_xarm_abs_cart`

## 6. 常见问题

4. `action_mode` 怎么选
- `action_mode: abs`：不改动作，直接学习绝对动作（当前这份 `state=joint` + `action=abs_ee` 数据推荐这个）。
- `action_mode: delta`：训练时把动作转成增量：`a[0]-s[0]`，后续步是 `a[t]-a[t-1]`。
- `action_mode: rel`：训练时把动作转成相对初始状态：`a[t]-s[0]`。

注意：当前这套数据里 state 和 action 语义不同（joint vs ee），不建议用 `delta/rel`。如果训练用 `delta/rel`，推理端也必须按同样模式把预测值还原成绝对动作再发给机器人。

1. dataloader 报 key 不存在
- 核对 `meta/modality.json` 的 `original_key` 和分段索引是否与数据一致。

2. 动作维度不匹配
- 目前默认左右臂各 7 维，共 14 维。
- 若你的 action/state 维度不同，要同步修改：
  - `modality.json` 的索引
  - `starvla_dual_xarm_abs_cart.yaml` 的 `action_dim` / `state_dim`
  - `DualXarmAbsCartDataConfig` 的 key 定义

3. 显存不足
- 降低 `per_device_batch_size`，或减少 `num_processes`。
