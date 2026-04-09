# starVLA DataLoader — gr00t_lerobot

## 架构概述

将 LeRobot 格式的数据集加载、混合、预处理为训练用 batch。核心由**三个注册表**驱动：

```
YAML config (data_mix="libero_goal")
        │
        ▼
DATASET_NAMED_MIXTURES["libero_goal"]    ← 混合注册表：mixture_name → [(dataset, weight, robot_type)]
        │
        ▼  对每个 (dataset, weight, robot_type):
ROBOT_TYPE_CONFIG_MAP[robot_type]         ← 数据配置注册表：robot_type → DataConfig (modality, transforms)
ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]  ← 体现标签注册表：robot_type → EmbodimentTag (用于 Action Expert 路由)
        │
        ▼
LeRobotSingleDataset(path, modality_configs, transforms, embodiment_tag)
        │
        ▼
LeRobotMixtureDataset([(dataset, weight), ...])  ← 按 weight 采样
        │
        ▼
DataLoader(collate_fn=padding_collate_fn)
```

## 核心文件

| 文件 | 职责 |
|------|------|
| `data_config.py` | 定义各 robot_type 的 DataConfig 类（video/state/action keys, modality indices, transforms），注册到 `ROBOT_TYPE_CONFIG_MAP` |
| `embodiment_tags.py` | 定义 `EmbodimentTag` 枚举、`EMBODIMENT_TAG_MAPPING`（tag→projector index）、`ROBOT_TYPE_TO_EMBODIMENT_TAG`（robot_type→tag） |
| `mixtures.py` | 定义 `DATASET_NAMED_MIXTURES`，每个 mixture 是 `(dataset_name, weight, robot_type)` 列表 |
| `datasets.py` | `LeRobotSingleDataset`（单数据集加载）、`LeRobotMixtureDataset`（多数据集混合采样） |
| `../lerobot_datasets.py` | 入口函数 `get_vla_dataset(data_cfg)` 串联上述三个注册表 |

## 添加新 Benchmark 流程

每个 benchmark 的配置放在 `examples/<BenchmarkName>/train_files/data_registry/` 下：

```
examples/LIBERO/train_files/data_registry/
├── __init__.py          # 空文件
├── data_config.py       # DataConfig 类 + ROBOT_TYPE_CONFIG_MAP 条目
├── embodiment_tags.py   # EmbodimentTag + ROBOT_TYPE_TO_EMBODIMENT_TAG 条目
└── mixtures.py          # DATASET_NAMED_MIXTURES 条目
```

这些文件会被 `gr00t_lerobot/registry.py` 自动发现并合并到全局注册表中。

### 步骤

1. 在 `examples/<YourBench>/train_files/data_registry/` 下创建配置文件
2. 定义 DataConfig 类、embodiment_tag 映射、mixture 定义
3. 导出为标准字典：`ROBOT_TYPE_CONFIG_MAP`、`ROBOT_TYPE_TO_EMBODIMENT_TAG`、`DATASET_NAMED_MIXTURES`
4. 系统启动时自动扫描 `examples/*/train_files/data_registry/` 并合并

## DataConfig 类结构

```python
class MyRobotDataConfig:
    video_keys = ["video.cam_high"]          # 视频字段名
    state_keys = ["state.joints"]            # 状态字段名
    action_keys = ["action.joints"]          # 动作字段名
    language_keys = ["annotation..."]        # 语言指令字段名
    observation_indices = [0]                # 观测时间步采样
    action_indices = list(range(16))         # 动作时间步采样

    def modality_config(self) -> dict:       # 返回 ModalityConfig 字典
    def transform(self) -> Transform:        # 返回预处理 pipeline
```

## 数据 Batch 结构

```python
sample = {
    "action": np.ndarray [T_action, action_dim],  # e.g., [16, 7]
    "state":  np.ndarray [T_state, state_dim],     # e.g., [1, 7]  (可选)
    "image":  list[np.ndarray [H, W, C]],          # e.g., [224, 224, 3]
    "lang":   str,                                  # 任务指令
}
```
