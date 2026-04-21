# World Model 模块设计 (`starVLA/model/modules/world_model/`)

## 定位

World Model 模块封装用于动作预测的 **世界模型** 后端。
与 VLM 模块不同，World Model 在预训练阶段关注物理世界建模
（如视频预测、空间推理），提供对物理规律更强的先验理解。

## 为什么 Cosmos-Reason2 用 Qwen3VL 架构？

Cosmos-Reason2 是 NVIDIA 基于 Qwen3-VL 架构进行物理推理微调的模型。
底层使用相同的 `Qwen3VLForConditionalGeneration` 和 `Qwen3VLProcessor`，
因此接口与 VLM wrapper 完全兼容。区别在于：

- **预训练任务不同**：Cosmos 侧重物理推理和因果理解
- **表征含义不同**：hidden_states 编码了更丰富的时空动态信息
- **应用场景**：更适合需要物理理解的操作任务

## 接口规范

与 VLM 模块完全对齐（方便 framework 层透明替换）：

| 方法 | 签名 | 用途 |
|------|------|------|
| `__init__` | `(config)` | 加载预训练模型 + processor |
| `forward` | `(**kwargs) -> CausalLMOutputWithPast` | 前向传播 |
| `generate` | `(**kwargs)` | 自回归生成 |
| `build_qwenvl_inputs` | `(images, instructions)` | 构建模型输入 |

## 工厂函数

```python
from starVLA.model.modules.world_model import get_world_model
wm = get_world_model(config)  # 根据模型名路由
```

## 现有实现

| 文件 | 类名 | 支持的模型 | 架构 |
|------|------|-----------|------|
| `CosmosReason2.py` | `_CosmosReason2_Interface` | nvidia/Cosmos-Reason2-2B | Qwen3-VL (自回归 LM), hidden=2048 |
| `CosmoPredict2.py` | `_CosmoPredict2_Interface` | nvidia/Cosmos-Predict2-2B | DiT (扩散 Transformer), hidden=4096 |

## Framework 层使用

World Model 由 `framework/WM4A/` 下的框架使用：

```python
# WM4A/CosmosGR00T.py
from starVLA.model.modules.world_model import get_world_model

class Cosmos_GR00T(baseframework):
    def __init__(self, config):
        self.qwen_vl_interface = get_world_model(config=self.config)
        # 后续与 VLM4A 框架完全一致
```

## 架构全景

```
starVLA/model/
├── modules/
│   ├── vlm/                # VLM 后端 (Qwen2.5-VL, Qwen3-VL, ...)
│   ├── world_model/        # 世界模型后端 (Cosmos-Reason2, ...)
│   ├── action_model/       # 动作头 (DiT, FAST, L1Regression, ...)
│   ├── dino_model/         # DINO 空间编码器
│   └── projector/          # QFormer 等投影模块
├── framework/
│   ├── base_framework.py   # 基类 + build_framework() + 自动导入
│   ├── share_tools.py      # 配置合并等工具
│   ├── VLM4A/              # VLM-for-Action 框架
│   │   ├── QwenGR00T.py    # VLM + DiT flow-matching
│   │   ├── QwenFast.py     # VLM + FAST 离散 token
│   │   ├── QwenAdapter.py  # VLM + Query adapter
│   │   ├── QwenPI.py       # VLM + Layer-wise FM
│   │   ├── QwenDual.py     # VLM + DINO + DiT
│   │   ├── LangForce.py    # VLM + Dual-branch DiT
│   │   ├── ABot_M0.py      # VLM + VGGT + DiT
│   │   └── M1.py           # VLM + DINO + QFormer + DiT
│   └── WM4A/               # World-Model-for-Action 框架
│       ├── CosmosGR00T.py           # CosmosReason2 + Flowmatching
│       └── CosmoPredict2GR00T.py    # CosmosPredict2 (DiT) + Flowmatching
└── tools.py                # Registry, 归一化工具
```
