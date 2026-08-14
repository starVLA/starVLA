# StarVLA UMI 数据支持：当前变更说明

本文解释 `examples/human2robots/UMI4Pretraining/` 模块及
`feature/umi-data-pipeline` 分支相对 StarVLA 上游提交
`0ed0aad2c83f587714f6167ef60cf7218b786590` 的变化、使用方式、验证结果和
当前限制。

对应提交：

- `aa64732`：加入 UMI 数据获取、registry、训练配置和专用 dataloader；
- `6166cd1`：修复 mask 未进入 loss、混合 action 语义等问题，并强化下载与校验。

## 1. 目标和总体设计

本次更新希望解决四个问题：

1. 公开 UMI 数据来源多、目录和下载方式不统一；
2. 转换后的数据虽然都能表示成张量，但 action 含义并不一致；
3. 通用 LeRobot loader 缺少面向 UMI 异构数据的边界检查；
4. 单卡环境可能因为一个不可用的可选 DeepSpeed 安装而无法启动。

整体数据流如下：

```text
公开 HF / 直链数据
        │
        ▼
一键下载、断点续传、完整性校验
        │
        ▼
外部转换流程 → LeRobot v2.1 policy view
        │
        ▼
StarVLA external data_registry
        │
        ▼
通用 LeRobot decoder → UMI safety adapter
        │
        ▼
QwenOFT（mask-aware action loss）
```

更新没有把 27 个来源直接写入 StarVLA 核心 registry。UMI 配置通过外部
`data_registry` 发现机制加载，因此更新上游 StarVLA 时不需要反复修改核心
`mixtures.py`、`data_config.py` 和 `embodiment_tags.py`。

## 2. 数据获取层的变化

新增入口：

```text
examples/human2robots/UMI4Pretraining/tools/download_umi.sh
examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py
examples/human2robots/UMI4Pretraining/tools/plans/*.lock.json
```

当前锁定了 30 个物理下载源，对应 27 个独立数据族。物理源更多是因为：

- UMI-3D 由 cup、curtain、door-cup 三个仓库组成；
- LivUMI 由 Grip 和 Ego 两个仓库组成。

下载器支持：

- Hugging Face snapshot 和普通 HTTP/Google Drive 直链；
- HF 原生断点续传及直链 `.part` 续传；
- 多 source 并发和单个 HF snapshot 内部并发；
- `HF_TOKEN` 或已有 `hf auth login` 登录态；
- gated dataset 协议未接受时明确报告；
- 磁盘余量保护；
- 原子更新 `state/download_status.json`；
- 文件、大小和可选 ZIP 逐成员检查；
- 与现有 `samples_400` 目录兼容，不另外复制一份数据。

优化后的完成语义是：只有完整的 30-source plan 全部通过验证，才会创建：

```text
.all_available_400_sources_downloaded
```

只验证一个 `--families` 子集不会再错误写入全量完成标记。下载结束后还会
重新检查所有必要文件；大小正确但仍叫 `.part` 的文件会直接恢复，异常超大
的 `.part` 会被隔离而不是继续追加。

### 使用方法

```bash
cd /project/vonneumann1/UMI_data/starVLA-latest-UMI
export UMI_DATA_ROOT=/project/vonneumann1/UMI_data/samples_400

# 查看来源、空间和目标目录，不下载
bash examples/human2robots/UMI4Pretraining/tools/download_umi.sh --dry-run

# 下载并验证所有来源
bash examples/human2robots/UMI4Pretraining/tools/download_umi.sh

# 只下载指定数据族
python3 examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py download \
  --families VISTA-UMI-5K,MV-UMI,UMI-3D

# 离线深度校验
python3 examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py verify --deep
```

`download_umi.sh` 会优先复用现有 `hf`/Python 环境；依赖不存在时自动创建
轻量虚拟环境。任何 token 都不会写进仓库或锁文件。

## 3. 数据转换层的边界

仓库内的一键入口目前负责“下载和源文件校验”。已经验证过的完整转换程序仍在：

```text
/project/vonneumann1/UMI_data/training_ready/scripts/
```

转换结果位于：

```text
/project/vonneumann1/UMI_data/training_ready/starvla_data
```

没有直接把全部历史转换脚本复制进仓库，是因为其中仍包含集群绝对路径、任务恢复
状态和数据族特定逻辑。当前代码不会把“已下载”错误地描述为“已转换”。要实现真正
跨机器的一键 download → convert → audit，还需要把这些转换器统一参数化。

另外，“400 case”表示最多选择 400 个真实可用 case，而不是强制补齐。公开源不足
400 条时保留实际数量，不复制轨迹、不用 padding 伪造 case。

## 4. Registry 和 action 语义

新增外部 registry：

```text
examples/human2robots/UMI4Pretraining/train_files/data_registry/data_config.py
```

其中包含：

- 13 个 UMI robot/data config；
- 24 个 policy mixture；
- 每个 config 的视频、state、action、语言字段映射；
- 每个 robot config 的 `action_semantics`。

当前明确区分的 action 语义包括：

- absolute EEF；
- delta EEF；
- dual-arm absolute/delta EEF；
- joint action；
- dexterous-hand action；
- 若干来源的 native action space。

这些 action 即使维度相同，也不能默认放进同一个 behavior-cloning batch。新的 UMI
loader 会检查 mixture 内的 `action_semantics`；发现多种语义时默认拒绝启动。只有对
数据定义和模型 head 有明确设计时，才能使用 `allow_mixed_action_semantics: true`
显式放开。

LEAP、SenseXperience-UMI、UMI-VQA、LivUMI、ToucHD-Mani 和 UMI-Benchmark
等 observation/VLM/benchmark 数据没有可靠 policy action，仍不能注册到 BC mixture。

## 5. UMI 专用 dataloader

新增：

```text
starVLA/dataloader/umi_datasets.py
```

它不是另一套视频解码器，而是通用 `LeRobotMixtureDataset` 外面的安全适配层。
底层继续复用 StarVLA 已有的 parquet、视频、统计和 transform 逻辑。

适配层负责：

- 检查 action/state 是否为有限数，拒绝 NaN 和 Inf；
- 严格检查 action horizon、action dim 和 state dim；
- 检查图像视角存在，并可限制最大视角数；
- 清理语言中的多余空白，为缺失语言提供明确 fallback；
- 可选 action 最大绝对值与静止 chunk 检查；
- 生成 `action_mask`、`state_mask` 和 `image_mask`；
- 坏样本按 index、epoch 和 seed 确定性地重采样；
- 保持 StarVLA 模型期望的 `List[dict]` batch 格式。

action/state 维度默认从 `framework.action_model` 推导，不需要在 dataset 配置里重复。
如果两处都填写但值不一致，loader 会在启动阶段报错。

### 启用方法

训练 YAML 中设置：

```yaml
datasets:
  vla_data:
    dataset_py: umi_datasets
    strict_dimensions: true
    max_views: 2
    retry_bad_samples: 20
```

完整覆盖示例见：

```text
examples/human2robots/UMI4Pretraining/train_files/umi_loader_overrides.yaml
```

正常训练建议保持 `strict_dimensions: true`。宽松模式主要用于诊断或已实现异构
action head 的实验，不应被当成混合不同机器人 action space 的捷径。

## 6. QwenOFT loss 的变化

UMI adapter 允许短 chunk 在张量层面对齐，但 padding 不能参与监督。原 QwenOFT
直接对整个 action 张量计算平均 L1，导致 padding 位置也产生梯度。

现在的计算为：

```text
loss = sum(abs(pred - target) * action_mask) / sum(action_mask)
```

行为兼容规则：

- batch 没有 `action_mask`：保持原来的全张量平均 L1；
- batch 全部包含 mask：只计算有效 action cell；
- 同一个 batch 只有部分样本包含 mask：直接报错；
- mask 没有任何有效位置或形状不匹配：直接报错。

这修复了“loader 生成 mask，但模型不使用 mask”的训练正确性问题。

## 7. 单 GPU / DeepSpeed 兼容变化

上游训练入口会无条件创建 `DeepSpeedPlugin`。若机器安装了损坏或与 CUDA 不兼容的
DeepSpeed，即便只是单 GPU 训练也会失败。

现在可以使用：

```bash
STARVLA_DISABLE_DEEPSPEED=1 accelerate launch \
  --config_file examples/human2robots/UMI4Pretraining/train_files/accelerate_single_gpu.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml examples/human2robots/UMI4Pretraining/train_files/starvla_dexwild_smoke.yaml
```

未设置该环境变量时，官方 DeepSpeed 行为保持不变。

## 8. 已完成的验证

在远端最新 StarVLA 和真实转换数据上验证过：

- 外部 registry：24 个 UMI mixture、13 个 robot config；
- DexWild：346 trajectories、123,485 steps；
- 真实 batch：action `(8, 23)`，184/184 action cell 有效；
- state `(1, 23)`，23/23 state cell 有效；
- 两个图像视角、语言和 `new_embodiment` tag 正常；
- action/state 维度从 model config 自动推导成功；
- QwenOFT masked L1 数值测试通过；
- 下载器 8 项回归测试通过；
- 此前同一 DexWild 数据与 QwenOFT 已完成 20/20 steps GPU smoke，checkpoint 和
  final model 均成功保存。

本轮优化没有再次提交 GPU 作业，因为登录节点没有 GPU，额外 2-step 作业还会生成
约 9 GB 重复 checkpoint；当前采用真实 batch、loss 数值和既有 20-step smoke 组合
验证。

## 9. 当前限制和后续工作

目前仍需关注：

1. 完整转换 pipeline 尚未全部迁入仓库并参数化；
2. `max_abs_action` 应在归一化后按 action semantics 分别设置，不能全局使用一个阈值；
3. `image_mask` 已生成，但所有模型 head 是否消费该 mask 仍需逐一确认；
4. observation/VLM-only 数据需要独立 loader/训练目标；
5. 真正跨语义联合训练需要 embodiment-aware action head 或显式 canonical action
   representation，不能依赖维度 padding；
6. GitHub 分支目前只存在于本地和远端计算集群，尚未推到官方仓库。GitHub SSH
   身份 `TruemanV5` 没有 `starVLA/starVLA` 写权限，且对应 fork 尚未创建。

建议后续顺序：

1. 创建 fork 并推送 `feature/umi-data-pipeline`；
2. 将转换器整理成 `convert` 子命令，移除所有集群硬编码；
3. 为 abs EEF、delta EEF、joint 和 hand 分别跑小规模训练/评测；
4. 再设计跨 embodiment 的统一 action tokenization 或多 head 路由。
