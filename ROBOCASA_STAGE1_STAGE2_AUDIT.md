# RoboCasa Stage1 / Stage2 训练链路审查

审查对象：`examples/simBenchmarks/Robocasa_tabletop`，重点是 GR1 RoboCasa Tabletop 的 VAR Stage1/Stage2 链路。

审查结论基于当前工作树、已有 checkpoint/cache、Stage2 仿真结果以及当前部署代码。本文档只记录审查结果，不代表已经修改了代码。

## 1. 总结结论

当前 RoboCasa 的 Stage1→Stage2 设计是闭环的，并且已经有完整训练产物：

```text
RoboCasa LeRobot 数据
  → 16×29 动作窗口
  → Stage1 ProductVQ tokenizer
  → 496 个离散 action token
  → 离线 token cache
  → Stage2 Qwen3-VL：图像 + 指令 + 58维状态 → token
  → 冻结的 Stage1 decoder
  → 16×29 归一化动作
  → 反归一化 / delta 累加
  → RoboCasa 环境绝对关节动作
```

当前最可靠的已验证版本是：

- Stage1：absolute action、E128、ProductVQ、16 groups、scales `[1,2,4,8,16]`
- Stage2：GBS512、100k steps、`lr=1e-4`（Qwen backbone 使用 `1e-5`）
- 仿真最佳 checkpoint：90k，24 tasks × 50 episodes，成功率 `40.33%`

delta-weighted 版本已经完成 Stage1、token cache 和 Stage2 100k 训练，但当前没有找到完整的 RoboCasa 仿真汇总，因此暂时不能证明它优于 absolute 版本。

## 2. RoboCasa 数据定义

主要数据配置位于：

- [data_config.py](examples/simBenchmarks/Robocasa_tabletop/train_files/data_registry/data_config.py)

当前 GR1 action/state 结构为：

| 组成 | 维度 |
|---|---:|
| `left_arm` | 7 |
| `right_arm` | 7 |
| `left_hand` | 6 |
| `right_hand` | 6 |
| `waist` | 3 |
| 合计 | 29 |

其他关键设置：

- 单路 `video.ego_view`
- 状态 key 共 5 组，原始状态为 29 维
- 状态经过 sin/cos transform 后为 58 维
- action horizon 为 16
- action 使用 min-max normalization
- 数据 mixture 主要覆盖 24 个 RoboCasa task

训练期的 transform 在 `data_config.py` 中依次包含：

1. state 转 Tensor
2. state sin/cos 编码
3. action 转 Tensor
4. action min-max normalization

## 3. Stage1：动作 tokenizer 训练

Stage1 入口：

- [train_var_stage1.py](starVLA/training/train_var_stage1.py)

### 3.1 数据读取

Stage1 使用 `VARStage1ActionDataset`：

- [var_stage1_action_dataset.py](starVLA/dataloader/var_stage1_action_dataset.py)

默认 `window_mode=full` 时，代码会枚举每条 trajectory 中所有完整的 16-step 窗口：

- 不加载图像和语言
- 读取动作及 delta/relative action 所需的当前 state
- 不对轨迹尾部做 padding
- 最后的不完整窗口会被排除

因此，完整窗口的最大起点是 `trajectory_length - horizon`。这能保证动作块长度严格为 16，但会减少每条 trajectory 最后若干个 base index 的采样机会。当前 late-phase weighting 只能部分补偿这一点，不能恢复被 `full` 模式排除的尾部样本。

### 3.2 action mode

action mode 逻辑位于：

- [datasets.py](starVLA/dataloader/gr00t_lerobot/datasets.py)

三种模式：

- `abs`：原始动作不变
- `delta`：第一个动作减当前 state，后续动作做相邻差分
- `rel`：所有动作减当前 state

delta 的训练目标大致是：

```text
delta[0] = action[0] - current_state
delta[t] = action[t] - action[t-1], t>0
```

Stage1 artifact 与 Stage2 cache 当前会校验 action mode 是否一致。

### 3.3 tokenizer 结构

主要实现：

- [var_action_tokenizer.py](starVLA/model/modules/action_tokenizer/var_action_tokenizer.py)

当前 ProductVQ 结构：

- action dim：29
- sequence length：16
- embedding dim：E128（旧的 E64 版本仍然存在）
- codebook size：512
- ProductVQ groups：16
- scales：`[1, 2, 4, 8, 16]`
- 每个 scale 都做 residual/coarse-to-fine quantization

token 总长度为：

```text
(1 + 2 + 4 + 8 + 16) × 16 = 496
```

也就是说，一个 16×29 的动作块会转换成 496 个离散位置，每个位置的类别数是 512。

### 3.4 Stage1 loss 与 checkpoint

Stage1 总 loss：

```text
total_loss = recon_loss
           + 0.5 × velocity_loss
           + jerk_weight × jerk_loss
           + warmed_vq_weight × vq_loss
```

delta-weighted 配置还会对以下样本提高权重：

- close task
- trajectory 后半段
- action chunk 后半段（通常从 timestep 8 开始）

该配置见：

- [delta ProductVQ Stage1 config](examples/simBenchmarks/Robocasa_tabletop/train_files/train_var_stage1_robocasa_gr1_delta_productvq_g16_s1_2_4_8_16_weighted.yaml)

其已保存的历史记录位于 NAS checkpoint 目录下，最佳记录约为：

- recon loss：`8.03e-6`
- velocity loss：`1.16e-5`
- vq loss：`8.04e-4`
- total loss：`7.82e-5`
- sample weight mean：`2.636`
- 50 epochs 完成

旧的 E64 absolute ProductVQ `epoch_027.ckpt` 是早期 Stage2 restart 文档使用的 baseline；当前 E128 absolute 是实际得到较好 RoboCasa 仿真结果的主线。

## 4. Stage1→Stage2 token cache

cache 构建入口：

- [build_var_stage2_token_cache.py](starVLA/training/build_var_stage2_token_cache.py)

cache 构建过程：

1. 加载冻结的 Stage1 artifact
2. 使用与 Stage1 相同的 deterministic dataset 和 full-window 枚举顺序
3. 用 Stage1 tokenizer 对每个动作块编码
4. 保存 token tensor、Stage1 checkpoint hash、action spec、数据集长度等 metadata

当前 RoboCasa 全量 cache 规模大约是：

- 样本数：约 566 万个 full windows
- token shape：`[5,660,058, 496]`
- dtype：当前为 `int64/torch.long`
- 文件大小：约 22.8GB

## 5. Stage2：QwenVARScaleParallel

Stage2 模型实现：

- [QwenVARScaleParallel.py](starVLA/model/framework/VLM4A/QwenVARScaleParallel.py)

### 5.1 输入

Stage2 dataset 会重新读取完整样本：

- image
- language/instruction
- transformed proprio state
- 从 cache 取得对应的 `action_tokens`

模型看到的 state 是 58 维，而不是原始 29 维。

proprio state 的使用方式：

- 取最后一个 state timestep
- 通过 proprio encoder
- 加到 pooled language/image context
- 作为额外 context token 拼接到 Qwen context

### 5.2 token prediction

Stage2 对 5 个尺度顺序预测：

```text
scale 1  →  scale 2  →  scale 4  →  scale 8  →  scale 16
```

每个 scale 内的 slots 是并行预测的。训练时使用上一尺度的 GT token embedding，即 teacher forcing；推理时上一尺度使用模型自己的 argmax 结果。

这意味着当前模型存在一定 exposure bias：训练阶段使用真实的前级 token，推理阶段使用预测 token，前级错误可能逐级传播。当前配置中 `code_condition_dropout=0`，没有 scheduled sampling。

Stage2 loss 是 496 个 token slot 上的 512 类 cross entropy，label smoothing 为 0.02。

### 5.3 训练配置

主线配置：

- [absolute E128 GBS512 config](examples/simBenchmarks/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs512.yaml)
- [delta-weighted config](examples/simBenchmarks/Robocasa_tabletop/stage2_files/train_qwen_var_productvq_g16_s124816_robocasa_delta_weighted_e128_100k_lr1e4_warmup5000_gbs512.yaml)

常见设置：

- Qwen-VL LR：`1e-5`
- 其他模块 LR：`1e-4`
- cosine scheduler，最小 LR `1e-6`
- warmup：5000 steps
- max train steps：100000
- per-device batch size：32
- gradient accumulation：2
- 8 GPU 时 global batch size 为 512
- DeepSpeed ZeRO-2
- bf16
- gradient checkpointing

当前机器环境只有 1 张 H100，旧启动脚本默认 8 个 process，因此在当前机器上不能直接按默认脚本启动 8-rank 训练。

## 6. 已有 Stage2 RoboCasa 结果

### 6.1 Absolute E128，GBS512

90k checkpoint：

- episodes：1200
- successes：484
- success rate：`40.33%`

结果文件：

- [/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs512_fullcache/robocasa_eval/steps_90000_pytorch_model_gr1_24_50eps_chunk50_robust/summary.txt](/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs512_fullcache/robocasa_eval/steps_90000_pytorch_model_gr1_24_50eps_chunk50_robust/summary.txt:25)

100k checkpoint：

- episodes：1200
- successes：470
- success rate：`39.17%`

结果文件：

- [/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs512_fullcache/robocasa_eval/steps_100000_pytorch_model_gr1_24_50eps_chunk50_robust/summary.txt](/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs512_fullcache/robocasa_eval/steps_100000_pytorch_model_gr1_24_50eps_chunk50_robust/summary.txt:25)

因此当前实际应优先保留和评估 90k，而不是默认把 100k final 当作最佳模型。

### 6.2 Absolute E128，GBS1024

100k checkpoint：

- episodes：1200
- successes：449
- success rate：`37.42%`

结果文件：

- [/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs1024_fullcache/robocasa_eval/steps_100000_pytorch_model_gr1_24_50eps_chunk50_robust/summary.txt](/root/feihong/starVLA/qwen_var_productvq_g16_s124816_robocasa_best_recon_e128_100k_lr1e4_warmup5000_gbs1024_fullcache/robocasa_eval/steps_100000_pytorch_model_gr1_24_50eps_chunk50_robust/summary.txt:25)

在当前结果中，GBS1024 不如 GBS512。

### 6.3 Delta-weighted E128

已有：

- Stage1 pure AE
- Stage1 delta ProductVQ
- 完整 delta token cache
- Stage2 100k final model

当前没有找到完整的 delta RoboCasa sim summary，因此还不能给出与 absolute 版本公平的 success-rate 对比。

## 7. 部署与 inference 链路

服务端 wrapper：

- [policy_wrapper.py](deployment/model_server/policy_wrapper.py)

当前部署逻辑是：

1. 客户端发送原始 29 维 state
2. 服务端调用训练期 transform，将 state 变为 58 维 sin/cos 输入
3. QwenVARScaleParallel 预测 496 个 token
4. Stage1 decoder 还原为 normalized action
5. PolicyNormProcessor 做 action inverse normalization
6. 如果 action mode 是 `delta`，以当前原始 state 为初值进行 `cumsum`
7. 返回给 RoboCasa 的是绝对环境动作

delta 还原代码位于：[policy_wrapper.py](deployment/model_server/policy_wrapper.py:329)。核心公式是：

```text
env_action[0] = current_state + predicted_delta[0]
env_action[t] = env_action[t-1] + predicted_delta[t]
```

当前客户端已改为发送原始 29 维状态：

- [model2robocasa_interface.py](examples/simBenchmarks/Robocasa_tabletop/eval_files/model2robocasa_interface.py:110)

服务端状态 transform 位于：

- [policy_norm_processor.py](deployment/model_server/policy_norm_processor.py:407)

## 8. 已确认的问题与风险

### 8.1 W&B secret 暴露

旧的 RoboCasa Stage2 启动脚本中存在硬编码的 W&B API key fallback：

- [run_qwen...epoch027_100k.sh](examples/simBenchmarks/Robocasa_tabletop/stage2_files/run_qwen_var_productvq_g16_s124816_robocasa_epoch027_100k.sh:17)

该 key 不应继续使用，建议立即 revoke/rotate，并改成仅从外部环境或 secret manager 注入。本文档没有记录具体 key。

### 8.2 `mse_score` 实际不是 MSE

Stage2 训练中：

```python
score = np.linalg.norm(predicted - ground_truth)
mse_score = score / np.prod(actions.shape)
```

实现位置：

- [train_starvla.py](starVLA/training/train_starvla.py:525)
- [trainer_tools.py](starVLA/training/trainer_utils/trainer_tools.py:331)

问题包括：

- 指标名称叫 `mse_score`，但实际不是平方误差均值
- 只在一次额外 batch 上评估
- 只由 main process 计算
- 没有 distributed reduction
- checkpoint retention 使用它作为 best metric

建议改成固定 validation subset，明确记录 token CE、token accuracy、decoded action MSE/MAE，并做跨 rank 聚合。

### 8.3 token cache 内存压力

当前 cache 采用 int64，约 22.8GB；每个 Stage2 rank 使用普通 `torch.load` 完整加载。8 rank 仅 cache 就可能占用约 182GB，接近主机 204.8GB 内存上限。

建议：

- token 存成 `uint16` 或 `int16`，因为类别范围只有 0–511
- 使用 mmap 或分片 cache
- 训练 cache 删除不必要的 `sample_metadata`
- 在启动前明确评估每个 rank 的 RSS

### 8.4 cache provenance 不完整

当前校验包含：

- Stage1 artifact id
- Stage1 checkpoint SHA256
- action spec
- token_dim
- source dataset length
- action mode

但还没有完整校验：

- data root
- data mix
- dataset 顺序
- trajectory/sample index fingerprint

因此如果数据集内容发生变化但样本总长度相同，仍有可能出现图像/状态与 token label 静默错配。建议生成并校验数据集 manifest hash，以及每个 sample 的 `(dataset, trajectory_id, base_index)` 指纹。

### 8.5 接口单测仍是旧契约

运行当前内置 unittest 后，4 项中 3 项通过，失败项是：

- [test_robocasa_tabletop_interface.py](tests/test_robocasa_tabletop_interface.py:64)

测试仍然断言客户端发送 `(1,58)`，但当前客户端已正确改为发送 `(1,29)` 原始 state，服务端再转成 58 维。测试需要更新，并补充：

- raw 29→server 58 的测试
- delta action cumsum 逆变换测试
- 缺少 state 时的错误测试

客户端中相关注释 `# N_history, 58 #Hack BUG` 也已经过期，应同步更新。

### 8.6 文档与启动脚本漂移

Restart 文档仍描述旧的 4 GPU、W&B disabled 默认值：

- [STAGE2_ROBOCASA_RESTART.md](examples/simBenchmarks/Robocasa_tabletop/STAGE2_ROBOCASA_RESTART.md:103)

而当前启动脚本默认 8 GPU、W&B online。文档与脚本还分别使用多套 `/root/tianyi`、`/root/feihong`、`/root/nas` 路径。

当前 E128 absolute Stage1 artifact 在 NAS 路径存在，但配置仍依赖机器特定绝对路径，复现性较差。

### 8.7 工作树没有形成可复现快照

当前分支相对远端存在 ahead/behind 差异，并有大量未提交、未跟踪的 Stage1、Stage2、部署和评估改动。因此已有训练结果不能简单等价为“当前 HEAD 的结果”。

建议每次正式训练保存：

- git commit SHA
- working-tree diff
- 完整 config
- Stage1 artifact hash
- token cache hash
- dataset manifest hash
- launcher 环境变量

### 8.8 Stage1 codebook usage 指标偏粗

Stage1 当前把所有 ProductVQ groups 的 token id flatten 后统一统计 code usage。`usage_ratio=1.0` 只能说明所有数值 id 在所有 group 合并后出现过，并不代表 16 个 codebook 分别都充分使用。

如果要用 codebook usage 做 best checkpoint 依据，应按 group 分别统计 usage，并报告 min/mean/max。

### 8.9 weighted Stage1 的 loss 不能直接和 unweighted 比

delta-weighted 配置使用 `normalization: none`，平均 sample weight 约 2.636。因此 weighted recon loss 与 absolute/unweighted Stage1 的 recon loss 不是同一量纲，不能直接按数值比较。

## 9. DeepSpeed optimizer step 核对结果

当前 [train_starvla.py](starVLA/training/train_starvla.py:553) 的 DeepSpeed 分支表面上没有显式写 `optimizer.step()`，但这不是漏更新。

仓库 `.venv` 使用 Accelerate 1.5.2；其 DeepSpeed wrapper 的 `accelerator.backward()` 会执行：

```text
engine.backward(loss)
engine.step()
```

当前代码先通过 `is_gradient_accumulation_boundary()` 判断是否应增加训练 step，再手动推进外部 scheduler。这个实现与 Accelerate/DeepSpeed 的行为是配套的，不能把“没有显式 optimizer.step”直接判定为 bug。

## 10. 建议的后续优先级

### P0

1. 立即撤销/轮换脚本中的 W&B key
2. 修正接口单测和过期注释
3. 正式训练前确认 GPU 数量与 launcher process 数量一致

### P1

1. 将 token cache 改成 uint16/int16 + mmap/分片
2. 增加 dataset manifest/index fingerprint 校验
3. 替换 `mse_score`，使用固定 validation set 和 distributed metrics
4. 固化当前最佳 90k checkpoint、Stage1 artifact、cache hash 和代码 diff
5. 更新 Restart 文档与实际 E128/GBS512/GSB1024 脚本

### P2

1. 重新跑 delta-weighted 的完整 RoboCasa 仿真
2. 按 ProductVQ group 统计 codebook usage
3. 评估 teacher forcing 与推理 exposure bias
4. 对 `full` 与 padded/late-phase sampling 做对照实验

## 11. 最终判断

从工程链路看，RoboCasa 的 Stage1/Stage2 已经不是“只有实验脚本”的半成品：Stage1 tokenizer、token cache、Stage2 token prediction、状态条件、action inverse transform 和仿真评估都已经串起来，并产生了可用结果。

如果下一步目标是继续提升 RoboCasa success rate，建议先以 absolute E128 GBS512 的 90k checkpoint 为基线，先完成指标、cache、配置和部署接口的整理，再单独比较 delta-weighted 版本。当前最需要修复的是复现和评估可信度，而不是重新设计 Stage1/Stage2 主干。

