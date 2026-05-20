# CALVIN ABC → D 技术路线选择论证与 Failure Pattern 分析报告

最后更新：2026-05-20  
负责人：WMH  
代码范围：`examples/calvin_autoresearch`  
任务目标：基于 CALVIN ABC 训练数据，完成 CALVIN D closed-loop evaluation，并构建、验证和优化合规的 StarVLA baseline。

---

## 1. Executive Summary

本报告围绕 CALVIN ABC → D 任务，对多条 StarVLA 技术路线进行了比较和论证。整体约束是：训练仅使用 CALVIN ABC，CALVIN D 仅用于 closed-loop evaluation，不使用任何上游 action-trained checkpoints，例如 LIBERO、RoboTwin、RoboCasa、Behavior 或 CALVIN-D action checkpoints。允许使用 base VLM checkpoint，当前主要 base model 为 `Qwen3-VL-4B-Instruct-Action`。

实验结果表明，baseline 的主要失败不是由简单的 action noise 或 action jitter 导致的，而是集中在 hard atomic tasks、first-step failure、contact/affordance grounding，以及部分 left/right directional tasks 上。也就是说，模型首先需要更可靠地理解和执行单个困难任务，才能进一步提升 long-horizon chained success。

当前最可靠、已经验证的 WMH 主路线是：

**hard-task balanced ABC training + controlled language paraphrase + task-aware image augmentation**。

该路线将 `base8k n300` 的 `avg_seq_len` 从 `1.05` 提升到 `1.85`，将 `SR@5` 从 `3.7%` 提升到 `12.0%`。这说明，与其单纯延长 baseline 训练，不如优先改善 hard-task data distribution 和任务相关的数据增强。

LoRA 也被证明有效。`LoRA2000 n300` 达到 `avg_seq_len=1.63`、`SR@5=9.7%`，明显优于 baseline，但单独使用 LoRA 仍弱于 hardv2 augmented data route。

MoE action head 具有潜力，尤其是 GTY MoE95k 在 N=100 的 D evaluation 上达到 `avg_seq_len=1.91`、`SR@5=12.0%`。但 MoE-Adaptive 同时暴露出明显 task bias：它显著提升 lightbulb tasks，却严重伤害 drawer tasks。因此 MoE 不能只看 aggregate score，必须通过 per-task metrics 做验证。

当前最高潜力候选路线是：

**GTY MoE95k + fresh Qwen LoRA + WMH hardv2 augmented ABC data**。

该路线尚未被 D n300/n1000 验证，因此不能直接作为最终结论。它应当先与 WMH hardv2 aug、LoRA2000 和 GTY MoE95k 做公平对比，再决定是否升级为最终主路线。

---

## 2. Problem Setting and Compliance Boundary

本任务的目标是构建一个合规的 StarVLA baseline，并在 CALVIN ABC → D 设置下提升 closed-loop performance。

具体含义是：

- 训练数据只使用 CALVIN ABC。
- CALVIN D 只用于 closed-loop evaluation，不参与训练。
- 不允许从上游 action-trained checkpoints 初始化，例如 LIBERO、RoboTwin、RoboCasa、Behavior 或 CALVIN-D action checkpoints。
- 允许使用 base VLM checkpoint。
- 我们自己在 CALVIN ABC 上训练得到的 checkpoints 可以作为后续训练的 continuation source。

主要共享路径如下：

```text
private repo:
/inspire/qb-ilm2/project/26summer-camp-10/26220172/WMH/starVLA

shared project root:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin

shared base model:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/models/base/Qwen3-VL-4B-Instruct-Action

shared ABC LeRobot data:
/inspire/qb-ilm2/project/26summer-camp-10/public/seven/starvla_calvin/shared/datasets/calvin_lerobot

official CALVIN D eval data:
/inspire/qb-ilm2/project/26summer-camp-10/public/inspire_shared/calvin_d_d
```

---

## 3. Evaluation Protocol and Metrics

主要指标是 CALVIN D 上的 chained success。CALVIN evaluation 通常由多个 5-task sequence 组成，模型需要连续完成多个 instruction。相比单个 task success，chained success 更能反映 closed-loop policy 的稳定性、任务理解能力和误差累积问题。

核心指标包括：

- `avg_seq_len`：每个 5-task sequence 中平均连续完成的子任务数量。
- `chain_sr@k`：完成至少连续 k 个 instruction 的 sequence 占比。
- `per_atomic_task`：每个 atomic CALVIN task 的成功率。
- `failure_position`：任务链首次失败的位置。
- `conditional_success`：在已经成功到达第 k 个位置的条件下，继续完成后续任务的概率。
- `action_stats`：动作幅值、action saturation、jitter、gripper switch rate。
- `near_miss`：失败 rollout 是否意外完成相关任务或其他任务。

评估策略如下：

- `n100` / `n300` 用于快速探索和候选分支排序。
- `n1000` 用于稳定 baseline 或 finalist branch 的正式验证。
- aggregate metrics 不能单独作为路线选择依据，必须结合 per-task metrics、failure position、near-miss 和 action diagnostics。

---

## 4. Main Results

以下结果均为 CALVIN D evaluation，除非特别说明。

| Branch | Checkpoint / Eval | N | Avg Seq Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WMH state8+connector 8k | `eval_8k_state8_connector_steps8000_fast_w4x8_d_n1000_0519_143010` | 1000 | 1.086 | 53.2% | 27.9% | 15.2% | 8.0% | 4.3% | Stable baseline |
| WMH base8k n300 | `eval_compare_d_n300_0520_031701/base8k` | 300 | 1.050 | 54.0% | 25.7% | 13.7% | 8.0% | 3.7% | 与 n1000 趋势一致的小规模评估 |
| WMH LoRA2000 | `eval_compare_d_n300_0520_031356/lora2000` | 300 | 1.630 | 64.0% | 43.7% | 27.3% | 18.3% | 9.7% | 相比 base8k 明显提升 |
| WMH hardv2 aug | `eval_compare_d_n300_0520_031701/aug_hardv2` | 300 | 1.847 | 72.0% | 51.0% | 29.3% | 20.3% | 12.0% | 当前 WMH 最强 verified branch |
| WMH mirror hardv2 | manually aggregated from `eval_compare_d_n300_0520_033750/mirror_hardv2/worker_*` | 300 | 1.753 | 72.7% | 47.7% | 29.0% | 16.3% | 9.7% | first-step 有帮助，但整体不如 non-mirror hardv2 |
| WMH MoE-Adaptive | `eval_moe_adaptive_d_n300_0520_043745` | 300 | 1.397 | 67.7% | 38.0% | 19.0% | 10.3% | 4.7% | mixed result；提升部分 light tasks，但伤害 drawer |
| GTY MoE95k | `eval_abc_augmented_moe_GTY_0519_092147_abc_to_d_n1000_0519_145552/results.json` | 100 | 1.910 | 76.0% | 54.0% | 32.0% | 17.0% | 12.0% | 强外部团队分支，但当前统计为 N=100，不能与 n300/n1000 完全公平比较 |

说明：虽然 GTY MoE95k 的路径名中包含 `n1000`，但当前可用/聚合的结果为 `N=100`，因此只能作为强潜力参考，不能直接与 WMH 的 n300/n1000 分支做完全公平的数值比较。

当前仍在训练、尚未进行 D evaluation 的分支：

- `abc_moe95k_lora_aug_3h_bs96_0520_045012`：基于 GTY MoE95k，加入 fresh LoRA，不使用 mirror，`BATCH_SIZE=96`，`NUM_PROCESSES=8`。
- `abc_moe95k_lora_mirror_3h_bs96_0520_045300`：同上，但加入 left/right mirror augmentation。

两个分支都已经至少产生 `steps_3000_pytorch_model.pt`，训练速度约为 `1.5-1.6s/step`，但尚未在 CALVIN D 上评估。

---

## 5. Candidate Technical Routes and Main Route Selection

本次探索不是预设单一路线，而是在相同合规边界下比较多个候选方向。路线选择标准不仅包括 `avg_seq_len`，还包括以下因素：

- 是否直接针对主要 failure pattern；
- 是否在 H200 training 下稳定；
- 是否能与已有 checkpoint / training pipeline 组合；
- 是否存在 task regression；
- 是否满足 ABC-only training 和 D-only evaluation 的约束。

| 候选路线 | 核心变化 | 实验证据 | 优势 | 风险 / 局限 | 决策 |
| --- | --- | --- | --- | --- | --- |
| A. Frozen-Qwen + GR00T action-head scaling | 保持 Qwen frozen，只训练 StarVLA QwenGR00T action head | baseline 和 state8+connector 约为 `avg_seq_len=1.05-1.09`，`SR@5=3.7-4.3%` | 干净、合规、易复现 | 不能解决 hard tasks 和 long-chain failure | 仅保留为 reference baseline |
| B. State/proprio + connector | 加入 8-D robot state，训练 interface/action head | state8+connector n1000: `avg_seq_len=1.086`, `SR@5=4.3%` | 理论上有助于 robot pose、gripper state 和 subtask progress | 当前结果不能证明 state 被有效使用，需要 zero/shuffle sanity tests | 作为中期支持方向，不作为当前主路线 |
| C. Hard-task balanced data + controlled language/image augmentation | oversample hard atomic tasks；加入 canonical mapping、paraphrases 和 task-aware light visual augmentation | hardv2 aug n300: `avg_seq_len=1.847`, `SR@1=72.0%`, `SR@5=12.0%` | 当前 WMH 最强 verified branch；直接针对主要 failure mass | sampler、paraphrase、image aug 的收益耦合，需要后续 ablation | 当前 verified main route |
| D. Left/right mirror augmentation | mirror images，交换 left/right language，并对 action axes 做 sign transform | mirror hardv2 n300: `avg_seq_len=1.753`, `SR@1=72.7%`, `SR@5=9.7%` | 不引入额外数据即可增强 left/right coverage | 略弱于 non-mirror hardv2；可能存在 wrist/action sign inconsistency | 保留为独立分支，不作为默认 |
| E. Qwen LoRA | 在 Qwen 后几层 attention 上加入 small LoRA，同时保持 base Qwen 基本隔离 | LoRA2000 n300: `avg_seq_len=1.630`, `SR@5=9.7%` | 用少量可训练参数增强 grounding 和 representation adaptation | LoRA alone 不如 hardv2 data route | 与更强数据分布或 action head 组合，不单独作为最终路线 |
| F. MoE / adaptive action head | 使用更强或更专门化的 action-head capacity，而不是单一 GR00T DiT head | MoE-Adaptive n300: `avg_seq_len=1.397`，但 light tasks 很强；GTY MoE95k n100: `avg_seq_len=1.91` | 对 task-specific motion modes 有较高 upside | Adaptive branch 严重伤害 drawer；GTY n100 不可直接与 n300/n1000 比较 | 有潜力，但必须做 n300/n1000 和 per-task verification |
| G. GTY MoE95k + fresh LoRA + hardv2 data | 从合规 ABC-trained team MoE checkpoint 出发，加入 fresh LoRA，并用 WMH hardv2 ABC data 训练 | 当前仍在训练，D n300 pending | 组合当前最强三个信号：hard-task data、LoRA、MoE head | 可能扰动强 MoE policy；mirror variant 可能引入 geometry noise | 当前 highest-upside experimental route |

### 5.1 Verified Main Route

当前已经验证的主路线是：

**hard-task balanced ABC training + controlled language paraphrase + task-aware image augmentation**。

选择它的理由是：

1. 它是目前 WMH 分支中最强的 verified result。
2. 它直接针对 failure analysis 中暴露出的主要问题：hard atomic tasks 和 first-step failure。
3. 它显著提升了 `SR@1`，说明模型对单个困难任务的理解和执行能力确实改善。
4. 它同时提升 `SR@5`，说明这种改善能够传递到 chained evaluation 中。
5. 它完全满足 ABC-only training 的合规约束。

关键对比：

```text
base8k n300:
  avg_seq_len = 1.05
  SR@1 = 54.0%
  SR@5 = 3.7%

hardv2 aug n300:
  avg_seq_len = 1.847
  SR@1 = 72.0%
  SR@5 = 12.0%
```

### 5.2 Highest-Upside Candidate Route

当前最高潜力候选路线是：

**GTY MoE95k + fresh Qwen LoRA + WMH hardv2 augmented ABC data**。

它的逻辑是组合三个已经观察到的有效信号：

- hard-task data distribution 能修复最大 failure mass；
- LoRA 能提升 representation adaptation，效果优于仅训练 connector；
- GTY MoE95k 提供了比 plain GR00T branch 更强的 ABC-trained action-head initialization。

但该路线还不能被称为最终主路线，因为它还缺少 D n300 / D n1000 验证。尤其需要确认：

- 它是否真的超过 hardv2 aug；
- 它是否保留了 MoE 对 light tasks 的优势；
- 它是否避免了 MoE-Adaptive 中出现的 drawer regression；
- mirror variant 是否引入 sign/view inconsistency。

### 5.3 Not Prioritized Route

不建议继续优先投入的路线是：

**单纯延长 frozen-Qwen + GR00T baseline 的训练时间**。

原因是当前 detailed metrics 表明，主要瓶颈不是随机动作噪声或简单 undertraining，而是 task grounding、contact/affordance execution 和 hard-task data coverage。若不改变数据分布、representation adaptation 或 action-head capacity，单纯增加训练步数大概率不是最高效的 GPU 使用方式。

---

## 6. Failure Pattern Analysis

### 6.1 Chain-Level Failure Pattern：Early-Chain Collapse

第一个主要失败模式是 early-chain collapse。也就是说，模型经常在任务链的第一个 instruction 就失败，导致后续 long-horizon composition 根本没有机会被测试。

在 `base8k n300` 中，`138/300` 条 sequence 在第一个 instruction 失败，接近一半 evaluation 被 first-step failure 截断。`hardv2 aug` 将 first-position failures 降低到 `84/300`，这解释了它大部分 aggregate gain。

| Branch | Fail @1 | Fail @2 | Fail @3 | Fail @4 | Fail @5 | Completed All 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base8k n300 | 138 | 85 | 36 | 17 | 13 | 11 |
| hardv2 aug n300 | 84 | 63 | 65 | 27 | 25 | 36 |
| LoRA2000 n300 | 108 | 61 | 49 | 27 | 26 | 29 |
| MoE-Adaptive n300 | 97 | 89 | 57 | 26 | 17 | 14 |

解释：

- `SR@1` 是判断模型是否理解 D 中单个 atomic task 的敏感指标。
- hardv2 将 `SR@1` 从 `54.0%` 提升到 `72.0%`，说明 hard-task coverage 对提升第一步成功率非常关键。
- 但 long-horizon robustness 仍未解决。即使 hardv2 aug，也只有 `36/300` 条 sequence 完整完成 5 个任务。

因此，当前路线的第一优先级不是直接追求复杂 long-horizon reasoning，而是先提升 hard atomic task 的可靠性。只有 `SR@1` 和 per-task success 足够高，后续 long-chain success 才有提升空间。

### 6.2 Task-Level Failure Pattern：Failures Are Not Uniform

失败并不是均匀分布在所有任务上，而是集中在 slider、light/LED、drawer、directional block manipulation 等类别上。不同路线修复的任务子集不同，因此 aggregate score alone 是不够的。

| Failure Group | 证据 | 技术含义 | 当前路线响应 | 下一步验证 |
| --- | --- | --- | --- | --- |
| Slider left/right | base8k top failures 包含 `move_slider_right=30`, `move_slider_left=28`；hardv2 仍有 `move_slider_left=36`, `move_slider_right=34` failures | 小接触目标 + 方向性歧义；generic training 不足以解决 | 保留 left/right-specific diagnostics；mirror 只作为 controlled branch | 对 slider-left/right 单独做 subset eval；检查 mirror action sign convention |
| Lightbulb on/off | base8k `turn_off_lightbulb=19%`, `turn_on_lightbulb=17%`；hardv2 提升到 `39%/41%`；MoE-Adaptive 达到 `96%/93%` | 需要 affordance understanding 和较专门的 motion primitive | 将 hard-task data 与更强 action head 结合 | 验证 MoE + hardv2 是否保留 lightbulb 优势，同时不引入 drawer regression |
| Drawer | hardv2: `close_drawer=100%`, `open_drawer=99%`；MoE-Adaptive: `close_drawer=0%`, `open_drawer=56%` | architecture change 可能提升一类技能，同时破坏另一类技能；drawer 是重要 regression detector | 所有 MoE/LoRA 分支都必须检查 drawer per-task success | 将 drawer 作为强制 regression test，不允许只看 avg_seq_len |
| Directional push/right tasks | hardv2 改善 `push_*_right`，但提升不均匀；mirror 结果 mixed | geometry transform、language swap 和 action sign 必须精确一致 | mirror 不作为默认，只作为独立分支 | 生成 mirror before/after visualization；做 sign-convention unit test |
| Stack/lift/place tasks | 多个分支仍存在 timeout failures | gripper timing、contact sequencing 和闭环纠偏仍较弱 | 暂不把 smoothing 作为主策略，后续考虑 gripper head / history | 分析 rollout video、gripper switch timing 和 predicate distance |

该表说明，当前路线不是简单追求 overall score，而是根据不同 failure group 选择不同实验响应。hardv2 适合作为 verified main route，是因为它对多个主要 failure group 都有稳定收益；MoE 适合作为 candidate，是因为它对部分 motion primitive 有强信号，但也有明显 regression risk。

### 6.3 Timeout and Near-Miss Pattern

详细 n300 metrics 显示，失败 subtasks 大多是在 evaluation horizon 处 timeout，即 `failure_step=360`。这说明大多数失败不是 policy 立即崩溃，而是模型在完整 rollout 中尝试了某种动作，但最终没有满足 task predicate。

这类失败更符合以下解释：

- 模型没有准确 grounding 当前 instruction；
- 模型找到了大致区域，但没有对准关键 affordance；
- contact timing 或 gripper timing 错误；
- 执行了相关但不完全正确的 motion primitive；
- 在 left/right 或 task state 上混淆。

Near-miss 结果也支持这一点：

| Branch | Failed Subtasks | Any-Task Near Miss | Related-Task Near Miss |
| --- | ---: | ---: | ---: |
| base8k | 289 | 10.0% | 4.8% |
| hardv2 aug | 264 | 11.4% | 5.3% |
| LoRA2000 | 271 | 11.8% | 5.5% |
| MoE-Adaptive | 286 | 22.0% | 9.1% |

MoE-Adaptive 的 near-miss rate 明显更高，说明它经常对环境产生有效操作，但没有完成被要求的 predicate。这是一个重要信号：MoE 的 motion capacity 可能更强，但 grounding 或 gating 可能偏离目标任务。因此 MoE 不能只凭部分 task 的高成功率直接作为最终路线，需要 per-task verification。

### 6.4 Action-Diagnostic Pattern：Action Noise 不是主瓶颈

从 action diagnostics 看，性能提升不能简单归因于动作更平滑。

| Branch | Jitter Mean L2 | Gripper Switch Rate | 解释 |
| --- | ---: | ---: | --- |
| base8k | 0.153 | 2.04% | baseline motion scale |
| hardv2 aug | 0.151 | 2.16% | smoothness 与 baseline 接近，但性能显著更好 |
| LoRA2000 | 0.143 | 1.89% | 略微更平滑，chain success 更好 |
| MoE-Adaptive | 0.181 | 1.95% | 动作更激进；部分 task 有收益，但 drawer regression 明显 |

关键结论：

- hardv2 aug 与 base8k 的 jitter 非常接近，但 hardv2 的 `avg_seq_len` 和 `SR@5` 明显更高。
- LoRA 稍微更平滑，但提升幅度小于 hardv2。
- MoE-Adaptive 的动作更激进，反而在 drawer 上退化。

因此，当前主要瓶颈不是“动作太抖”，而是：

1. 模型对少数 D-critical hard tasks 的识别和执行不足。
2. slider handle、drawer handle、light switch 等 contact affordance 不稳定。
3. 某些 action-head architecture 可以表达有用 motion primitive，但可能造成 task-specific overfitting 或 regression。
4. long-chain failure 是 atomic failure 叠加后的结果；修复 `SR@1` 是必要条件，但不是充分条件。

### 6.5 State/Proprio Underuse Pattern

state/proprio 在理论上很重要，因为机器人位姿、gripper state、drawer/slider/light state 和当前 subtask progress 很难完全从单帧 RGB + language 中稳定推断。

但当前 state8+connector 结果并没有证明 proprio 被有效利用：

```text
state8+connector n1000:
  avg_seq_len = 1.086
  SR@1 = 53.2%
  SR@5 = 4.3%
```

它与 base8k n300 接近，因此目前只能说明 state path 已接通，不能说明模型真正使用了 state。

可能原因包括：

- action head 仍主要依赖 visual/token features；
- state normalization 或 embedding scale 不合适；
- 没有 auxiliary objective 强迫模型理解 state；
- 训练步数或数据分布不足以让 state 起作用。

下一步必须做 sanity tests：

- state zero eval；
- state shuffle eval；
- 只训练 `state_projector + action head`；
- 训练 connector without state 作为对照；
- 比较 `state_projector`、`qwen_vl_interface`、`action_model` 的 gradient norms。

在这些验证完成前，state/proprio 应作为中期增强方向，而不是当前主路线的核心依据。

### 6.6 How Failure Pattern Drives Route Selection

Failure Pattern 直接决定了当前路线选择：

- 因为失败集中在少数 hard tasks，所以使用 hard-task balanced sampling。
- 因为 hard tasks 通常具有模板化语言和小视觉 affordance，所以使用 controlled paraphrases 和 task-aware image augmentation，而不是强 generic augmentation。
- 因为 representation adaptation 有帮助但不足以单独解决问题，所以 LoRA 应与更好的数据分布或 action head 结合。
- 因为 MoE 可以解决某些 motion modes，但也会引入 regression，所以 MoE 必须作为高潜力候选，并通过 per-task n300/n1000 验证。
- 因为 action jitter 不是主因，所以 smoothing-only method 不是当前最高优先级。

由此形成路线层级：

1. **Verified WMH main route**：hardv2 augmented ABC training。
2. **Highest-upside candidate**：GTY MoE95k + fresh LoRA + hardv2 ABC data。
3. **Diagnostic branch**：left/right mirror，仅在 directional task subsets 上验证有效后再考虑使用。
4. **Deferred support work**：proprio/state sanity checks、gripper head、auxiliary state/success prediction、short history。

---

## 7. Route Decision and Next Experiments

### 7.1 当前路线决策

当前最终可陈述的路线决策是：

在合规约束下，本任务不应优先继续 scale frozen-Qwen + GR00T baseline，而应优先采用 failure-driven data/model adaptation。当前最可靠的已验证路线是 hard-task balanced ABC training，并结合 controlled language paraphrase 和 task-aware image augmentation。该路线直接针对 first-step failure 和 hard atomic task failure，并已在 D n300 上显著提升 `avg_seq_len` 和 `SR@5`。

同时，LoRA 和 MoE 具有进一步提升空间，但不应被单独视为最终解法。LoRA 需要与 stronger data route 结合；MoE 需要通过 per-task verification 防止 drawer 等任务退化。当前最高潜力实验是 GTY MoE95k + fresh Qwen LoRA + hardv2 augmented ABC data。

### 7.2 Immediate Next Steps

1. 完成当前 `MoE95k + LoRA + aug` 和 `MoE95k + LoRA + mirror` 训练。
2. 对两个 latest checkpoints 跑 CALVIN D n300。
3. 与以下分支比较：
   - WMH hardv2 aug n300；
   - WMH LoRA2000 n300；
   - GTY MoE95k n100 / n300，如果可用。
4. 如果其中一个分支超过 hardv2 aug，则继续跑 D n1000。
5. 保存所有 final configs、logs、checkpoint paths 和 evaluation reports。

### 7.3 Near-Term Verification

1. 对 GTY MoE95k 跑可比较的 n300/n1000 evaluation，避免只依赖 n100 结果。
2. 对 state-aware checkpoints 做 state zero/shuffle eval，验证 proprio 是否真的被使用。
3. 对 mirror branch 只在 left/right task subsets 上做 targeted evaluation。
4. 基于 n300 per-task failures 调整 hard-task sampler weights。
5. 加入 separate gripper head 或 gripper hysteresis ablation。
6. 对 MoE/LoRA 分支强制检查 drawer regression。

### 7.4 Longer-Term Directions

1. 加入 lightweight auxiliary prediction heads：
   - drawer open/closed；
   - slider left/right；
   - light/LED on/off；
   - success/value prediction。
2. 加入 short observation/state history，增强闭环纠偏能力。
3. 设计 wrist-specific augmentation 和 camera-consistent geometry transforms。
4. 建立 automatic report comparison script，用于比较 `base -> candidate` 的任务级变化。
5. 只对 finalist branches 使用 n1000，不对每个 probe 都跑完整 n1000，以节省评估时间。

---

## 8. Appendix: Exploration Path and Detailed Branch Notes

本节保留主要探索过程，作为正文技术路线论证的补充材料。

### 8.1 Baseline: Qwen3-VL + QwenGR00T Action Head

选择方向：

- 使用 `Qwen3-VL-4B-Instruct-Action` 作为 base VLM。
- 使用 StarVLA QwenGR00T action head。
- 只在 CALVIN ABC 上训练。
- 在 official CALVIN D 上评估。

选择原因：

- 满足 base model allowed、upstream action-trained checkpoints not allowed 的规则。
- 提供加入 state、adapter、LoRA、augmentation 前的 clean reference point。
- QwenGR00T 与 StarVLA code path 和 action chunking interface 匹配。

训练分支：

```text
abc_pretrain_qwen3vl_gr00t_headonly_h200_60k_0518_163437
```

观察结果：

- 建立了第一条可工作的 closed-loop D eval pipeline。
- 性能较低，尤其在 long chains 和 hard first subtasks 上明显不足。
- 证明 pipeline correctness alone 不足以取得好结果，必须进行 model/data 改进。

结论：

- 保留该分支作为 compliance baseline 和 infrastructure baseline。
- 不建议继续主要投入 frozen/action-only baseline scaling。

### 8.2 Evaluation Infrastructure

选择方向：

- 围绕 StarVLA websocket policy server 搭建一键 CALVIN D evaluation。
- 支持多个 worker/server pairs 并行 evaluation。
- 增加 conditional success、failure position、per-atomic-task success、action stats 和 near-miss rate。

效果：

- stable checkpoints 可以进行 n1000 formal evaluation。
- candidate branches 可以用 n300 进行快速排序。
- 新增 diagnostics 说明，主要提升来自 first-step success 和 hard-task handling，而不是 action smoothness alone。

### 8.3 State / Proprio Input

选择方向：

- 启用 8-D CALVIN proprio/state：

```text
state.x, state.y, state.z, state.roll, state.pitch, state.yaw, state.pad, state.gripper
```

- 通过 `CALVIN_SEND_STATE=1` 在 evaluation 中发送 state。
- 训练 state-aware branch，并结合 connector changes。

观察结果：

- state-aware smoke test 和小规模 evaluation 通过。
- metadata 正确显示 `model_state_dim=8`。
- 但 8k state+connector n1000 结果偏弱，`avg_seq_len=1.086`，`SR@5=4.3%`。

结论：

- state path 已接通，但当前结果不能证明 proprio 被有效使用。
- 需要 state zero/shuffle eval 和 gradient norm 检查。

### 8.4 Connector / Interface Training

选择方向：

- 不再 freeze 整个 `qwen_vl_interface`。
- 保持 Qwen backbone frozen。
- 训练 connector/interface 和 action head。

观察结果：

- connector + state 8k 单独没有明显提升。
- 后续 LoRA 和 augmentation 分支说明 representation adaptation 有价值，但 connector-only 不够。

结论：

- connector 应保持 trainable。
- 更适合与 hard-task augmentation 或 LoRA 组合。

### 8.5 Hard-Task Balanced Sampling

选择方向：

对以下 hard tasks 进行 oversampling：

- `open_drawer`
- `close_drawer`
- `move_slider_left`
- `turn_off_led`
- `turn_off_lightbulb`
- `turn_on_lightbulb`
- `push_*_right`

效果：

```text
base8k n300:
  avg_seq_len = 1.05
  SR@1 = 54.0%
  SR@5 = 3.7%

hardv2 aug n300:
  avg_seq_len = 1.847
  SR@1 = 72.0%
  SR@5 = 12.0%
```

结论：

- hard-task balance 是当前最高价值方向。
- 它尤其改善 drawer、LED/light 和 right-push tasks。
- `move_slider_left` 仍然困难，需要针对性处理。

### 8.6 Language Paraphrase and Task-Aware Image Augmentation

选择方向：

- 加入 canonical task mapping。
- 对 hard tasks 增加 controlled language paraphrases。
- 使用 task-aware light visual augmentation，而非强 random crop / color jitter。
- 保持 canonical task label 不变。

结论：

- hardv2 aug 的成功与该路线有关，但它与 hard-task sampling 耦合，不能单独归因。
- 未来任何 geometry-affecting augmentation 都应先生成 preview sheets 人工检查。

### 8.7 Left/Right Mirror Augmentation

选择方向：

- mirror primary 和 wrist images。
- 在 language 中交换 `left` / `right`。
- 对 action axes 做 sign transforms。
- 只在 transform 定义明确的 left/right tasks 上使用。

结果：

```text
mirror hardv2 n300:
  avg_seq_len = 1.753
  SR@1 = 72.7%
  SR@5 = 9.7%
```

结论：

- mirror 明显优于 base8k，但略弱于 non-mirror hardv2。
- 可能存在 wrist camera / action sign convention / view consistency 问题。
- 应作为独立 diagnostic branch，而非默认设置。

### 8.8 LoRA Exploration

选择方向：

- 在 Qwen 最后 4 层 language layers 上加入 LoRA。
- target modules: `q_proj,k_proj,v_proj,o_proj`。
- 使用 `rank=8`、`alpha=16`。
- 保持 base Qwen backbone 不被大规模破坏。

结果：

```text
LoRA2000 n300:
  avg_seq_len = 1.63
  SR@1 = 64.0%
  SR@5 = 9.7%
```

结论：

- LoRA 明显有效。
- LoRA alone 不如 hardv2 aug。
- 更推荐 LoRA + hardv2 或 LoRA + MoE，而不是单独依赖 LoRA。

### 8.9 MoE and Adaptive Action Head

选择方向：

- 评估和继续 team MoE branch。
- 探索 action-head architecture 是否能比单一 GR00T DiT 更好处理 hard tasks。

结果：

```text
GTY MoE95k n100:
  avg_seq_len = 1.91
  SR@1 = 76.0%
  SR@5 = 12.0%

WMH MoE-Adaptive n300:
  avg_seq_len = 1.397
  SR@1 = 67.7%
  SR@5 = 4.7%
```

结论：

- MoE architecture 有信号，但 adaptive branch 不稳定且存在 task bias。
- MoE-Adaptive 提升 lightbulb tasks，但严重伤害 drawer。
- GTY MoE95k 有潜力，但需要 n300/n1000 公平对比。

### 8.10 Current MoE95k + Fresh LoRA + Augmented ABC Training

当前训练两个分支：

```text
no mirror:
abc_moe95k_lora_aug_3h_bs96_0520_045012

mirror:
abc_moe95k_lora_mirror_3h_bs96_0520_045300
```

训练设置：

```text
NUM_PROCESSES = 8
BATCH_SIZE = 96
DATALOADER_NUM_WORKERS = 12
SAVE_INTERVAL = 1000
```

当前状态：

- 两个分支训练正常。
- 已经至少产生 `steps_3000_pytorch_model.pt`。
- 尚未进行 D evaluation。

必须做的 evaluation：

- no mirror latest checkpoint，D n300；
- mirror latest checkpoint，D n300；
- 如果某个分支超过 hardv2 aug，则继续 D n1000。

---

## 9. Submission Summary

本次路线选择是由 failure pattern 驱动的。baseline 主要失败在少数 hard atomic tasks 上，并且大多数失败是在 evaluation horizon 处 timeout，而不是立即崩溃。这说明主要问题是 grounding、contact 和 affordance execution，而不是简单 action-noise problem。

当前最可靠的 verified WMH route 是：

**hard-task balanced ABC training + controlled language paraphrase + task-aware image augmentation**。

关键结果如下：

```text
base8k n300:
  avg_seq_len = 1.05
  SR@5 = 3.7%

hardv2 aug n300:
  avg_seq_len = 1.85
  SR@5 = 12.0%
```

LoRA 也有效，但不是最佳 standalone route：

```text
LoRA2000 n300:
  avg_seq_len = 1.63
  SR@5 = 9.7%
```

MoE action head 有潜力，尤其是 GTY MoE95k，但必须用 per-task metrics 判断，因为 MoE-Adaptive 虽然显著提升 light tasks，却严重退化 drawer tasks。

当前 highest-upside candidate 是：

**GTY MoE95k + WMH hard-task augmented ABC data + fresh Qwen LoRA**，并分别测试 mirror / non-mirror 两个版本。

该路线只有在 D n300/n1000 验证其超过 hardv2 aug，且没有引入新的 task regression 后，才能升级为最终主路线。

