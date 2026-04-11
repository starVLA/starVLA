# StarVLA Awesome List PR Guide

> **Goal**: Get StarVLA listed in major awesome lists to increase visibility.
>
> **Paper**: https://arxiv.org/abs/2604.05014
> **Code**: https://github.com/starVLA/starVLA
> **Project Page**: https://starvla.github.io/

---

## Status Summary

| # | Repository | ⭐ Stars | Status | Priority |
|---|-----------|---------|--------|----------|
| 1 | [Jiaaqiliu/Awesome-VLA-Robotics](https://github.com/Jiaaqiliu/Awesome-VLA-Robotics) | 504 | ❌ Not listed | 🔥 High |
| 2 | [MilkClouds/awesome-vla-study](https://github.com/MilkClouds/awesome-vla-study) | 203 | ❌ Not listed | 🔥 High |
| 3 | [keon/awesome-physical-ai](https://github.com/keon/awesome-physical-ai) | 188 | ⚠️ Outdated entry | 🔥 High (update) |
| 4 | [FutureTwT/awesome-world-models-for-vla-agents](https://github.com/FutureTwT/awesome-world-models-for-vla-agents) | 39 | ❌ Not listed | Medium |
| 5 | [whitbrunn/Awesome-RL-in-VLA](https://github.com/whitbrunn/Awesome-RL-in-VLA) | 4 | ❌ Not listed | Low |
| 6 | [Noietch/Awesome-Learning-for-Manipulation](https://github.com/Noietch/Awesome-Learning-for-Manipulation) | 3 | ❌ Not listed | Low |
| 7 | [HyperbolicCurve/Awesome-World-Action-Model](https://github.com/HyperbolicCurve/Awesome-World-Action-Model) | 16 | ✅ Listed | No action needed |

---

## PR 1: Jiaaqiliu/Awesome-VLA-Robotics

> **504 ⭐ — Highest priority**

**PR Title**: `Add StarVLA: A Lego-like Codebase for VLA Model Developing`

**Target file**: `README.md`

**Location**: Section `3.2.1 Manipulation > 2026` (add among the existing 2026 entries)

**Content to add** (matches the existing format exactly):

```markdown
- [2026] **StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing** [[paper](https://arxiv.org/abs/2604.05014)] [[project](https://starvla.github.io/)] [[code](https://github.com/starVLA/starVLA)]
```

**PR Description**:
```
Add StarVLA to the Manipulation section (2026).

StarVLA is an open-source, modular codebase for VLA model development.
It supports multiple VLA architectures (FAST, OFT, PI, GR00T),
diverse training recipes (SFT, co-training, cross-embodiment),
and broad benchmark integration (LIBERO, SimplerEnv, RoboCasa, RoboTwin, etc.).

- Paper: https://arxiv.org/abs/2604.05014
- Code: https://github.com/starVLA/starVLA
- Project: https://starvla.github.io/
```

---

## PR 2: MilkClouds/awesome-vla-study

> **203 ⭐**

**PR Title**: `Add StarVLA to See Also section`

**Target file**: `README.md`

**Location**: `See Also` section (after the existing entries like vla0-trl, vla-eval)

**Content to add** (matches the existing format):

```markdown
- 🔥 **[StarVLA](https://github.com/starVLA/starVLA)** — A Lego-like open-source codebase for VLA model development. Supports multiple VLA architectures (FAST, OFT, PI, GR00T), diverse training recipes, and broad benchmark integration (LIBERO, SimplerEnv, RoboCasa, RoboTwin, etc.). [[Paper](https://arxiv.org/abs/2604.05014)]
```

**PR Description**:
```
Add StarVLA to the See Also section.

StarVLA is a modular, Lego-like open-source codebase for developing
Vision-Language-Action models. It implements multiple VLA architectures
(StarVLA-FAST, StarVLA-OFT, StarVLA-PI, StarVLA-GR00T) with a unified
data interface, and supports various benchmarks (LIBERO, SimplerEnv,
RoboCasa, RoboTwin, BEHAVIOR, Calvin).

- Paper: https://arxiv.org/abs/2604.05014
- Code: https://github.com/starVLA/starVLA
```

---

## PR 3: keon/awesome-physical-ai

> **188 ⭐ — Update existing outdated entry**

**PR Title**: `Update StarVLA entry with arXiv paper link`

**Target file**: `README.md`

**Location**: End-to-End VLAs section (line ~161)

**Current content** (outdated — uses Overleaf link):
```markdown
- **StarVLA**: "StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing", *arXiv, 2025*. [[Report](https://www.overleaf.com/read/qqtwrnprctkf#d5bdce)] [[Code](https://github.com/starVLA/starVLA)]
```

**Replace with**:
```markdown
- **StarVLA**: "StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing", *arXiv, Apr 2026*. [[Paper](https://arxiv.org/abs/2604.05014)] [[Project](https://starvla.github.io/)] [[Code](https://github.com/starVLA/starVLA)]
```

**PR Description**:
```
Update StarVLA entry with the official arXiv paper link.

The previous entry linked to an Overleaf report. The paper has now been
published on arXiv (2604.05014) and a project page is available.

Changes:
- Updated year from 2025 to Apr 2026
- Replaced Overleaf Report link with arXiv Paper link
- Added Project page link
```

---

## PR 4: FutureTwT/awesome-world-models-for-vla-agents

> **39 ⭐ — Medium priority**

**PR Title**: `Add StarVLA with WM4A (World Model for Action) support`

**Target file**: `README.md`

**Location**: Section `🧱 Foundation Models` (StarVLA is a framework/platform, not a single paper — fits best here)

**Content to add** (matches the badge-based format used in this repo):

```markdown
* **StarVLA** - StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing. (2026) [![arXiv](https://img.shields.io/badge/arXiv-2604.05014-b31b1b.svg)](https://arxiv.org/abs/2604.05014) [![GitHub](https://img.shields.io/badge/GitHub-Code-1a73e8.svg?logo=github)](https://github.com/starVLA/starVLA)
```

**PR Description**:
```
Add StarVLA to the Foundation Models section.

StarVLA is an open-source, modular codebase for VLA model development
that supports WM4A (World Model for Action), which uses pretrained
video-generation DiT models (Cosmos-Predict2, Wan2.2) as backbones
for action prediction.

- Paper: https://arxiv.org/abs/2604.05014
- Code: https://github.com/starVLA/starVLA
- WM4A docs: https://github.com/starVLA/starVLA/blob/starVLA/docs/WM4A.md
```

---

## PR 5: whitbrunn/Awesome-RL-in-VLA

> **4 ⭐ — Low priority**

**PR Title**: `Add StarVLA to Toolkits & Projects section`

**Target file**: `README.md`

**Location**: `Toolkits & Projects` section

**Content to add** (as a new subsection or inline entry):

```markdown
### VLA Frameworks

| VLA Platform | Description | Code |
| ------------ | ----------- | ---- |
| [StarVLA](https://github.com/starVLA/starVLA) | Lego-like modular codebase for VLA development. Supports RL post-training via [RLinf integration](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html). | [![GitHub](https://img.shields.io/github/last-commit/starVLA/starVLA?label=last%20update)](https://github.com/starVLA/starVLA) |
```

**PR Description**:
```
Add StarVLA to the Toolkits & Projects section.

StarVLA is a modular VLA development codebase that now supports
RL post-training via RLinf integration, making it relevant to
this awesome list's focus on RL in VLA.

- Paper: https://arxiv.org/abs/2604.05014
- Code: https://github.com/starVLA/starVLA
- RL tutorial: https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html
```

---

## PR 6: Noietch/Awesome-Learning-for-Manipulation

> **3 ⭐ — Low priority**

**PR Title**: `Add StarVLA to VLA section (2026)`

**Target file**: `README.md`

**Location**: Section `🤖 VLA — Vision-Language-Action Models > 2026 (Preprints)` (there's a subsection for 2026 preprints)

**Content to add** (matches the table format):

```markdown
| [StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing](https://arxiv.org/abs/2604.05014) | — | 2026 | [Paper](https://arxiv.org/abs/2604.05014) \| [Code](https://github.com/starVLA/starVLA) | Modular VLA codebase supporting FAST, OFT, PI, GR00T architectures with SOTA benchmarks |
```

**PR Description**:
```
Add StarVLA to the VLA section (2026 Preprints).

StarVLA is an open-source, modular codebase for VLA model development
supporting multiple architectures and achieving SOTA on LIBERO,
SimplerEnv, RoboCasa, and more.

- Paper: https://arxiv.org/abs/2604.05014
- Code: https://github.com/starVLA/starVLA
```

---

## How to Submit PRs

For each target repository:

1. **Fork** the target repository on GitHub
2. **Clone** your fork locally
3. **Create a branch**: `git checkout -b add-starvla`
4. **Edit** `README.md` with the content specified above
5. **Commit**: `git commit -am "Add StarVLA"`
6. **Push**: `git push origin add-starvla`
7. **Open PR** on the original repository with the title and description provided above

**Recommended order** (by impact): PR 1 → PR 2 → PR 3 → PR 4 → PR 5 → PR 6
