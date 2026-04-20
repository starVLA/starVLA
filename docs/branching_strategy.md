# StarVLA Branching & Release Strategy

This document describes how branches, releases, and version tags are managed in the StarVLA repository.

## Branch Model

StarVLA follows a **two-branch model** inspired by GitHub Flow:

| Branch | Purpose | Stability |
|--------|---------|-----------|
| `starVLA` | Stable release branch. Contains verified, production-ready code. | ✅ Stable |
| `starVLA_dev` | Active development branch. New features and improvements land here first. | ⚠️ May be temporarily unstable |

```
feature/xxx ──► starVLA_dev ──► starVLA (stable release)
   fix/xxx ──┘                      │
                                    ▼
                               tag: vX.Y.Z
```

### Why Two Branches (Not GitFlow)?

- **Simple**: contributors only need to target one branch (`starVLA_dev`).
- **Fast iteration**: features reach developers quickly via `starVLA_dev`.
- **Stable baseline**: users who need reproducible results can always use `starVLA`.

We deliberately avoid the complexity of GitFlow (`develop`, `release/*`, `staging`, etc.) — it adds overhead that doesn't match our release cadence.

## Merge Flow

### 1. Feature / Fix Development

All contributions start as **feature branches** forked from `starVLA_dev`:

```bash
git checkout starVLA_dev
git pull origin starVLA_dev
git checkout -b feat/my-new-feature
```

### 2. Pull Request → `starVLA_dev`

- Open a PR targeting `starVLA_dev`.
- Pass `make check` (Black + Ruff).
- Get at least **one maintainer approval**.
- Merge via **Squash Merge** to keep the commit history clean.

<!-- ### 3. Stable Release → `starVLA`

When `starVLA_dev` reaches a stable milestone:

1. A maintainer opens a PR from `starVLA_dev` → `starVLA`.
2. Core team reviews the accumulated changes.
3. After merge, a **version tag** is created on `starVLA`.

```bash
git checkout starVLA
git merge starVLA_dev
git tag -a v0.3.0 -m "Release v0.3.0: LIBERO support, QwenGR00T framework"
git push origin starVLA --tags
``` -->

### 4. Hotfix

For critical bugs on the stable branch:

```bash
git checkout starVLA
git checkout -b hotfix/fix-critical-bug
# ... fix and test ...
# PR → starVLA (direct)
# Then cherry-pick or merge back into starVLA_dev
git checkout starVLA_dev
git cherry-pick <hotfix-commit>
```

## Branch Naming Convention

| Prefix | Use Case | Example |
|--------|----------|---------|
| `feat/` | New feature or capability | `feat/cosmos-world-model` |
| `fix/` | Bug fix | `fix/oom-in-gr00t-training` |
| `docs/` | Documentation only | `docs/add-libero-tutorial` |
| `refactor/` | Code restructuring (no behavior change) | `refactor/dataloader-registry` |
| `exp/` | Experimental / research branch | `exp/diffusion-policy-head` |
| `hotfix/` | Urgent fix for stable branch | `hotfix/checkpoint-loading-crash` |

**Rules:**
- Use lowercase with hyphens: `feat/my-feature` (not `feat/MyFeature`).
- Keep names short but descriptive.
- Include issue number when applicable: `fix/192-action-stats-cache`.


## Community PR Guidelines

> Core principle: **Minimize changes, stay focused, and make it verifiable.**

### 1. File Isolation

StarVLA's architecture is designed for **file-level isolation** — new frameworks / benchmarks / datasets should be self-contained in their own directories, avoiding modifications to shared modules.

| Scenario | Recommended | Not Allowed |
|----------|-------------|-------------|
| New framework | Create a new module under `starVLA/model/framework/` | Directly modify core logic of existing frameworks |
| New benchmark | Create a new directory under `examples/<benchmark>/` | Scatter configs across multiple existing directories |
| New dataset | Register in `examples/<benchmark>/train_files/data_registry/` | Modify the shared dataloader interface (unless well justified) |
| Bug fix | Precisely modify the affected file, include a unit test | Refactor unrelated code along the way |

If you do need to modify shared modules (dataloader, config, trainer), please **clearly explain the reason and scope of impact** in the PR description.

### 2. Benchmark / Framework Contributions Must Include Validation

PRs of the following types **must** provide validation materials:

- Adding or modifying a training framework
- Adding or modifying a benchmark integration
- Modifying core dataloader / training loop logic

**Required materials:**

| Material | Requirement |
|----------|-------------|
| **Benchmark results** | Quantitative evaluation results on at least one benchmark (e.g., LIBERO success rate, SimplerEnv score), presented as a table or screenshot in the PR description |
| **Checkpoint** | Trained checkpoint uploaded to the contributor's own Hugging Face account and set to **public**, with a link provided in the PR |
| **Training config** | Complete training config YAML, placed under the corresponding `examples/` directory |
| **Reproduction instructions** | Brief explanation of how to reproduce evaluation results using the provided config + checkpoint |

Example format (in the PR description):

```markdown
## Validation

| Benchmark | Metric | Result |
|-----------|--------|--------|
| LIBERO-Goal | Success Rate (avg 3 seeds) | 78.5% |

- Checkpoint: https://huggingface.co/<your-username>/starvla-xxx
- Config: `examples/LIBERO/train_files/xxx.yaml`
- Reproduce: `bash examples/LIBERO/eval.sh --ckpt <hf-path>`
```

### 3. Testing

- **New features**: Must include at least one test script or test case, placed in `tmp/` or the corresponding `examples/` directory for the PR.
- **Bug fixes**: Describe reproduction steps, ideally with a minimal test that triggers the bug.
- **Refactors**: Ensure existing `make check` passes; if behavior changes are involved, add supplementary tests.

### 4. PR Scope Control

| ✅ Recommended | ❌ Avoid |
|----------------|----------|
| One PR does one thing | Mix new framework + bug fix + documentation update in one PR |
| Diff < 500 lines | Overly large PRs (>1000 lines), unless adding an independent module |
| Only modify relevant files | Incidentally format / refactor unrelated code |
| Open an Issue for discussion first | Submit large-change PRs directly |


> For the detailed PR submission workflow, commit message conventions, and review process, see [docs/PR_readme.md](PR_readme.md).

## Summary

```
                  hotfix/xxx
                     │
                     ▼
  ┌──────────────────────────────────────┐
  │           starVLA (stable)           │  ◄── tags: v0.1.0, v0.2.0, ...
  └──────────────┬───────────────────────┘
                 │ merge (milestone)
                 │
  ┌──────────────▼───────────────────────┐
  │         starVLA_dev (active)         │  ◄── PRs land here
  └──┬───────────┬───────────┬───────────┘
     │           │           │
  feat/a      fix/b      docs/c
```

**One sentence**: contributors branch from `starVLA_dev`, submit PRs back to `starVLA_dev`, and maintainers periodically promote stable snapshots to `starVLA` with version tags.
