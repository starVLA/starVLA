# Pull Request Guidelines

Thank you for contributing to StarVLA! To ensure the more stable development of StarVLA, we are once again attempting to establish PR guidelines. This document describes how to submit a high-quality PR that can be reviewed and merged efficiently.


> **TL;DR**: Open an Issue first → branch from `starVLA_dev` → submit PR with clear description → address review feedback.

## Before You Start

1. **Check existing work** — Search open PRs and Issues to make sure your idea isn't already in progress.

2. **Read the branching strategy** — See [docs/branching_strategy.md](branching_strategy.md) for how branches are organized.

## PR Lifecycle

```
1. Issue / Discussion       Align scope with maintainers
        │
2. Create Branch            git checkout -b feat/xxx starVLA_dev
        │
3. Develop & Test           Write code, run make check
        │
4. Open PR                  Target: starVLA_dev
        │
5. Code Review              Address reviewer feedback
        │
6. Merge (Squash)           Maintainer merges after approval
```

## Step-by-Step

### 1. Fork & Clone (External Contributors)

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/<your-username>/starVLA.git
cd starVLA
git remote add upstream https://github.com/starVLA/starVLA.git
```

### 2. Create a Feature Branch

Always branch from the latest `starVLA_dev`:

```bash
git fetch upstream
git checkout -b feat/my-feature upstream/starVLA_dev
```

See [branching_strategy.md](branching_strategy.md) for naming conventions (`feat/`, `fix/`, `docs/`, etc.).

### 3. Write Code

- Follow existing code style (Black + Ruff).
- Keep changes focused — one PR per logical change.
- Add or update docstrings where appropriate.
- If you add a new training framework or dataset, include an example config in `examples/`.

### 4. Pre-Submit Checklist

> **Note on `make check`**: the repository currently carries a historical
> Black / Ruff backlog, so running `make check` against the **whole repo** is
> expected to fail. Until that backlog is cleaned up, **only check the files
> your PR actually touches**. Mixing in formatting fixes for unrelated files
> violates the "No unrelated changes" rule.

Recommended local check, scoped to your PR:

```bash
# Files modified or added in this PR (compared to starVLA_dev)
FILES=$(git diff --name-only --diff-filter=ACMR origin/starVLA_dev | grep -E '\.py$')

# Format + lint just those files
black $FILES
python -m ruff check --fix $FILES

# Verify
black --check $FILES
python -m ruff check $FILES
```

The full-repo entry points are still useful as a reference:

```bash
make check        # Verify formatting (Black) and lint (Ruff) on the whole repo
make autoformat   # Auto-fix formatting issues across the whole repo (do NOT commit unrelated reformat noise)
```

Self-review checklist:

- [ ] `black --check` and `ruff check` pass **on the files this PR touches**
- [ ] No unrelated changes (debug prints, unrelated refactors, drive-by reformats)
- [ ] New features have example configs or documentation
- [ ] No secrets, API keys, or large binary files committed
- [ ] Config YAML changes are backward-compatible (or clearly noted as breaking)

### 5. Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Build process, CI, or tooling changes |

**Examples:**

```
feat(dataloader): add streaming statistics with Welford's algorithm
fix(training): remove invalid resolve kwarg in save_full_config (#192)
docs(examples): add LIBERO 4-in-1 training tutorial
refactor(model): move QwenOFT to framework/VLM4A/
```

**Rules:**
- Use English for commit messages.
- Keep the subject line under 72 characters.
- Reference related Issues: `Closes #42`, `Fixes #108`.

### 6. Open a Pull Request

Target branch: **`starVLA_dev`** (not `starVLA`).

Use the following template for your PR description:

---

#### PR Description Template

```markdown
## Motivation

<!-- Why is this change needed? Link to the related Issue. -->
Closes #<issue-number>

## Changes

<!-- What does this PR do? List the key changes. -->
- Added ...
- Fixed ...
- Refactored ...

## Testing

<!-- How was this tested? -->
- [ ] Ran `black --check` / `ruff check` on the files this PR touches — passes
- [ ] Tested on [dataset/framework/environment]: ...
- [ ] Training runs for N steps without error

## Breaking Changes

<!-- Does this PR break backward compatibility? If yes, describe. -->
None / Yes: ...

## Screenshots / Logs (optional)

<!-- Attach training curves, evaluation results, or relevant logs if applicable. -->
```

---

### 7. Code Review

- A maintainer will be assigned to review your PR.
- Address all review comments and push follow-up commits.
- Resolve conversations after making requested changes.
- If the PR is stale for >14 days without response, it may be closed.

### 8. Merge

- PRs are merged via **Squash Merge** to keep the main branch history clean.
- The maintainer will write a clean squash commit message based on your PR title.
- After merge, your feature branch can be deleted.

## What Makes a Good PR

| ✅ Do | ❌ Don't |
|-------|----------|
| One logical change per PR | Mix unrelated changes in one PR |
| Clear, descriptive title | Vague titles like "Update code" |
| Reference the related Issue | Submit without prior discussion |
| Include before/after comparison | Leave reviewers guessing about impact |
| Keep diff small (<500 lines ideally) | Submit 3000-line PRs without context |
| Run Black + Ruff on your changed files before pushing | Push code that fails lint/format on the lines you touched |

## Special Cases

### Adding a New Training Framework

1. Create the framework module under `starVLA/model/framework/`.
2. Register it in the framework config system.
3. Add a default config dataclass (e.g., `QwenOFTDefaultConfig`).
4. Add an example config YAML under `examples/<benchmark>/train_files/`.
5. Include a brief section in `docs/model_zoo.md`.
6. **Provide at least one benchmark result + public HF checkpoint** (see [branching_strategy.md § Community PR Guidelines](branching_strategy.md#community-pr-guidelines)).

### Adding a New Dataset / Benchmark

1. Create data configs under `examples/<benchmark>/train_files/data_registry/`.
2. Ensure the dataset is in LeRobot format.
3. Add an example training script under `examples/<benchmark>/`.
4. Document any special setup steps.
5. **Provide quantitative evaluation results + public HF checkpoint** (see [branching_strategy.md § Community PR Guidelines](branching_strategy.md#community-pr-guidelines)).

### Documentation-Only PRs

- Target `starVLA_dev` as usual.
- No need for extensive testing, but verify links and formatting.
- These are always welcome!

## Need Help?

- **Office Hours**: Every Friday afternoon — fill in the [Cooperation Form](https://forms.gle/R4VvgiVveULibTCCA).
- **Discussions**: Use [GitHub Issues](https://github.com/starVLA/starVLA/issues) for technical questions.
- **Quick questions**: Tag a maintainer in your PR comments.

---

*Last updated: April 2026*
