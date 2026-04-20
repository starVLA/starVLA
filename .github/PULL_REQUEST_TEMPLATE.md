## Description
<!-- Briefly describe what this PR does -->


## Motivation
<!-- Why is this change needed? Please link the related Issue -->
Closes #

## Changes
<!-- List the key changes -->
-
-

## Testing
<!-- How was this verified? Check at least one -->
<!-- - [ ] `make check` passes  TODO: enable later -->
- [ ] Local training for N steps without errors
- [ ] Benchmark evaluation results (**required** for framework / benchmark changes, see table below)

| Benchmark | Metric | Result |
|-----------|--------|--------|
|           |        |        |

- **Checkpoint**: <!-- Required for framework/benchmark changes: public HF link -->
- **Config**: <!-- Config path under examples/ -->
- **Reproduce**: <!-- Reproduction command -->

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] New framework / benchmark integration
- [ ] Breaking change
- [ ] Documentation only
- [ ] Refactor (no behavior change)

## Checklist
<!-- - [ ] `make check` passes (Black + Ruff)  TODO: enable later -->
- [ ] No unrelated changes mixed in
- [ ] New features have example config in `examples/`
- [ ] No secrets, API keys, or large binaries committed
- [ ] Files stay isolated — no unnecessary modification to shared modules (`starVLA/dataloader/`, `starVLA/training/`, `starVLA/config/`)
- [ ] No modifications to shared files, or justified above
- [ ] If adding framework/benchmark: checkpoint uploaded to personal HF (public)

## Screenshots / Logs (optional)
<!-- Training curves, evaluation result screenshots, relevant logs -->
