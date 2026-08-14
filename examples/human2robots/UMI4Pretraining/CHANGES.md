# StarVLA UMI Data Support: Change Guide

This document describes the `examples/human2robots/UMI4Pretraining/` module and
the changes on `feature/umi-data-pipeline` relative to upstream StarVLA commit
`0ed0aad2c83f587714f6167ef60cf7218b786590`.

The implementation was developed in three commits:

- `aa64732`: add UMI acquisition, registry, training configuration, and the
  UMI-specific dataloader;
- `6166cd1`: connect validity masks to the training loss, reject mixed action
  semantics, and harden acquisition and verification;
- `ac30f23`: move the module to `examples/human2robots/UMI4Pretraining/` and
  add integration documentation.

## 1. Goals and architecture

The update addresses four practical problems:

1. public UMI datasets use many repositories, layouts, and download methods;
2. converted action tensors can have matching shapes while representing
   incompatible control semantics;
3. the generic LeRobot loader does not enforce UMI-specific data boundaries;
4. a broken optional DeepSpeed installation can prevent single-GPU training
   from starting.

The data path is:

```text
Public Hugging Face / direct sources
                 │
                 ▼
Resumable download and source verification
                 │
                 ▼
External converters → LeRobot v2.1 policy views
                 │
                 ▼
StarVLA external data_registry
                 │
                 ▼
Generic LeRobot decoder → UMI safety adapter
                 │
                 ▼
QwenOFT with mask-aware action loss
```

The 27 UMI families are not hard-coded into StarVLA's core registry. They are
loaded through the recursive external `data_registry` discovery mechanism.
Updating upstream StarVLA therefore does not require repeatedly patching core
`mixtures.py`, `data_config.py`, or `embodiment_tags.py` files.

## 2. Data acquisition changes

The acquisition entry points are:

```text
examples/human2robots/UMI4Pretraining/tools/download_umi.sh
examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py
examples/human2robots/UMI4Pretraining/tools/plans/*.lock.json
```

The lock files currently describe 30 physical sources representing 27
independent data families. The source count is larger because UMI-3D uses
three task repositories and LivUMI uses separate Grip and Ego repositories.

The downloader supports:

- Hugging Face snapshots, regular HTTP sources, and Google Drive sources;
- Hugging Face resume behavior and `.part` resume for direct downloads;
- parallel source downloads and per-snapshot Hugging Face workers;
- `HF_TOKEN` or an existing `hf auth login` session;
- explicit reporting for gated repositories whose agreements are not accepted;
- a configurable free-space reserve;
- atomic updates to `state/download_status.json`;
- file, byte-size, and optional per-member ZIP verification;
- the established `samples_400` layout, avoiding a second copy of existing data.

The global completion marker is created only when the complete 30-source plan
passes verification:

```text
.all_available_400_sources_downloaded
```

Verifying a `--families` subset never writes the global marker. After a
snapshot returns, all required paths are checked again. A complete `.part`
file is promoted without another network request, while an oversized partial
file is isolated instead of repeatedly appended.

### Usage

```bash
cd /project/vonneumann1/UMI_data/starVLA-latest-UMI
export UMI_DATA_ROOT=/project/vonneumann1/UMI_data/samples_400

# Inspect sources, free space, and destinations without downloading.
bash examples/human2robots/UMI4Pretraining/tools/download_umi.sh --dry-run

# Download and verify every available source.
bash examples/human2robots/UMI4Pretraining/tools/download_umi.sh

# Download selected families only.
python3 examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py download \
  --families VISTA-UMI-5K,MV-UMI,UMI-3D

# Run an offline deep verification pass.
python3 examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py verify --deep
```

`download_umi.sh` reuses an existing `hf` CLI or Python installation. If
neither is available, it bootstraps a small virtual environment. Credentials
are never stored in the repository or lock files.

## 3. Conversion boundary

The repository entry point currently covers source acquisition and source-file
verification. The validated operational converters remain at:

```text
/project/vonneumann1/UMI_data/training_ready/scripts/
```

Their converted StarVLA views are stored at:

```text
/project/vonneumann1/UMI_data/training_ready/starvla_data
```

The historical converters were not copied into this module because they still
contain cluster paths, recovery state, and family-specific operational logic.
The current implementation does not describe a downloaded family as converted.
A portable download → convert → audit command requires parameterizing those
converters first.

The 400-case target means "up to 400 genuine usable cases." A public source
with fewer than 400 cases retains its real count. Trajectories are never
duplicated and fake cases are never created through padding.

## 4. Registry and action semantics

The external registry is defined at:

```text
examples/human2robots/UMI4Pretraining/train_files/data_registry/data_config.py
```

It contains:

- 13 UMI robot/data configurations;
- 24 policy mixtures;
- video, state, action, and language mappings for each configuration;
- an explicit `action_semantics` value for every UMI robot configuration.

The current semantic buckets include:

- absolute end-effector control;
- delta end-effector control;
- dual-arm absolute and delta end-effector control;
- joint control;
- dexterous-hand control;
- source-native action spaces that are not yet canonicalized.

Actions with equal dimensionality are not necessarily interchangeable behavior
cloning targets. The UMI loader checks the `action_semantics` values in a
mixture and rejects multiple semantics by default. An experiment must use
`allow_mixed_action_semantics: true` to opt in explicitly, and should do so
only when the model head and action definition are designed for that mixture.

LEAP, SenseXperience-UMI, UMI-VQA, LivUMI, ToucHD-Mani, and UMI-Benchmark
do not provide reliable policy actions. They remain observation/VLM/benchmark
data and must not be registered as behavior-cloning mixtures.

## 5. UMI-specific dataloader

The safety adapter is implemented in:

```text
starVLA/dataloader/umi_datasets.py
```

It is not a second video decoder. It wraps the generic
`LeRobotMixtureDataset`, preserving StarVLA's parquet, video, statistics, and
transform logic while enforcing UMI-specific sample contracts.

The adapter provides:

- finite-value checks for action and state arrays;
- strict action horizon, action dimension, and state dimension validation;
- required image-view validation and optional view limiting;
- whitespace-normalized language with an explicit fallback instruction;
- optional action-range and static-chunk checks;
- `action_mask`, `state_mask`, and `image_mask` generation;
- deterministic bad-sample recovery based on index, epoch, and seed;
- the `List[dict]` batch structure expected by StarVLA frameworks.

Action horizon, action dimension, and state dimension are inferred from
`framework.action_model`, so they do not need to be repeated under the dataset
configuration. If both sections specify a value and the values disagree, the
loader fails at startup.

### Enabling the adapter

```yaml
datasets:
  vla_data:
    dataset_py: umi_datasets
    strict_dimensions: true
    max_views: 2
    retry_bad_samples: 20
```

See the complete override fragment at:

```text
examples/human2robots/UMI4Pretraining/train_files/umi_loader_overrides.yaml
```

Normal training should keep `strict_dimensions: true`. Relaxed mode is intended
for diagnosis or experiments with a deliberately heterogeneous action head; it
must not be used as a shortcut for mixing incompatible robot action spaces.

## 6. QwenOFT loss changes

The UMI adapter can tensor-align short action chunks, but padded cells must not
produce gradients. Previously, QwenOFT averaged L1 loss over the complete
action tensor even when the loader supplied a validity mask.

The loss is now:

```text
loss = sum(abs(prediction - target) * action_mask) / sum(action_mask)
```

Compatibility rules are:

- no `action_mask` in the batch: preserve the original full-tensor mean L1;
- masks on every example: supervise valid action cells only;
- masks on only part of a batch: fail with a clear error;
- empty or shape-incompatible masks: fail with a clear error.

This closes the previous correctness gap where the dataloader emitted masks
but the model ignored them.

## 7. Single-GPU and DeepSpeed compatibility

Upstream `train_starvla.py` creates `DeepSpeedPlugin` unconditionally. A broken
or CUDA-incompatible optional DeepSpeed installation can therefore prevent a
single-GPU job from starting.

The compatibility path is:

```bash
STARVLA_DISABLE_DEEPSPEED=1 accelerate launch \
  --config_file examples/human2robots/UMI4Pretraining/train_files/accelerate_single_gpu.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml examples/human2robots/UMI4Pretraining/train_files/starvla_dexwild_smoke.yaml
```

When `STARVLA_DISABLE_DEEPSPEED` is not set, the upstream DeepSpeed behavior is
unchanged.

## 8. Validation completed

The following checks passed against the latest remote StarVLA checkout and
real converted data:

- external registry discovery: 24 UMI mixtures and 13 robot configurations;
- DexWild inventory: 346 trajectories and 123,485 steps;
- real batch action: `(8, 23)`, with 184/184 valid action cells;
- real batch state: `(1, 23)`, with 23/23 valid state cells;
- two image views, language, and the `new_embodiment` tag;
- action/state shape inference from the model configuration;
- numerical tests for mask-aware QwenOFT L1 loss;
- eight acquisition-pipeline regression tests;
- an earlier 20/20-step DexWild QwenOFT GPU smoke, including checkpoint and
  final-model saves.

The hardening pass did not submit another GPU job because the login node has no
GPU and a redundant two-step job would produce roughly 9 GB of checkpoint
artifacts. The real-batch checks, numerical loss tests, and prior 20-step smoke
provide the current regression coverage.

## 9. Known limitations and next steps

Current limitations are:

1. the complete conversion pipeline is not yet parameterized and included in
   the repository;
2. `max_abs_action` should be configured after normalization for each action
   semantic bucket, not as one global threshold;
3. `image_mask` is generated, but mask consumption must still be verified for
   every model framework;
4. observation/VLM-only sources require a separate loader and objective;
5. cross-semantic joint training requires an embodiment-aware action head or a
   canonical action representation, not dimensional padding;
6. the feature branch is published to `TruemanV5/starVLA`, but still requires a
   pull request and upstream review before entering `starVLA/starVLA`.

Recommended next steps are:

1. open a pull request from `feature/umi-data-pipeline` to upstream
   `starVLA_dev`;
2. expose the converters behind a portable `convert` subcommand and remove
   cluster-specific paths;
3. run small training/evaluation sweeps separately for absolute EEF, delta EEF,
   joint, and dexterous-hand buckets;
4. design canonical action tokenization or multi-head routing before attempting
   cross-embodiment joint pretraining.
