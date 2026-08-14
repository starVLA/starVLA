# UMI4Pretraining: human demonstrations for StarVLA

See `CHANGES.md` for the complete change guide, design boundaries, validation
record, and known limitations.

This integration uses StarVLA's external `data_registry` discovery. It does not
modify the core `mixtures.py`, `data_config.py`, or `embodiment_tags.py` files.

The converted LeRobot v2.1 dataset root is expected to contain the dataset
directories referenced by the UMI mixtures. In the validated installation this
is:

```text
/project/vonneumann1/UMI_data/training_ready/starvla_data
```

Policy datasets are registered here. Observation/VLM-only UMI manifests have
explicit null actions and must be trained through the VLM dataloader, never the
behavior-cloning mixture.

Action semantics and safe mixture buckets are documented in
`STARVLA_UMI_USAGE.yaml` in the converted-data root.

## Compatibility status

Validated against upstream commit `0ed0aad2c83f587714f6167ef60cf7218b786590`:

- external registry discovery: 24 UMI mixtures and 13 robot data configs;
- real DexWild loader: 346 trajectories / 123,485 steps;
- sample output: action `(8, 23)`, state `(1, 23)`, two images, language and
  `new_embodiment` tag;
- QwenOFT smoke: 20/20 optimizer steps, checkpoint and final model saved.

The policy registry and training configs can be added directly to current
StarVLA. A portable acquisition entry point is included under `tools/`; it does
not contain cluster paths, job IDs, credentials, or source-specific recovery
state.

## One-command 400-case acquisition

Select a data root. `download_umi.sh` reuses an existing `hf` CLI/Python package
or bootstraps a small virtual environment automatically:

```bash
export UMI_DATA_ROOT=/path/with/enough/free/space/UMI_data/samples_400
```

For gated Hugging Face datasets, accept the repository agreements in the
browser first. The downloader uses `HF_TOKEN` when set, otherwise it uses the
existing `hf auth login` session. Tokens are never saved in this repository.

Inspect the plan, then download and verify all currently available sources:

```bash
python3 examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py doctor
bash examples/human2robots/UMI4Pretraining/tools/download_umi.sh
```

This resolves 30 physical sources into 27 independent UMI families. UMI-3D
uses three task repositories and LivUMI uses Grip plus Ego, so source count is
larger than family count. Hugging Face snapshots and direct `.part` files are
resumable. Progress is atomically recorded at
`$UMI_DATA_ROOT/state/download_status.json`; a successful offline verification
creates `$UMI_DATA_ROOT/.all_available_400_sources_downloaded`.
The directory layout is backward-compatible with the existing `samples_400`
tree, so rerunning the command adopts partial snapshots instead of creating a
second copy.

Useful targeted commands:

```bash
# Show every destination without transferring data.
bash examples/human2robots/UMI4Pretraining/tools/download_umi.sh --dry-run

# Download one family (repeat --families or use a comma-separated list).
python3 examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py download \
  --families VISTA-UMI-5K,UMI-3D

# Offline size/file check, optionally testing every ZIP member.
python3 examples/human2robots/UMI4Pretraining/tools/umi_pipeline.py verify --deep
```

The lock files deliberately select complete archives or complete LeRobot
episodes/tasks instead of arbitrary byte ranges. Some public releases contain
fewer than 400 usable trajectories (notably the OpenEAI subset); the target is
therefore "up to 400, without fabricating or duplicating cases." Conversion
reports must retain such exceptions rather than silently padding them.

## Single-GPU smoke

Current upstream `train_starvla.py` constructs `DeepSpeedPlugin` unconditionally.
On a machine with an installed but unusable DeepSpeed/CUDA toolchain, set
`STARVLA_DISABLE_DEEPSPEED=1` with the small compatibility guard included in
this integration and use `accelerate_single_gpu.yaml`. The default behavior is
unchanged when the environment variable is absent.

Example:

```bash
STARVLA_DISABLE_DEEPSPEED=1 accelerate launch \
  --config_file examples/human2robots/UMI4Pretraining/train_files/accelerate_single_gpu.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml examples/human2robots/UMI4Pretraining/train_files/starvla_dexwild_smoke.yaml
```

## Optional UMI-specific dataloader

Set `datasets.vla_data.dataset_py: umi_datasets` and add the fields shown in
`train_files/umi_loader_overrides.yaml` to route the standard LeRobot decoder
through `starVLA/dataloader/umi_datasets.py`. The adapter adds:

- finite-value, action/state shape, image and optional action-range checks;
- deterministic bad-sample recovery without shared worker RNG state;
- normalized language and optional multi-view limiting;
- fixed horizon packing plus action/state/image validity masks;
- the same list-of-dictionaries batch contract consumed by StarVLA models.

Action horizon/dimension and state dimension are read from
`framework.action_model`; they need not be duplicated under `vla_data`. If
both sections specify a value, startup fails when they disagree instead of
silently producing a mismatched label tensor.

QwenOFT applies `action_mask` to its L1 loss when the UMI adapter supplies it,
so a short final chunk does not train against padded cells. Keep
`strict_dimensions: true` for the normal training path: it catches conversion
errors early and guarantees the configured horizon/dimensions. More
importantly, do not put absolute pose, delta EEF, joint-space and
dexterous-hand actions into one mixture merely because tensors can be padded
to the same width. The UMI loader rejects mixed semantics unless the config
explicitly opts in.
