# UMI integration report

## Directly portable

- LeRobot v2.x policy views and their `modality.json` metadata.
- UMI `DataConfig` classes via external `data_registry` discovery.
- Per-family and safe same-action-semantics mixtures.
- StarVLA YAML configs and single-/multi-GPU training entry points.
- Dataset/manifest integrity auditing.

## Kept separate from behavior cloning

LEAP, SenseXperience-UMI, UMI-VQA, LivUMI, ToucHD-Mani, and UMI-Benchmark
manifests have explicit null actions. They must use a VLM/observation or
benchmark loader, not the policy mixture registry.

## Requires parameterization before upstreaming

- Hugging Face credentials and gated-dataset acceptance.
- Absolute `/project/vonneumann1/...` paths.
- Slurm allocation/job IDs and GPU assignment.
- Dataset-specific resume logs and temporary recovery files.
- Source download selection and disk-watermark policies.

The recommended upstream layout is:

```text
examples/human2robots/UMI4Pretraining/
  train_files/
    data_registry/data_config.py
    starvla_*.yaml
    accelerate_single_gpu.yaml
  tools/
    download.py
    convert.py
    audit.py
  README.md
```

The validated operational conversion code remains in
`/project/vonneumann1/UMI_data/training_ready/scripts/`; it should be migrated
behind a parameterized CLI rather than copied verbatim.
