# vla-eval

Optional LIBERO and RoboTwin 2.0 evaluation through `vla-eval`. The adapter reuses
`PolicyServerWrapper` for checkpoint loading and action unnormalization.

In an existing StarVLA uv environment with the model dependencies installed,
add the optional integration from the repository root:

```bash
uv pip install -e ".[evaluation]"
```

Start the model server on a GPU, then run the benchmark in another terminal:

```bash
python -m deployment.vla_eval.model_server \
  --config examples/vla_eval/model_servers/libero_qwen3_oft.yaml
vla-eval run \
  --config examples/vla_eval/benchmarks/libero_smoke.yaml
```

For RoboTwin, substitute `robotwin_qwen3_oft.yaml` and
`robotwin_smoke.yaml` in the commands above.

The model configs accept a Hugging Face repository ID, local run directory, or
weight file via `--args.checkpoint=...`. A run needs `config.yaml`,
`dataset_statistics.json`, and `checkpoints/*.pt` or `*.safetensors`. Directories
select the highest numeric step; use a specific weight path to pin a checkpoint.
Set `--args.unnorm_key=...` if checkpoint statistics have no unambiguous default.

| Profile | Camera order | Action conversion |
|---|---|---|
| LIBERO | `agentview`, `wrist` | gripper: `1 - 2 * (x > 0.5)` |
| RoboTwin | `head_camera`, `left_camera`, `right_camera` | left arm/gripper, then right arm/gripper |

This adapter currently supports policies that do not require state inputs.
Images are resized to 224 × 224 using PIL
bilinear for LIBERO and OpenCV area for RoboTwin, matching the native clients.
Actions use the checkpoint's statistics and chunk horizon; episode boundaries
clear the harness buffer.

The configs run smoke evaluations, not full-suite score reproductions. RoboTwin
skips expert filtering and uses the harness's generic task instruction. Results
are saved to the configured `output_dir`.

Run the adapter contract tests with:

```bash
uv pip install pytest
python -m pytest tests/test_vla_eval_adapter.py
```
