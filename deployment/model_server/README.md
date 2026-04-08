
# start policy server


```bash
your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

python -m deployment.model_server.server_policy \
  --ckpt_path ${your_ckpt} \
  --port 10093 \
  --device auto \
  --use_bf16
```

Use `--device cpu` for a functional smoke test on machines without CUDA. For meaningful latency numbers, prefer a real GPU-backed run.


# connect to policy server for debug

```bash
python -m deployment.model_server.tools.debug_server_policy \
  --image assets/starVLA_LOGO.png \
  --instruction "pick up the red block"

# plus server_policy.py into your vla controler by ref to tools/debug_server_policy.py
```


# benchmark cache reuse

Use the same image/instruction repeatedly to compare cold requests against same-session reuse:

```bash
python -m deployment.model_server.tools.benchmark_policy_server \
  --host 127.0.0.1 \
  --port 10093 \
  --image assets/starVLA_LOGO.png \
  --instruction "pick up the red block" \
  --mode compare \
  --runs 10 \
  --warmup 1 \
  --output-json benchmark-cache-report.json
```

Replace `assets/starVLA_LOGO.png` with a real observation frame if you want more meaningful latency numbers.

What the benchmark reports:

- end-to-end request latency
- server-side `predict_action` latency
- throughput in requests/sec
- cache hit / miss counts from `return_cache_info`
- cache footprint metadata (`cache_entries`, `cache_bytes`) so you can see the memory cost of reuse
- server metadata in the JSON report, including the resolved device and checkpoint name
- in `--mode compare`, an extra comparison block showing latency reduction, throughput uplift, and hit-rate change for same-session reuse vs forced cold requests

When your controller switches tasks or instructions, call `client.reset_cache()` before the next request to clear stale server-side VLM features for that session.
Set `--warmup 0` if you want the first measured request in reuse mode to include the initial cache miss instead of steady-state hits only.
