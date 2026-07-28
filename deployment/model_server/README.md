
# Policy Server

## Start the server

```bash
your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16
```

## GR00T-protocol server (ZMQ) — drop-in for Isaac-GR00T clients

For robot stacks already speaking the Isaac-GR00T N1.6 `PolicyServer` ZMQ
protocol (e.g. the GR00T-WBC-Bridge deploy script for the Unitree G1 whole-body
stack), serve the same checkpoint over that protocol instead of websockets:

```bash
python deployment/model_server/server_policy_gr00t_zmq.py \
    --ckpt_path ${your_ckpt} \
    --port 5555 \
    --use_bf16
```

The existing GR00T client keeps working unchanged (same msgpack
`__ndarray_class__` codec, same `ping` / `reset` / `get_action` endpoints).
The server:

- flattens the client's named state dict (`{"left_arm": (1, n, 7), ...}`)
  into the flat state vector, **in the checkpoint DataConfig's `state_keys`
  order**, taking the latest history frame per group;
- splits the unnormalized flat action chunk `(1, T, D)` back into named groups
  (`{"left_arm": (1, T, 7), ...}`) using the DataConfig's `action_keys` /
  `action_key_dims`.

So bridging a new embodiment needs no protocol code — only a registered
DataConfig whose state/action key names match what the client sends/consumes.
`get_modality_config` returns the full key/dim contract for handshake checks.
Components: `gr00t_obs_adapter.py` (conversion) +
`tools/zmq_policy_server.py` (transport). Requires `pyzmq`.

Caveats:
- Only the N1.6 "custom" codec is implemented (GR00T-WBC-Bridge default
  `--server_codec custom`); the N1.7 msgpack-numpy codec is not.
- State/video history stacking beyond "latest frame" is not supported —
  matches the single-frame `examples` schema used across starVLA eval clients.

## Connect to server for debug

```bash
python deployment/model_server/debug_server_policy.py

# use server_policy.py in your eval client by referencing debug_server_policy.py
```

---

## Architecture Update: Server-side Un-normalization (2025-05)

### What changed

Previously, eval clients (LIBERO, SimplerEnv, Robotwin, etc.) each implemented their
own hand-rolled un-normalization logic (`unnormalize_actions`, `get_action_stats`,
`read_mode_config`). This was fragile and led to subtle mismatches with training-time
transforms.

**Now the server owns un-normalization.** The new components are:

| File | Role |
|---|---|
| `policy_norm_processor.py` | Wraps the training-time `ComposedModalityTransform.unapply()`. Single source of truth for un-normalization. |
| `policy_wrapper.py` | Wraps `baseframework` + `PolicyNormProcessor`. Exposes `predict_action()` returning already-unnormalized actions. |
| `server_policy.py` | Entry point (unchanged API): instantiates `PolicyServerWrapper` and starts the websocket server. |

### Server response format change

**Before:** server returned `response["data"]["normalized_actions"]` — clients had to unnormalize locally.

**After:** server returns `response["data"]["actions"]` — already unnormalized, ready to use.

### Client-side migration (all eval interfaces updated)

Every `model2*_interface.py` was updated:

1. **Remove** `read_mode_config` import and all local norm-stat helpers (`get_action_stats`, `get_state_stats`, `get_action_chunk_size`, `unnormalize_actions`, `_check_unnorm_key`).
2. **Add** `server_meta = self.client.get_server_metadata()` in `__init__` to fetch `action_chunk_size` from the server.
3. **Add** `vla_input["unnorm_key"] = self.unnorm_key` before each `predict_action` call.
4. **Change** response parsing from `response["data"]["normalized_actions"]` → `response["data"]["actions"]`.

Updated interfaces: `SimplerEnv`, `VLA-Arena`, `LIBERO-plus`, `Robocasa_tabletop`, `Robocasa_365`, `Robotwin`, `DOMINO`.

### Multi-dataset checkpoints

Checkpoints trained on multiple datasets (e.g. `bridge_rt_1` = `oxe_bridge` + `oxe_rt1`)
are now supported. `PolicyServerWrapper` starts in *multi-key lazy mode* — it does not
build any processor at startup. Each client request must pass `unnorm_key` to select the
correct embodiment statistics. The server metadata includes `available_unnorm_keys`.

```python
# Example client usage (multi-dataset checkpoint)
server_meta = client.get_server_metadata()
# server_meta["available_unnorm_keys"] → ["oxe_bridge", "oxe_rt1"]

vla_input = {"examples": [...], "unnorm_key": "oxe_bridge", ...}
response = client.predict_action(vla_input)
actions = np.array(response["data"]["actions"][0])  # (chunk, D), already unnormalized
```

### action_chunk_size config compatibility

`PolicyServerWrapper` now handles two config styles automatically:
- `action_model.action_horizon` (OXE-style checkpoints)
- `action_model.future_action_window_size + 1` (LIBERO-style checkpoints)

Clients should read `action_chunk_size` from `get_server_metadata()` instead of
computing it locally from the checkpoint.
