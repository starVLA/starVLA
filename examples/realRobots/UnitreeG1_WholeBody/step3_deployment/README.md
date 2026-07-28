# Step 3: Deployment

This step connects a trained StarVLA checkpoint to the GR00T-WholeBodyControl / SONIC-style G1 controller stack. The purpose is to replace the model-serving part of the NVlabs VLA workflow with StarVLA while leaving real-time robot execution to the controller infra.

The recommended shape is:

```text
StarVLA checkpoint
  -> StarVLA WebSocket policy server
  -> G1 policy client / adapter
  -> user-owned controller stack
  -> Unitree G1
```

For the primary example, keep GR00T-WholeBodyControl / SONIC as the controller side. StarVLA should replace or provide only the VLA policy server/client boundary, not the whole-body controller.

## NVlabs Deployment Shape

The upstream inference path has:

```text
Isaac-GR00T PolicyServer
  -> VLA inference client
  -> camera server
  -> C++ deploy / SONIC WBC
  -> Unitree G1
```

The StarVLA version should be:

```text
StarVLA WebSocket policy server
  -> G1 StarVLA adapter
  -> camera server / robot state source
  -> C++ deploy / SONIC WBC or equivalent controller
  -> Unitree G1
```

Useful upstream references:

- VLA inference: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html
- VLA workflow: https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html
- GR00T-WholeBodyControl repo: https://github.com/NVlabs/GR00T-WholeBodyControl

## StarVLA Policy Server

Start StarVLA's policy server on the GPU machine:

```bash
export PYTHONPATH=$(pwd):${PYTHONPATH}

CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
  --ckpt_path /path/to/g1/checkpoints/steps_<N>_pytorch_model.pt \
  --port 5694 \
  --use_bf16
```

The server returns unnormalized action chunks through `deployment/model_server/policy_wrapper.py`. For this G1 example, continuous state/action groups should be trained and unnormalized with `q99` statistics (`q01`/`q99`), not `min_max`, unless an existing checkpoint explicitly used a different normalization contract.

This replaces the upstream `run_gr00t_server.py` role. The controller side should still see the same high-level action semantics expected by the selected G1 route.

## StarVLA Inference Wrapper

The sim / eval data flow is:

```text
camera + robot state
  -> GR00T inference loop
  -> PolicyClient shim
  -> StarVLA websocket policy server
  -> unnormalized 78D action chunk
  -> SONIC / C++ deploy
```

The client wrapper lives here:

```text
examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/model2unitree_g1_interface.py
```

It converts the GR00T-style observation dict into the StarVLA request format:

```python
{
    "examples": [{
        "image": [ego_view_image],
        "lang": task_prompt,
        "state": flat_proprioception,
    }],
    "unnorm_key": "new_embodiment",
}
```

and splits the returned action chunk into:

```text
action.motion_token      -> [T, 64]
action.left_hand_joints  -> [T, 7]
action.right_hand_joints -> [T, 7]
```

To let the existing GR00T `run_vla_inference.py` import this wrapper without editing its source, prepend both the StarVLA repo root and this deployment directory to `PYTHONPATH` before launching the GR00T sim stack:

```bash
cd starVLA
export PYTHONPATH=$PWD:$PWD/examples/realRobots/UnitreeG1_WholeBody/step3_deployment:${PYTHONPATH}
```

The compatibility shim at `step3_deployment/gr00t/policy/server_client.py` then shadows the GR00T `PolicyClient` import and forwards calls to StarVLA.

The deployment scripts now discover the normalization key from the policy server metadata when possible. For the current smoke-tested checkpoint, that key is `new_embodiment`.

For a direct local smoke test:

```bash
cd starVLA
PYTHONPATH=$PWD:$PWD/examples/realRobots/UnitreeG1_WholeBody/step3_deployment \
  python examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/local_self_test.py \
    --ckpt-path results/Checkpoints/starvla_qwenoft_g1_sonic_smoke/checkpoints/steps_1_pytorch_model.pt \
    --server-host 127.0.0.1 \
    --server-port 5694
```

The helper script `step3_deployment/run_starvla_eval.sh` wraps the same path setup for repeated checks.
The matching server launcher is `step3_deployment/run_policy_server.sh`.

## G1 Adapter Responsibilities

The G1-side adapter should do only bridge work:

1. Read camera images and robot state from the user's G1 infra.
2. Build the StarVLA request:

   ```python
   {
       "examples": [{
           "image": [cam0, cam1],
           "state": state,
           "lang": prompt,
       }],
       "unnorm_key": "unitree_g1_sonic",
   }
   ```

3. Receive `actions` with shape `[B, T, action_dim]`.
4. Split each flat action into controller groups.
5. Apply safety limits.
6. Publish to the SONIC decoder / C++ deploy side / user G1 controller bridge.

For the primary NVlabs-style route, the adapter should preserve the 78D action meaning:

```text
action[0:64]   -> SONIC motion latent
action[64:71]  -> left hand command
action[71:78]  -> right hand command
```

Verify this against the actual training dataset and controller contract before real execution.

The adapter should not reimplement normalization math. It should consume the unnormalized action chunk returned by the StarVLA policy server and only handle grouping, clipping, stale-action checks, and controller publishing.

## Suggested Deployment Files

When implementation starts, add:

```text
examples/realRobots/UnitreeG1_WholeBody/step3_deployment/eval_files/
  run_policy_server.sh
  model2unitree_g1_interface.py
  local_self_test.py
  replay_lerobot_episode.py
  mock_g1_controller.py
  g1_controller_adapter.py
  g1_safety_limits.example.yaml
```

## Dry-Run Order

Do not run directly on the real robot. Use this order:

1. `local_self_test.py` with synthetic observation.
2. `replay_lerobot_episode.py` with a real recorded episode.
3. `mock_g1_controller.py` to inspect action groups.
4. Simulation or third-party controller dry-run.
5. Real robot with policy paused.
6. Real robot with low-speed, limited action groups.
7. Gradually enable more action groups.

## Safety Requirements

Deployment must have:

- emergency stop independent from StarVLA,
- action clipping per group,
- velocity / acceleration limits,
- stale action timeout,
- camera/state freshness check,
- policy pause command,
- controller health check,
- clear operator procedure.

If any of these are missing, stay in dry-run or simulation.

## Handoff to sdk_tools

Use [sdk_tools](../sdk_tools/README.md) for visualization, data inspection, mock controller scripts, third-party clone instructions, and conversion sanity checks.
