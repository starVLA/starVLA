# EgoVLA policy server for the G1 red-ball sim (GR00T-WBC-Bridge)

Serves the starVLA **EgoVLA** model (`framework.name: EgoVLA`) as a GR00T-protocol
policy on `:5555`, so the **GR00T-WBC-Bridge → Isaac-Lab G1** red-ball eval runs
with EgoVLA as the policy — the sim and bridge are used **unchanged**.

```text
Isaac-Lab G1 (g1_redball_eval, unchanged)
  ── ZMQ ──►  GR00T-WBC-Bridge (unchanged, :5555 REQ)
  ── ZMQ ──►  examples/realRobots/EgoVLA/eval_files/server_egovla_g1.py   ← this
```

EgoVLA outputs a 48-dim camera-frame action (per hand: 3 wrist-translation +
6 rot6d + 15 MANO). This server decodes it into the bridge's joint-target
contract:

- **arms**: wrist EE pose (camera → pelvis via the G1 D435 transform) → 7-DoF
  arm joints by pinocchio damped-least-squares IK (`g1_kinematics.py`), warm-started
  from the robot's current arm state;
- **hand**: MANO hand → MANO forward kinematics → fingertips → `HandActuationNet`
  → Inspire finger angles → Dex3-7 (per-finger; articulation is preserved so a
  commanded grasp is physically reachable — grasp *success* is then up to the model);
- `base_height`/`navigate` → neutral (fixed-base tabletop task).

## ⚠️ Required downloads (license-gated — NOT included in this repo)

To use EgoVLA as the policy you must obtain these yourself and point the server at
an **EgoVLA_Release** checkout that contains them:

1. **EgoVLA checkpoint** — the VILA-style `ego_vla_checkpoint/ckpt-*` dir
   (`{llm, vision_tower, mm_projector, traj_decoder}`). From the EgoVLA release.
2. **MANO models** — `mano_v1_2/models/MANO_LEFT.pkl` / `MANO_RIGHT.pkl`. Register
   and download from <https://mano.is.tue.mpg.de> (research license); place under
   `<EgoVLA_Release>/mano_v1_2/models/`.
3. **Hand-retarget net** — `hand_actuation_net.pth` (from the EgoVLA release).
   A copy is expected at `examples/realRobots/EgoVLA/eval_files/assets/hand_actuation_net.pth`
   (git-ignored here; copy it in).

`g1_29dof_with_hand.urdf` (Unitree G1) ships in `assets/` — pinocchio needs only
the URDF text (no meshes).

## Environment

starVLA env (transformers 4.57, torch 2.8/cu128) plus:

```bash
pip install pinocchio smplx chumpy      # pytorch3d is already a starVLA dep
```

(A numpy-alias shim for chumpy is applied automatically at import.)

## Run

```bash
# 1) EgoVLA policy server on :5555
python examples/realRobots/EgoVLA/eval_files/server_egovla_g1.py \
    --port 5555 --egovla_release /path/to/EgoVLA_Release

# 2) then launch the G1 sim + GR00T-WBC-Bridge as usual
#    (bridge: --server_port 5555 --groot_version n1.7
#             --language_instruction "place the red ball in the box")
```

## Files

- `server_egovla_g1.py` — entry; reuses `deployment/model_server`'s `ZmqGr00tPolicyServer`.
- `egovla_g1_policy.py` — `EgoVLAG1Policy`: obs → EgoVLA framework → 48-dim → decode → action dict.
- `decode.py` — 48-dim → pelvis-frame EE poses + Inspire hand (imports only MANO FK from EgoVLA).
- `g1_kinematics.py` — pinocchio G1 arm FK/IK.
- `g1_camera.py` — G1 D435 optical-frame transform (camera → pelvis).
- `assets/` — G1 URDF (+ your copies of the license-gated hand nets).

## Notes / calibration

- EgoVLA is H1-trained; the camera/EE/hand transforms here are the G1 mapping.
  Whether the robot *grasps* depends on the (H1-trained) model output — the
  pipeline is capability-complete (arms reach, fingers articulate); it does not
  cap what the model can command. Finetuning EgoVLA on G1 data is the path to
  reliable grasps.
- The Inspire-12 → Dex3-7 finger correspondence assumes EgoVLA's Inspire order
  matches the Isaac URDF-12 order; this is a calibration point (articulation is
  preserved regardless).
