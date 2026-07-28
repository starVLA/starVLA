# EgoVLA — a selectable VLA model in starVLA

Adds **EgoVLA** (a VILA-based bimanual VLA) to starVLA's plug-and-swap model
zoo, alongside `QwenOFT` / `QwenGR00T` / `PI0` / `PI05`. Select it with
`framework.name: EgoVLA`.

EgoVLA predicts, per future step, a **48-dim** action = two hands ×
(3 wrist-translation + 6 rot6d + 15 MANO pose), in the head-camera frame.

## Architecture (vendored natively — no `EgoVLA_Release` / `llava` dependency)

| Part | Implementation | Where |
|---|---|---|
| Vision tower | SigLIP-384 → `SiglipVisionModel` (transformers, native) | — |
| LLM | Qwen2-1.5B → `Qwen2ForCausalLM` (transformers, native) | — |
| Projector | `mlp_downsample` → `EgoVLAMMProjector` (vendored) | `starVLA/model/modules/vlm/vila_egovla/projector.py` |
| Multimodal splice | `prepare_inputs_labels_for_multimodal` (vendored) | `starVLA/model/modules/vlm/vila_egovla/arch.py` |
| Backbone wrapper | `_EgoVLA_VILA_Interface` | `starVLA/model/modules/vlm/VILA.py` |
| Action head | transformer trajectory decoder → `EgoVLATrajDecoder` (vendored) | `starVLA/model/modules/action_model/EgoVLA_ActionHeader.py` |
| Framework | `EgoVLA` (`@FRAMEWORK_REGISTRY.register("EgoVLA")`) | `starVLA/model/framework/VLM4A/EgoVLA.py` |

The whole pipeline runs inside starVLA's single (transformers 4.57) environment.

## Checkpoint compatibility (verified)

The public `ego_vla_checkpoint/ckpt-6720` is stored VILA-style as four HF
sub-models `{llm, vision_tower, mm_projector, traj_decoder}`. Verified against
those weights:

- `SiglipVisionModel` loads `vision_tower/` natively; `AutoModelForCausalLM`
  loads `llm/` (Qwen2, vocab 151648 incl. action-query placeholder tokens).
- vendored **mm_projector** state-dict matches the checkpoint **6/6** keys
  (strict load OK); vendored **traj_decoder** matches **98/98** keys.
- End-to-end: `encode_images` → multimodal splice → Qwen2 → action-query
  latent (`output_mask` over placeholder ids 151195–151375) → trajectory
  decoder → **(T, 48)** prediction, all with real weights.

Point `framework.qwenvl.base_vlm` at the checkpoint dir to initialise from it.

## Data / proprio

EgoVLA is bimanual + camera-frame. The DataConfig should supply, per step:
`image` (List[PIL], 384px), `lang`, `action` `[T,48]`, and camera-frame
proprio (`proprio_3d` 2×3, `proprio_rot` 2×3, `proprio_hand_finger_tip`
2×5×3). Missing proprio defaults to zeros (model still runs).

## Run

```bash
# train / finetune  (single GPU; on 16GB freeze the VLM or use LoRA)
accelerate launch starVLA/training/train_starvla.py \
  --config_yaml examples/realRobots/EgoVLA/train_files/starvla_egovla.yaml

# deploy over the GR00T ZMQ protocol on :5555 (unchanged GR00T-WBC-Bridge)
python deployment/model_server/server_policy_gr00t_zmq.py \
  --ckpt_path <trained_ckpt> --port 5555 --use_bf16 --unnorm_key <your_key>
```
