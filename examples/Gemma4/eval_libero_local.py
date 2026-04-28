"""
In-process LIBERO evaluation for Gemma4-VLA.

Skips the websocket policy server entirely — loads `Gemma4_PI.from_pretrained`
in the same Python process as the LIBERO simulator and calls `predict_action`
directly. Single GPU, single conda env (`gemma_vla`).

Training used the agentview camera only, so we feed a single front
camera view per step. Action chunk size = future_action_window_size + 1 = 8.

Usage:
    unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
    conda activate gemma_vla
    cd /data/LFT-W02_data/haodong/git_workspace/gemma-vla
    export PYTHONPATH=$PWD
    export LIBERO_HOME=/data/LFT-W02_data/junjie/LIBERO
    export LIBERO_CONFIG_PATH=$LIBERO_HOME/libero
    export PYTHONPATH=$PYTHONPATH:$LIBERO_HOME
    export MUJOCO_GL=osmesa     # CPU rendering, leave GPU for the model
    export HF_HUB_OFFLINE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    CUDA_VISIBLE_DEVICES=1 python examples/Gemma4/eval_libero_local.py \
      --ckpt results/Checkpoints/gemma4_e2b_pi_libero_all_R002d/checkpoints/steps_50000_pytorch_model.pt \
      --task-suite libero_goal --num-trials 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# LIBERO ships pickled numpy init_states; PyTorch 2.6 defaults weights_only=True which
# refuses such files. We trust LIBERO, so make torch.load fall back to the legacy path.
_original_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat


def _patch_gemma4_vision_position_embeddings():
    """
    Replace Gemma4 vision tower's `_position_embeddings` with a gather-based version.

    Stock implementation materializes a `(B, P, 2, position_embedding_size)` one-hot tensor
    (~400 MB at 2 views × 196 patches × 131072 positions in fp32) which OOMs on a contended
    49 GB A6000. Mathematically equivalent gather drops it to a few MB.
    """
    try:
        from transformers.models.gemma4 import modeling_gemma4 as _g4

        cls = _g4.Gemma4VisionPatchEmbedder if hasattr(_g4, "Gemma4VisionPatchEmbedder") else None
        if cls is None:
            # find by attribute
            for name in dir(_g4):
                obj = getattr(_g4, name)
                if isinstance(obj, type) and "PatchEmbedder" in name:
                    cls = obj
                    break
        if cls is None:
            return False

        def _gather_position_embeddings(self, pixel_position_ids: torch.Tensor, padding_positions: torch.Tensor) -> torch.Tensor:
            clamped = pixel_position_ids.clamp(min=0)
            x_idx = clamped[..., 0]
            y_idx = clamped[..., 1]
            table = self.position_embedding_table  # (2, P_size, hidden)
            emb_x = table[0][x_idx]
            emb_y = table[1][y_idx]
            position_embeddings = (emb_x + emb_y).to(table.dtype)
            position_embeddings = torch.where(padding_positions.unsqueeze(-1), torch.zeros_like(position_embeddings), position_embeddings)
            return position_embeddings

        cls._position_embeddings = _gather_position_embeddings
        return True
    except Exception as e:
        print(f"[warn] could not patch Gemma4 vision position embeddings: {e}")
        return False


_patch_gemma4_vision_position_embeddings()

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_libero_local")


def _quat2axisangle(quat):
    """Copied from robosuite (used by LIBERO)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _binarize_gripper_open(open_val) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
    action_high = np.asarray(action_norm_stats["max"])
    action_low = np.asarray(action_norm_stats["min"])
    normalized_actions = np.clip(normalized_actions, -1, 1)
    normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )
    return actions


def get_max_steps(task_suite: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[task_suite]


@dataclasses.dataclass
class EvalArgs:
    ckpt: str
    task_suite: str = "libero_goal"
    num_trials: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    video_out: Optional[str] = None
    unnorm_key: str = "franka"
    log_per_step: bool = False


def run(args: EvalArgs) -> None:
    np.random.seed(args.seed)

    # === LIBERO sim ===
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks
    max_steps = get_max_steps(args.task_suite)
    log.info(f"task_suite={args.task_suite}  num_tasks={num_tasks}  max_steps={max_steps}  trials/task={args.num_trials}")

    # === Model ===
    from starVLA.model.framework.VLM4A.Gemma4PI import Gemma4_PI

    log.info(f"loading checkpoint: {args.ckpt}")
    t0 = time.time()
    model = Gemma4_PI.from_pretrained(args.ckpt)
    log.info(f"from_pretrained ok in {time.time() - t0:.1f}s")

    # Drop unused submodules to free GPU memory on a contended A6000.
    # - audio_tower / embed_audio: ~307M params, never touched in PI path
    # - lm_head: tied to embed_tokens (no extra weight memory) AND required by forward, do NOT drop
    # The forward also computes logits over the full vocab; we patch Gemma4.py::forward to pass
    # logits_to_keep=1 below, which is the bigger memory win (~150 MB at L≈280 in bf16).
    inner = model.qwen_vl_interface.model.model  # the Gemma4Model (one level deeper)
    for attr in ("audio_tower", "embed_audio"):
        if hasattr(inner, attr):
            delattr(inner, attr)
            log.info(f"  dropped model.{attr}")
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if hasattr(torch.cuda, "memory_allocated"):
        log.info(f"GPU memory after move: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    log.info(f"model on {device}, hidden_size={model.qwen_vl_interface.model.config.hidden_size}")

    # === Action stats / chunk size ===
    norm_stats = model.norm_stats[args.unnorm_key]["action"]
    chunk_size = model.future_action_window_size + 1
    log.info(f"unnorm_key={args.unnorm_key}  action_chunk_size={chunk_size}")
    log.info(f"action min: {np.asarray(norm_stats['min'])}")
    log.info(f"action max: {np.asarray(norm_stats['max'])}")

    if args.video_out:
        pathlib.Path(args.video_out).mkdir(parents=True, exist_ok=True)

    # === Eval loop ===
    total_episodes = 0
    total_successes = 0
    per_task_results: Dict[str, Dict[str, int]] = {}

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)

        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)

        task_episodes = 0
        task_successes = 0
        log.info(f"[task {task_id+1}/{num_tasks}] {task_description}")

        for ep_idx in range(args.num_trials):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            t = 0
            step = 0
            done = False
            replay_imgs: List[np.ndarray] = []
            cached_unnorm_actions: Optional[np.ndarray] = None

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Match training preprocessing: rotate both views 180° (training data was rotated)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                replay_imgs.append(img)

                # 7-dim state to match action_model.state_dim=7. LIBERO's robot0_gripper_qpos
                # is 2-finger; take just the first finger to land at exactly 7 dims.
                gripper_q = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
                state7 = np.concatenate(
                    (
                        np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                        _quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32).reshape(-1),
                        gripper_q[:1],
                    )
                )
                assert state7.shape == (7,), f"unexpected state shape {state7.shape}"

                if step % chunk_size == 0:
                    # Training used 2 views (Libero4in1DataConfig: video.primary_image +
                    # video.wrist_image), so we must too. + 7-dim state.
                    example = {
                        "image": [
                            Image.fromarray(img.astype(np.uint8)),
                            Image.fromarray(wrist_img.astype(np.uint8)),
                        ],
                        "lang": str(task_description),
                        "state": state7[None].astype(np.float32),  # (1, 7)
                    }
                    with torch.no_grad():
                        out = model.predict_action([example])
                    normed = out["normalized_actions"][0]  # (chunk, 7)
                    cached_unnorm_actions = unnormalize_actions(normed, norm_stats)

                cur_action = cached_unnorm_actions[step % chunk_size]
                # Re-binarize gripper using training convention
                wv = cur_action[:3]
                rot = cur_action[3:6]
                grip = _binarize_gripper_open(cur_action[6:7])
                action7 = np.concatenate([wv, rot, grip], axis=0)
                obs, _, done, _ = env.step(action7.tolist())

                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            log.info(
                f"  ep{ep_idx} {'SUCCESS' if done else 'fail'}  "
                f"task_sr={task_successes}/{task_episodes}  "
                f"total_sr={total_successes}/{total_episodes} "
                f"({100 * total_successes / total_episodes:.1f}%)"
            )

            # Optional video
            if args.video_out and replay_imgs:
                import imageio

                tag = "success" if done else "failure"
                fname = pathlib.Path(args.video_out) / f"task{task_id}_ep{ep_idx}_{tag}.mp4"
                imageio.mimwrite(str(fname), replay_imgs, fps=10)

        per_task_results[task_description] = {"success": task_successes, "total": task_episodes}
        log.info(
            f"[task {task_id+1}] FINAL "
            f"sr={task_successes}/{task_episodes} "
            f"({100 * task_successes / task_episodes:.1f}%)"
        )
        env.close()

    log.info("=" * 60)
    log.info(f"FINAL TOTAL SR: {total_successes}/{total_episodes} ({100 * total_successes / total_episodes:.1f}%)")
    log.info("Per task:")
    for k, v in per_task_results.items():
        sr = 100 * v["success"] / v["total"] if v["total"] else 0
        log.info(f"  {sr:5.1f}%  {v['success']:3d}/{v['total']:3d}  {k}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task-suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-out", default=None)
    parser.add_argument("--unnorm-key", default="franka")
    args = parser.parse_args()

    run(EvalArgs(
        ckpt=args.ckpt,
        task_suite=args.task_suite,
        num_trials=args.num_trials,
        num_steps_wait=args.num_steps_wait,
        seed=args.seed,
        video_out=args.video_out,
        unnorm_key=args.unnorm_key,
    ))


if __name__ == "__main__":
    main()
