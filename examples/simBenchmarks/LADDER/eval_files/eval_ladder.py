# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""Evaluate a StarVLA policy on the LADDER long-horizon manipulation benchmark."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from model2ladder_interface import ModelClient

SIMPLER_TASKS: tuple[str, ...] = (
    "all10_plate",
    "all10_towel",
    "all1_plate",
    "all1_towel",
    "all2_plate",
    "all2_towel",
    "all3_plate",
    "all3_towel",
    "all4_plate",
    "all4_towel",
    "all5_plate",
    "all5_towel",
    "all6_plate",
    "all6_towel",
    "all7_plate",
    "all7_towel",
    "all8_plate",
    "all8_towel",
    "all9_plate",
    "all9_towel",
    "except_eggplant_2",
    "except_eggplant_3",
    "except_eggplant_4",
    "except_eggplant_5",
    "move_plate_to_towel_2",
    "move_plate_to_towel_3",
    "move_plate_to_towel_4",
    "move_plate_to_towel_5",
    "move_towel_to_plate_3",
    "ordered_plate_2",
    "ordered_plate_3",
    "relay10e",
    "relay2e",
    "relay3e",
    "relay4e",
    "relay5e",
    "relay6e",
    "relay7e",
    "relay8e",
    "relay9e",
    "shuttle10",
    "shuttle7",
    "shuttle8",
    "shuttle9",
    "split2",
    "split3",
    "split4",
    "split5",
    "split6",
    "split7",
    "spoons2",
    "spoons3",
    "spoons4",
    "spoons5",
)

ROBOT_ARENA_TASKS: tuple[str, ...] = (
    "scene36_1goal",
    "scene36_2goal",
    "scene45_1goal",
    "scene45_2goal",
    "scene45_3goal",
    "scene45_4goal",
    "scene45_5goal",
    "scene45_6goal",
    "scene45_all_2",
    "scene45_all_3",
    "scene45_all_4",
    "scene45_all_5",
    "scene45_all_6",
    "scene45_except_2",
    "scene45_except_3",
    "scene45_except_4",
    "scene45_except_5",
    "scene45_except_6",
    "scene50_3goal",
    "scene54_1goal",
    "scene54_2goal",
    "scene54_2goal_ordered",
    "scene54_3goal",
    "scene54_3goal_ordered",
    "scene54_duplicates_2",
    "scene54_duplicates_3",
    "scene54_except",
    "scene54_except_3",
    "scene54_relay_2",
    "scene54_relay_4",
    "scene54_revisit_10",
    "scene54_revisit_3",
    "scene54_revisit_6",
    "scene54_revisit_7",
    "scene54_revisit_8",
    "scene54_revisit_9",
    "scene54_route_2",
    "scene54_route_4",
    "scene54_route_6",
    "scene54_route_7",
    "scene54_route_8",
    "scene54_split_2",
    "scene54_split_3",
    "scene54_transfer_2",
    "scene76_1goal",
    "scene76_2goal",
    "scene76_2goal_ordered",
)

ROBOLAB_TASKS: tuple[str, ...] = (
    "comp_bin_1",
    "comp_bin_10",
    "comp_bin_2",
    "comp_bin_2_distractor",
    "comp_bin_4_distractor",
    "comp_bin_6",
    "comp_identical_2",
    "comp_identical_4",
    "comp_ordered_bin_10",
    "comp_ordered_bin_2",
    "comp_ordered_bin_3",
    "comp_ordered_bin_4",
    "comp_ordered_bin_5",
    "comp_ordered_bin_6",
    "comp_ordered_bin_7",
    "comp_ordered_bin_8",
    "comp_ordered_bin_9",
    "comp_plate_1",
    "comp_relay_2",
    "comp_relay_3",
    "comp_relay_4",
    "comp_relay_5",
    "comp_revisit_1",
    "comp_revisit_2",
    "comp_revisit_3",
    "comp_route_10",
    "comp_route_2",
    "comp_route_2_binfirst",
    "comp_route_3",
    "comp_route_4",
    "comp_route_5",
    "comp_route_6",
    "comp_route_7",
    "comp_route_8",
    "comp_route_9",
    "comp_split_plate_2",
    "comp_split_plate_3",
    "comp_split_plate_4",
    "comp_split_plate_5",
    "comp_split_plate_6",
    "comp_split_plate_7",
    "comp_split_plate_8",
    "comp_transfer_2",
    "comp_transfer_3",
)

SUITE_TASKS: dict[str, tuple[str, ...]] = {
    "simpler": SIMPLER_TASKS,
    "robot_arena": ROBOT_ARENA_TASKS,
    "robolab": ROBOLAB_TASKS,
}

ROBOT_ARENA_K8_OFFSETS: tuple[tuple[float, float], ...] = (
    (-0.03, -0.02),
    (-0.03, +0.02),
    (-0.01, -0.02),
    (-0.01, +0.02),
    (+0.01, -0.02),
    (+0.01, +0.02),
    (+0.03, -0.02),
    (+0.03, +0.02),
)


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _axisangle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-10:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    x, y, z = float(axis[0]), float(axis[1]), float(axis[2])
    c = math.cos(angle)
    s = math.sin(angle)
    v = 1.0 - c
    return np.array(
        [
            [x * x * v + c, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, y * y * v + c, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, z * z * v + c],
        ],
        dtype=np.float64,
    )


def _resolve_ladder_root(ladder_root: Optional[str]) -> Path:
    candidates = []
    if ladder_root:
        candidates.append(Path(ladder_root))
    env_root = os.environ.get("LADDER_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([Path("/app/ladder"), Path.home() / "ladder"])
    for cand in candidates:
        expanded = cand.expanduser().resolve()
        if (expanded / "ladder").is_dir():
            for p in (
                expanded,
                expanded / "build_support" / "python",
                expanded / "robolab_src",
                expanded / "openpi_src" / "packages" / "openpi-client" / "src",
            ):
                if p.is_dir():
                    s = str(p)
                    if s not in sys.path:
                        sys.path.insert(0, s)
            return expanded
    raise FileNotFoundError("LADDER source tree not found. Set LADDER_ROOT or pass --ladder-root /path/to/ladder.")


def _unpack_obs_metrics(raw_obs: Any, prompt_mode: str) -> tuple[np.ndarray, str, bool, bool, float, int]:
    if isinstance(raw_obs, dict):
        images = raw_obs.get("images") or {}
        metadata = raw_obs.get("metadata") or {}
    else:
        images = raw_obs.images or {}
        metadata = raw_obs.metadata or {}

    main_img = np.asarray(images.get("main", next(iter(images.values()))), dtype=np.uint8)
    step_done = bool(metadata.get("done", False))
    truncated = bool(metadata.get("truncated", False))
    oracle = metadata.get("oracle") if isinstance(metadata.get("oracle"), dict) else None
    achieved = [bool(v) for v in oracle.get("achieved", [])] if oracle else []
    subtasks = [str(s) for s in oracle.get("subtasks", [])] if oracle else []
    n_goals = max(len(achieved), 1)
    completed = int(metadata.get("num_subgoals_complete", sum(achieved)))
    if step_done and len(achieved) > 0:
        completed = max(completed, len(achieved))
    progress = float(completed) / float(len(achieved)) if len(achieved) > 0 else float(step_done)

    instruction = str(metadata.get("instruction", ""))
    if prompt_mode == "oracle_hlc" and subtasks:
        active_idx = next((idx for idx, done_flag in enumerate(achieved) if not done_flag), len(subtasks) - 1)
        if 0 <= active_idx < len(subtasks) and subtasks[active_idx]:
            instruction = subtasks[active_idx]

    return main_img, instruction, step_done, truncated, progress, n_goals


def _delta_ee_to_manipulator_actions(
    delta_actions: np.ndarray,
    target_poses: list[np.ndarray],
) -> list[Any]:
    from ladder.components.inference.api.core.action import EndEffectorPoseAction, ManipulatorAction

    out = []
    for i in range(delta_actions.shape[0]):
        raw = delta_actions[i]
        pos = raw[:3].astype(np.float32)
        rot_in = raw[3:6].astype(np.float64)
        gripper = float(1.0 if raw[6] > 0.0 else 0.0) if raw[6] < 0.0 else float(np.clip(raw[6], 0.0, 1.0))

        target_eef = target_poses[i]
        translation = (target_eef[:3, 3] + pos).astype(np.float32)
        r_delta = _axisangle_to_matrix(rot_in)
        rotation = (r_delta @ target_eef[:3, :3]).astype(np.float32)

        new_target = np.eye(4, dtype=np.float32)
        new_target[:3, 3] = translation
        new_target[:3, :3] = rotation
        target_poses[i] = new_target

        out.append(
            ManipulatorAction(
                arm_action=EndEffectorPoseAction(translation=translation, rotation=rotation),
                gripper_action=gripper,
            )
        )
    return out


def evaluate_simpler(
    client: ModelClient,
    ladder_root: Path,
    tasks: Sequence[str],
    n_envs: int,
    max_steps: Optional[int],
    prompt_mode: str,
) -> list[dict[str, Any]]:
    from ladder.components.inference.robots.simpler.env_builder import ABS_CONTROL_MODE
    from ladder.components.inference.robots.simpler.simpler_env import SimplerEnv, SimplerEnvConfig

    asset_cand = ladder_root / "ladder" / "components" / "inference" / "robots" / "simpler" / "maniskill3"
    if "MS_ASSET_DIR" not in os.environ and asset_cand.is_dir():
        os.environ["MS_ASSET_DIR"] = str(asset_cand)

    results = []
    episode_ids = list(range(n_envs))
    for task_name in tasks:
        cfg = SimplerEnvConfig(
            env_name=f"{task_name}-v0",
            robot_name="widowx",
            scene_name="bridge_table_1_v1",
            rgb_overlay_path="",
            sim_freq=500,
            control_mode=ABS_CONTROL_MODE,
            control_freq=5,
            max_episode_steps=0,
            raytracing=False,
            n_envs=n_envs,
            episode_layouts=max(24, n_envs),
        )
        env = SimplerEnv(cfg)
        try:
            obs_list = env.reset(episode_ids)
            client.reset(n_envs=n_envs)
            target_poses = [
                np.asarray(getattr(o.state, "eef_pose_from_target", o.state.eef_pose), dtype=np.float32)
                for o in obs_list
            ]
            ever_success = np.zeros(n_envs, dtype=bool)
            max_progress = np.zeros(n_envs, dtype=np.float32)
            num_subgoals = 1
            horizon = int(max_steps) if max_steps is not None else 400

            for _ in range(horizon):
                images_per_env = []
                instructions = []
                all_truncated = True
                for i, o in enumerate(obs_list):
                    main_img, instr, step_done, truncated, prog, n_goals = _unpack_obs_metrics(o, prompt_mode)
                    num_subgoals = max(num_subgoals, n_goals)
                    ever_success[i] |= step_done
                    max_progress[i] = max(max_progress[i], 1.0 if ever_success[i] else prog)
                    all_truncated = all_truncated and (truncated or ever_success[i])
                    images_per_env.append([main_img])
                    instructions.append(instr)

                if bool(np.all(ever_success)) or all_truncated:
                    break

                delta_actions = client.step_batch(images_per_env, instructions)
                manip_actions = _delta_ee_to_manipulator_actions(delta_actions, target_poses)
                obs_list = env.step(manip_actions)
                for i, o in enumerate(obs_list):
                    target_poses[i] = np.asarray(
                        getattr(o.state, "eef_pose_from_target", o.state.eef_pose), dtype=np.float32
                    )

            task_res = {
                "task": task_name,
                "num_subgoals": num_subgoals,
                "n_episodes": int(n_envs),
                "success_rate": float(np.mean(ever_success)),
                "subgoal_progress": float(np.mean(max_progress)),
            }
            print(
                f"[simpler] {task_name:14s} | subgoals={num_subgoals} | "
                f"success={task_res['success_rate']*100:5.1f}% | "
                f"progress={task_res['subgoal_progress']*100:5.1f}%"
            )
            results.append(task_res)
        finally:
            env.close()
    return results


def evaluate_robot_arena(
    client: ModelClient,
    ladder_root: Path,
    tasks: Sequence[str],
    n_envs: int,
    max_steps: Optional[int],
    prompt_mode: str,
    seed: int = 42,
) -> list[dict[str, Any]]:
    from ladder.components.inference.robots.robot_arena.composition.registry import COMPOSITION_TASKS
    from ladder.components.inference.robots.robot_arena.robot_arena_env import RobotArenaEnv, RobotArenaEnvConfig
    from ladder.components.inference.robots.robot_arena.scene_builder import set_genesis_seed
    from ladder.core.compute.seeder import SEEDER

    data_root_candidates = [
        Path(os.environ["LADDER_ROBOT_ARENA_ASSETS"]) if "LADDER_ROBOT_ARENA_ASSETS" in os.environ else None,
        ladder_root
        / "ladder"
        / "components"
        / "inference"
        / "robots"
        / "robot_arena"
        / "robot_arena_data"
        / "examples"
        / "data",
        Path.home() / ".cache" / "vla-eval" / "assets" / "ladder_robot_arena" / "examples" / "data",
        Path.home() / ".cache" / "vla-eval" / "assets" / "ladder_robot_arena",
    ]
    data_root = next(
        (str(c) for c in data_root_candidates if c is not None and ((c / "bridge").is_dir() or (c / "scenes").is_dir())),
        None,
    )
    if data_root is None:
        raise FileNotFoundError("RobotArena assets not found. Set LADDER_ROBOT_ARENA_ASSETS.")

    results = []
    env: Any = None
    try:
        for task_name in tasks:
            SEEDER.seed_everything(seed)
            set_genesis_seed(seed)
            comp_task = COMPOSITION_TASKS[task_name]
            offsets = [ROBOT_ARENA_K8_OFFSETS[ep % len(ROBOT_ARENA_K8_OFFSETS)] for ep in range(n_envs)]
            padded_single = len(offsets) == 1
            if padded_single:
                offsets = [offsets[0], ROBOT_ARENA_K8_OFFSETS[0]]

            horizon = int(max_steps) if max_steps is not None else int(comp_task.max_episode_steps)
            cfg = RobotArenaEnvConfig(
                data_root=data_root,
                scene_name=comp_task.scene,
                robot_name="widowx",
                max_episode_steps=horizon,
                control_freq=5,
                sim_freq=200,
                sim_substeps=1,
                rigid_constraint_timeconst=0.02,
                camera_width=640,
                camera_height=480,
                per_env_object_offsets=offsets,
                composition_task=task_name,
            )
            if env is not None and env.can_reuse_for(cfg):
                env.reconfigure(cfg)
            else:
                if env is not None:
                    env.close()
                env = RobotArenaEnv(config=cfg)

            obs_list = env.reset()
            if padded_single:
                obs_list = obs_list[:1]
            client.reset(n_envs=n_envs)
            target_poses = [
                np.asarray(getattr(o.state, "eef_pose_from_target", o.state.eef_pose), dtype=np.float32)
                for o in obs_list
            ]
            ever_success = np.zeros(n_envs, dtype=bool)
            max_progress = np.zeros(n_envs, dtype=np.float32)
            num_subgoals = 1

            for _ in range(horizon):
                images_per_env = []
                instructions = []
                all_truncated = True
                for i, o in enumerate(obs_list):
                    main_img, instr, step_done, truncated, prog, n_goals = _unpack_obs_metrics(o, prompt_mode)
                    num_subgoals = max(num_subgoals, n_goals)
                    ever_success[i] |= step_done
                    max_progress[i] = max(max_progress[i], 1.0 if ever_success[i] else prog)
                    all_truncated = all_truncated and (truncated or ever_success[i])
                    images_per_env.append([main_img])
                    instructions.append(instr)

                if bool(np.all(ever_success)) or all_truncated:
                    break

                delta_actions = client.step_batch(images_per_env, instructions)
                manip_actions = _delta_ee_to_manipulator_actions(delta_actions, target_poses)
                step_actions = [manip_actions[0], manip_actions[0]] if padded_single else manip_actions
                obs_list = env.step(step_actions)
                if padded_single:
                    obs_list = obs_list[:1]
                for i, o in enumerate(obs_list):
                    target_poses[i] = np.asarray(
                        getattr(o.state, "eef_pose_from_target", o.state.eef_pose), dtype=np.float32
                    )

            task_res = {
                "task": task_name,
                "num_subgoals": num_subgoals,
                "n_episodes": int(n_envs),
                "success_rate": float(np.mean(ever_success)),
                "subgoal_progress": float(np.mean(max_progress)),
            }
            print(
                f"[robot_arena] {task_name:14s} | subgoals={num_subgoals} | "
                f"success={task_res['success_rate']*100:5.1f}% | "
                f"progress={task_res['subgoal_progress']*100:5.1f}%"
            )
            results.append(task_res)
    finally:
        if env is not None:
            env.close()
    return results


def evaluate_robolab(
    client: ModelClient,
    ladder_root: Path,
    tasks: Sequence[str],
    n_envs: int,
    max_steps: Optional[int],
    prompt_mode: str,
    action_mode: str,
) -> list[dict[str, Any]]:
    import cv2  # noqa: F401
    from isaaclab.app import AppLauncher

    asset_candidates = [
        Path(os.environ["ROBOLAB_ASSET_DIR"]) if "ROBOLAB_ASSET_DIR" in os.environ else None,
        Path(os.environ["LADDER_ROBOLAB_ASSETS"]) if "LADDER_ROBOLAB_ASSETS" in os.environ else None,
        ladder_root / "ladder" / "components" / "inference" / "robots" / "robolab" / "assets",
        Path.home() / ".cache" / "vla-eval" / "assets" / "ladder_robolab" / "assets",
        Path.home() / ".cache" / "vla-eval" / "assets" / "ladder_robolab",
    ]
    for cand in asset_candidates:
        if cand is not None and cand.is_dir():
            os.environ["ROBOLAB_ASSET_DIR"] = str(cand)
            break

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    launcher = AppLauncher(headless=True, enable_cameras=True)
    import carb.settings
    import omni.kit.app
    import omni.timeline
    import robolab.constants
    from ladder.components.inference.api.core.action import JointAnglesAction, ManipulatorAction
    from ladder.components.inference.robots.robolab.composition.specs import (
        SPECS,
        decomp_cache_entries,
        task_class_name,
        task_target,
    )
    from ladder.components.inference.robots.robolab.droid_conversions import DroidJointposConverter
    from ladder.components.inference.robots.robolab.oracle import build_oracle
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    carb.settings.get_settings().set("/persistent/app/usd/muteUsdDiagnostics", False)
    atexit.register(lambda: (sys.stdout.flush(), sys.stderr.flush(), os._exit(0)))
    _ = launcher.app

    auto_register_droid_envs(
        task_dirs=None,
        task=[task_target(t) if t in SPECS else t for t in tasks],
        cameras=None,
    )
    plan_entries = decomp_cache_entries()

    results = []
    for task_name in tasks:
        cls_name = task_class_name(task_name) if task_name in SPECS else task_name
        env, env_cfg = create_env(
            cls_name,
            device="cuda:0",
            num_envs=n_envs,
            instruction_type="default",
            policy="ladder_runner",
            renderer="realtime",
            rendering_mode="balanced",
        )
        try:
            oracle = build_oracle(getattr(env.cfg, "subtasks", None), env_cfg.instruction)
            raw_obs, _ = env.reset()
            raw_obs, _ = env.reset()
            converter = DroidJointposConverter(num_envs=n_envs, image_size=(224, 224))
            client.reset(n_envs=n_envs)

            horizon = int(max_steps) if max_steps is not None else int(env.max_episode_length)
            ever_success = np.zeros(n_envs, dtype=bool)
            max_progress = np.zeros(n_envs, dtype=np.float32)
            num_subgoals = max(len(oracle) if oracle else 1, 1)

            commands_list = [g.command for g in oracle] if oracle else [env_cfg.instruction]
            cached = (plan_entries.get(cls_name) or {}).get("subtasks")
            subtasks = list(cached) if cached and len(cached) == len(commands_list) else commands_list

            for _ in range(horizon):
                image_obs = raw_obs["image_obs"]
                proprio = raw_obs["proprio_obs"]
                env_results = env.get_env_results()
                active = set(env.active_env_ids)

                images_per_env = []
                instructions = []
                states_list = []
                joints_batch = proprio["arm_joint_pos"].detach().cpu().numpy().astype(np.float32)
                grip_batch = proprio["gripper_pos"].detach().cpu().numpy().astype(np.float32)

                for i in range(n_envs):
                    succ = bool(env_results[i]["success"])
                    ever_success[i] |= succ
                    if oracle:
                        achieved = [bool(g.check(env, env_id=i)) for g in oracle]
                        prog = float(sum(achieved)) / float(len(achieved))
                        active_idx = next((idx for idx, d in enumerate(achieved) if not d), len(subtasks) - 1)
                        instr = subtasks[active_idx] if prompt_mode == "oracle_hlc" else str(env_cfg.instruction)
                    else:
                        prog = float(succ)
                        instr = str(env_cfg.instruction)
                    max_progress[i] = max(max_progress[i], 1.0 if ever_success[i] else prog)

                    main_img = image_obs["over_shoulder_left_camera"][i].detach().cpu().numpy().astype(np.uint8)
                    wrist_img = image_obs["wrist_cam"][i].detach().cpu().numpy().astype(np.uint8)
                    images_per_env.append([main_img, wrist_img])
                    instructions.append(instr)
                    grip_val = float(grip_batch[i].reshape(-1)[0])
                    states_list.append(np.concatenate([joints_batch[i][:7], np.array([grip_val], dtype=np.float32)]))

                if bool(np.all(ever_success)) or not active:
                    break

                raw_act = client.step_batch(images_per_env, instructions, states=np.stack(states_list, axis=0))
                manip_actions = []
                for i in range(n_envs):
                    arm = raw_act[i, :7].astype(np.float32)
                    joints = (
                        (joints_batch[i, :7] + np.clip(arm, -1.0, 1.0) * 0.2).astype(np.float32)
                        if action_mode == "joint_vel"
                        else arm
                    )
                    gripper = 0.0 if float(raw_act[i, 7]) > 0.5 else 1.0
                    manip_actions.append(
                        ManipulatorAction(arm_action=JointAnglesAction(angles=joints), gripper_action=gripper)
                    )

                joint_cmds = converter.actions_to_jointpos(manip_actions, joints_batch)
                timeline = omni.timeline.get_timeline_interface()
                kit_app = omni.kit.app.get_app()
                while not timeline.is_playing():
                    kit_app.update()
                raw_obs, _, _, _, _ = env.step(torch.as_tensor(joint_cmds, device=env.device))

            task_res = {
                "task": task_name,
                "num_subgoals": num_subgoals,
                "n_episodes": int(n_envs),
                "success_rate": float(np.mean(ever_success)),
                "subgoal_progress": float(np.mean(max_progress)),
            }
            print(
                f"[robolab] {task_name:14s} | subgoals={num_subgoals} | "
                f"success={task_res['success_rate']*100:5.1f}% | "
                f"progress={task_res['subgoal_progress']*100:5.1f}%"
            )
            results.append(task_res)
        finally:
            env.close()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate StarVLA policy on LADDER benchmark suites.")
    parser.add_argument(
        "--suite",
        type=str,
        default="simpler",
        choices=("simpler", "robot_arena", "robolab"),
        help="LADDER benchmark suite to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma-separated list of task names or 'all'.",
    )
    parser.add_argument("--n-envs", type=int, default=None, help="Parallel environments / episodes per task.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max step override.")
    parser.add_argument(
        "--prompt-mode",
        type=str,
        default="flat",
        choices=("flat", "oracle_hlc"),
        help="Instruction mode: 'flat' (high-level goal) or 'oracle_hlc' (active subgoal prompt).",
    )
    parser.add_argument(
        "--action-mode",
        type=str,
        default="joint_vel",
        choices=("delta_ee", "joint_pos", "joint_vel"),
        help="Action mode for RoboLab DROID tasks.",
    )
    parser.add_argument("--policy-setup", type=str, default=None, help="Override policy_setup.")
    parser.add_argument("--unnorm-key", type=str, default=None, help="Dataset unnormalization key.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="StarVLA policy server host.")
    parser.add_argument("--port", type=int, default=10093, help="StarVLA policy server port.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--ladder-root", type=str, default=None, help="Path to insait-institute/ladder checkout.")
    parser.add_argument("--results-file", type=str, default=None, help="Path to output JSON summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed_everywhere(args.seed)
    ladder_root = _resolve_ladder_root(args.ladder_root)

    all_suite_tasks = SUITE_TASKS[args.suite]
    if args.tasks.strip().lower() == "all":
        tasks = list(all_suite_tasks)
    else:
        tasks = [t.strip() for t in args.tasks.replace(",", " ").split() if t.strip()]
        unknown = [t for t in tasks if t not in all_suite_tasks]
        if unknown:
            raise ValueError(f"Unknown task(s) {unknown} for suite={args.suite!r}.")

    n_envs = args.n_envs if args.n_envs is not None else (24 if args.suite == "simpler" else 8)
    policy_setup = args.policy_setup or ("franka_droid" if args.suite == "robolab" else "widowx_bridge")

    client = ModelClient(
        policy_setup=policy_setup,
        unnorm_key=args.unnorm_key,
        action_ensemble=(args.suite == "simpler"),
        host=args.host,
        port=args.port,
        n_envs=n_envs,
    )

    t0 = time.time()
    if args.suite == "simpler":
        per_task = evaluate_simpler(client, ladder_root, tasks, n_envs, args.max_steps, args.prompt_mode)
    elif args.suite == "robot_arena":
        per_task = evaluate_robot_arena(client, ladder_root, tasks, n_envs, args.max_steps, args.prompt_mode, args.seed)
    else:
        per_task = evaluate_robolab(
            client, ladder_root, tasks, n_envs, args.max_steps, args.prompt_mode, args.action_mode
        )
    elapsed_s = time.time() - t0

    mean_sr = float(np.mean([r["success_rate"] for r in per_task])) if per_task else 0.0
    mean_prog = float(np.mean([r["subgoal_progress"] for r in per_task])) if per_task else 0.0

    summary = {
        "suite": args.suite,
        "prompt_mode": args.prompt_mode,
        "n_tasks": len(per_task),
        "n_envs_per_task": n_envs,
        "success_rate": mean_sr,
        "subgoal_progress": mean_prog,
        "elapsed_seconds": round(elapsed_s, 2),
        "server_metadata": client.server_metadata,
        "per_task": per_task,
    }
    print(
        f"\n=== LADDER ({args.suite}) Summary ===\n"
        f"Tasks evaluated  : {len(per_task)}\n"
        f"Success rate     : {mean_sr * 100:.2f}%\n"
        f"Subgoal progress : {mean_prog * 100:.2f}%\n"
    )
    if args.results_file:
        out_path = Path(args.results_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved evaluation summary to {out_path}")


if __name__ == "__main__":
    main()
