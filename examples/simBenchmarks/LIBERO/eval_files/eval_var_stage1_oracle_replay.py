"""Oracle replay evaluation for VAR Stage 1 action tokenizers on LIBERO.

This script compares original expert actions against tokenizer-reconstructed
expert actions in the real LIBERO simulator.  It is intentionally action-only:
no policy model, images, language model, or flow-matching action head is used.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Passing a BlockManager to DataFrame is deprecated.*")

import imageio
import imageio.v3 as iio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf
from tqdm import tqdm

from examples.LIBERO.eval_files.eval_var_stage1_reconstruction import _load_model
from starVLA.dataloader.var_stage1_action_dataset import VARStage1ActionDataset
from starVLA.training.train_var_stage1 import load_starvla_base_config


# LIBERO stores trusted numpy arrays in torch files for init states.  PyTorch
# 2.6+ defaults to weights_only=True, which rejects those files unless we fall
# back to the legacy mode.
_TORCH_LOAD_ORIG = torch.load


def _torch_load_with_libero_fallback(*args: Any, **kwargs: Any) -> Any:
    try:
        return _TORCH_LOAD_ORIG(*args, **kwargs)
    except Exception as exc:
        if "Weights only load failed" not in str(exc):
            raise
        kwargs = dict(kwargs)
        kwargs["weights_only"] = False
        return _TORCH_LOAD_ORIG(*args, **kwargs)


torch.load = _torch_load_with_libero_fallback


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SUITE_TO_DATASET_PREFIX = {
    "libero_spatial": "libero_spatial_",
    "libero_object": "libero_object_",
    "libero_goal": "libero_goal_",
    "libero_10": "libero_10_",
}
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
LIBERO_HDF5_ROOT = Path("/home/zhangfeihong/LIBERO/libero/datasets")


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float64)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_libero_env(task: Any, seed: int) -> tuple[Any, str]:
    task_description = str(task.language)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env, task_description


def _safe_task_text(tasks: Any, task_index: int) -> str:
    if task_index in tasks.index:
        row = tasks.loc[task_index]
        if hasattr(row, "to_dict"):
            return str(row["task"])
        return str(row)
    row = tasks[tasks["task_index"] == task_index].iloc[0]
    return str(row["task"])


def _dataset_for_suite(stage1_dataset: VARStage1ActionDataset, suite_name: str) -> Any:
    prefix = SUITE_TO_DATASET_PREFIX[suite_name]
    matches = [
        dataset
        for dataset in stage1_dataset.source_dataset.datasets
        if str(dataset.dataset_name).startswith(prefix)
    ]
    if len(matches) != 1:
        names = [str(dataset.dataset_name) for dataset in stage1_dataset.source_dataset.datasets]
        raise ValueError(f"Expected one dataset for {suite_name}, got {len(matches)} from {names}.")
    return matches[0]


def _get_episode_task_index(dataset: Any, trajectory_id: int) -> int:
    data = dataset.get_trajectory_data(int(trajectory_id))
    if "task_index" not in data:
        raise KeyError(f"Trajectory {trajectory_id} in {dataset.dataset_name} has no task_index column.")
    return int(data["task_index"].iloc[0])


def _select_episodes(
    dataset: Any,
    *,
    task_ids: set[int] | None,
    max_tasks: int,
    num_episodes_per_task: int,
) -> list[dict[str, Any]]:
    task_to_seen: dict[int, int] = defaultdict(int)
    selected: list[dict[str, Any]] = []
    enabled_tasks: set[int] = set()

    for trajectory_id, trajectory_length in zip(dataset.trajectory_ids, dataset.trajectory_lengths, strict=True):
        task_index = _get_episode_task_index(dataset, int(trajectory_id))
        if task_ids is not None and task_index not in task_ids:
            continue
        if max_tasks > 0 and task_index not in enabled_tasks and len(enabled_tasks) >= max_tasks:
            continue
        if task_to_seen[task_index] >= num_episodes_per_task:
            continue

        episode_ordinal_for_task = task_to_seen[task_index]
        task_to_seen[task_index] += 1
        enabled_tasks.add(task_index)
        selected.append(
            {
                "trajectory_id": int(trajectory_id),
                "trajectory_length": int(trajectory_length),
                "task_index": int(task_index),
                "task_description": _safe_task_text(dataset.tasks, int(task_index)),
                "init_state_index": int(episode_ordinal_for_task),
            }
        )
    return selected


def _task_text_to_hdf5_path(suite_name: str, task_text: str) -> Path:
    stem = task_text.lower().replace(" ", "_")
    suite_root = LIBERO_HDF5_ROOT / suite_name
    exact = suite_root / f"{stem}_demo.hdf5"
    if exact.exists():
        return exact
    matches = sorted(suite_root.glob(f"*_{stem}_demo.hdf5"))
    if len(matches) == 1:
        return matches[0]
    return exact


def _concat_action_dict(data: dict[str, Any], action_keys: list[str]) -> np.ndarray:
    values = []
    for key in action_keys:
        value = data[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        values.append(np.asarray(value, dtype=np.float32))
    return np.concatenate(values, axis=1)


def _split_action_array(actions: np.ndarray | torch.Tensor, action_keys: list[str]) -> dict[str, torch.Tensor]:
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions, dtype=torch.float32)
    chunks = torch.split(actions, [1] * len(action_keys), dim=1)
    return {key: chunk.contiguous() for key, chunk in zip(action_keys, chunks, strict=True)}


def _denormalize_action_chunk(dataset: Any, normalized_actions: torch.Tensor) -> np.ndarray:
    data = _split_action_array(normalized_actions.detach().cpu(), dataset.modality_keys["action"])
    raw_data = dataset.transforms.unapply(data)
    return _concat_action_dict(raw_data, dataset.modality_keys["action"])


def _get_expert_actions(dataset: Any, trajectory_id: int) -> np.ndarray:
    trajectory = dataset.get_trajectory_data(int(trajectory_id))
    actions = np.stack(trajectory["action"].to_numpy()).astype(np.float32)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected raw expert actions [T, 7], got {actions.shape}.")
    return actions


def _hdf5_actions_as_lerobot_open(actions: np.ndarray) -> np.ndarray:
    converted = np.asarray(actions, dtype=np.float32).copy()
    converted[:, 6] = (converted[:, 6] < 0).astype(np.float32)
    return converted


def _match_hdf5_demo_by_actions(
    *,
    suite_name: str,
    task_text: str,
    expert_actions: np.ndarray,
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for init_state_strategy='hdf5_action'.") from exc

    hdf5_path = _task_text_to_hdf5_path(suite_name, task_text)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Could not find LIBERO HDF5 demo file: {hdf5_path}")

    best: dict[str, Any] | None = None
    with h5py.File(hdf5_path, "r") as handle:
        for demo_name in handle["data"].keys():
            demo = handle["data"][demo_name]
            demo_actions = _hdf5_actions_as_lerobot_open(np.asarray(demo["actions"], dtype=np.float32))
            best_shift = 0
            best_mse = float("inf")
            best_count = 0
            max_shift = min(25, len(demo_actions))
            for shift in range(max_shift):
                count = min(len(demo_actions) - shift, len(expert_actions))
                if count <= 10:
                    continue
                mse = float(((demo_actions[shift : shift + count] - expert_actions[:count]) ** 2).mean())
                if mse < best_mse:
                    best_mse = mse
                    best_shift = shift
                    best_count = count
            if best is None or best_mse < best["action_mse"]:
                best = {
                    "hdf5_path": str(hdf5_path),
                    "demo_name": str(demo_name),
                    "init_state": np.asarray(demo.attrs["init_state"]),
                    "model_file": str(demo.attrs["model_file"]),
                    "action_mse": float(best_mse),
                    "action_shift": int(best_shift),
                    "matched_steps": int(best_count),
                    "demo_length": int(len(demo_actions)),
                }

    if best is None:
        raise RuntimeError(f"No HDF5 demo action match found for {task_text!r}.")
    return best


def _reset_env_from_hdf5_model_file(env: Any, model_file: str) -> None:
    """Reset the simulator to the exact MuJoCo XML stored with a LIBERO demo."""

    def rewrite_legacy_paths(xml: str) -> str:
        import robosuite

        robosuite_root = Path(robosuite.__file__).resolve().parent
        libero_assets_root = Path(get_libero_path("assets")).resolve()
        legacy_roots = {
            "/Users/yifengz/workspace/robosuite-master/robosuite": robosuite_root,
            "/home/yifengz/workspace/robosuite-master/robosuite": robosuite_root,
            "/Users/yifengz/workspace/libero-dev/chiliocosm/assets": libero_assets_root,
            "/Users/yifengz/workspace/libero-dev/chiliocosm": libero_assets_root.parent,
            "/home/yifengz/workspace/libero-dev/chiliocosm/assets": libero_assets_root,
            "/home/yifengz/workspace/libero-dev/chiliocosm": libero_assets_root.parent,
        }
        for old_root, new_root in legacy_roots.items():
            xml = xml.replace(old_root, str(new_root))
        return xml

    inner_env = getattr(env, "env", None)
    if inner_env is not None and hasattr(inner_env, "edit_model_xml"):
        try:
            model_file = inner_env.edit_model_xml(model_file)
        except ValueError:
            # Some LIBERO demo XMLs contain legacy absolute asset paths that do
            # not include the "robosuite" path component expected by robosuite's
            # helper. Rewrite those paths here before handing the XML to mujoco.
            model_file = rewrite_legacy_paths(model_file)
    model_file = rewrite_legacy_paths(model_file)
    env.reset_from_xml_string(model_file)
    sim = getattr(getattr(env, "env", env), "sim", None)
    if sim is not None and hasattr(sim, "reset"):
        sim.reset()


def _get_reconstructed_actions(
    *,
    stage1_dataset: VARStage1ActionDataset,
    dataset: Any,
    trajectory_id: int,
    trajectory_length: int,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    horizon = int(stage1_dataset.action_spec.horizon)
    reconstructed: list[np.ndarray] = []

    with torch.no_grad():
        for base_index in range(0, int(trajectory_length), horizon):
            raw_data = stage1_dataset._get_action_only_data(dataset, int(trajectory_id), int(base_index))
            transformed = dataset.transforms(dict(raw_data))
            normalized = _concat_action_dict(transformed, dataset.modality_keys["action"])
            normalized_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=device).unsqueeze(0)
            recon_norm = model(normalized_tensor)["recon"][0].detach().cpu()
            recon_raw = _denormalize_action_chunk(dataset, recon_norm)
            reconstructed.append(recon_raw)

    return np.concatenate(reconstructed, axis=0)[: int(trajectory_length)].astype(np.float32)


def _get_fast_reconstructed_actions(
    *,
    stage1_dataset: VARStage1ActionDataset,
    dataset: Any,
    trajectory_id: int,
    trajectory_length: int,
    fast_tokenizer: Any,
) -> np.ndarray:
    horizon = int(stage1_dataset.action_spec.horizon)
    reconstructed: list[np.ndarray] = []

    for base_index in range(0, int(trajectory_length), horizon):
        raw_data = stage1_dataset._get_action_only_data(dataset, int(trajectory_id), int(base_index))
        transformed = dataset.transforms(dict(raw_data))
        normalized = _concat_action_dict(transformed, dataset.modality_keys["action"])
        tokens = fast_tokenizer(normalized[None].astype(np.float32))
        decoded = fast_tokenizer.decode(tokens, time_horizon=horizon, action_dim=stage1_dataset.action_spec.action_dim)
        decoded_tensor = torch.as_tensor(np.asarray(decoded)[0], dtype=torch.float32)
        recon_raw = _denormalize_action_chunk(dataset, decoded_tensor)
        reconstructed.append(recon_raw)

    return np.concatenate(reconstructed, axis=0)[: int(trajectory_length)].astype(np.float32)


def _to_libero_env_action(action: np.ndarray, *, gripper_mode: str) -> list[float]:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 7:
        raise ValueError(f"Expected action shape [7], got {action.shape}.")
    if gripper_mode == "open01":
        gripper = 1.0 - 2.0 * float(action[6] > 0.5)
    elif gripper_mode == "close01":
        gripper = 2.0 * float(action[6] > 0.5) - 1.0
    elif gripper_mode == "raw":
        gripper = float(action[6])
    else:
        raise ValueError(f"Unsupported gripper_mode={gripper_mode!r}.")
    return np.concatenate([action[:6], np.asarray([gripper], dtype=np.float32)]).astype(np.float32).tolist()


def _run_episode(
    *,
    env: Any,
    init_state: np.ndarray,
    actions: np.ndarray,
    max_steps: int,
    num_steps_wait: int,
    gripper_mode: str,
    video_path: Path | None,
    reset_before_init: bool = True,
) -> dict[str, Any]:
    if reset_before_init:
        env.reset()
    obs = env.set_init_state(init_state)
    done = False
    replay_images: list[np.ndarray] = []
    success_trace: list[bool] = []

    def append_frame() -> None:
        if video_path is not None:
            replay_images.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))

    append_frame()

    for _ in range(num_steps_wait):
        obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        success_trace.append(bool(done))
        append_frame()
        if done:
            break

    steps_executed = 0
    if not done:
        for action in actions[:max_steps]:
            obs, _, done, _ = env.step(_to_libero_env_action(action, gripper_mode=gripper_mode))
            steps_executed += 1
            success_trace.append(bool(done))
            append_frame()
            if done:
                break

    final_check_success = bool(env.check_success()) if hasattr(env, "check_success") else bool(done)

    if video_path is not None and replay_images:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(video_path, replay_images, fps=10)

    return {
        "success": bool(done),
        "final_check_success": final_check_success,
        "success_ever": bool(done or any(success_trace)),
        "steps_executed": int(steps_executed),
        "num_actions": int(actions.shape[0]),
    }


def _get_primary_video_path(dataset: Any, trajectory_id: int) -> Path:
    chunk_index = dataset.get_episode_chunk(int(trajectory_id))
    pattern = str(dataset.video_path_pattern)
    candidates = [
        "observation.images.image",
        "primary_image",
        "video.primary_image",
    ]
    for video_key in candidates:
        path = dataset.dataset_path / pattern.format(
            episode_chunk=chunk_index,
            episode_index=int(trajectory_id),
            video_key=video_key,
        )
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find primary video for trajectory {trajectory_id}; tried keys={candidates}."
    )


def _render_init_state_match_images(
    *,
    env: Any,
    init_states: np.ndarray,
    num_steps_wait: int,
) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for idx, init_state in enumerate(init_states):
        env.reset()
        obs = env.set_init_state(init_state)
        for _ in range(num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

        image = np.asarray(obs["agentview_image"], dtype=np.float32)
        images.append(image)
    return images


def _match_init_state_by_first_frame(
    *,
    candidate_images: list[np.ndarray],
    dataset: Any,
    trajectory_id: int,
) -> tuple[int, float]:
    video_path = _get_primary_video_path(dataset, int(trajectory_id))
    target = np.asarray(iio.imread(video_path, index=0), dtype=np.float32)
    best_index = 0
    best_mse = float("inf")
    for idx, image in enumerate(candidate_images):
        mse = min(
            float(((candidate - target) ** 2).mean())
            for candidate in (image, image[::-1, ::-1])
        )
        if mse < best_mse:
            best_mse = mse
            best_index = idx
    return int(best_index), float(best_mse)


def _load_stage1(checkpoint_path: Path, device: torch.device) -> tuple[Any, Any, VARStage1ActionDataset]:
    model, checkpoint = _load_model(checkpoint_path, device)
    train_cfg = OmegaConf.create(checkpoint["stage1_config"])
    base_cfg = load_starvla_base_config(train_cfg)
    stage1_dataset = VARStage1ActionDataset(
        base_cfg,
        mode="train",
        balance_dataset_weights=bool(train_cfg.data.get("balance_dataset_weights", False)),
        balance_trajectory_weights=bool(train_cfg.data.get("balance_trajectory_weights", False)),
        seed=int(train_cfg.experiment.get("seed", 42)),
        return_raw_actions=False,
        window_mode=str(train_cfg.data.get("window_mode", "full")),
    )
    return model, checkpoint, stage1_dataset


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        device = torch.device("cpu")

    model, checkpoint, stage1_dataset = _load_stage1(args.checkpoint, device)
    dataset = _dataset_for_suite(stage1_dataset, args.task_suite_name)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    benchmark_task_by_text = {str(task_suite.get_task(i).language): i for i in range(task_suite.n_tasks)}

    task_ids = None
    if args.task_ids:
        task_ids = {int(item) for item in args.task_ids.split(",") if item.strip()}
    episodes = _select_episodes(
        dataset,
        task_ids=task_ids,
        max_tasks=int(args.max_tasks),
        num_episodes_per_task=int(args.num_episodes_per_task),
    )

    modes: list[str]
    if args.mode == "both":
        modes = ["expert", "recon"]
    elif args.mode == "all":
        modes = ["expert", "recon", "fast"]
    else:
        modes = [args.mode]

    fast_tokenizer = None
    if "fast" in modes:
        from transformers import AutoProcessor

        fast_tokenizer = AutoProcessor.from_pretrained(args.fast_tokenizer_name, trust_remote_code=True)

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "task_suite_name": args.task_suite_name,
        "dataset_name": str(dataset.dataset_name),
        "mode": args.mode,
        "num_steps_wait": int(args.num_steps_wait),
        "gripper_mode": str(args.gripper_mode),
        "init_state_strategy": str(args.init_state_strategy),
        "max_steps": int(args.max_steps) if args.max_steps > 0 else SUITE_MAX_STEPS[args.task_suite_name],
        "episodes": [],
        "summary": {mode: {"successes": 0, "episodes": 0, "success_rate": 0.0} for mode in modes},
        "per_task": {},
    }

    max_steps = int(report["max_steps"])
    video_root = Path(args.video_out_dir) if args.save_videos else None

    env_cache: dict[int, tuple[Any, str, np.ndarray]] = {}
    init_match_image_cache: dict[int, list[np.ndarray]] = {}
    try:
        for episode in tqdm(episodes, desc="oracle replay"):
            task_text = str(episode["task_description"])
            if task_text not in benchmark_task_by_text:
                raise KeyError(f"Task text from dataset not found in benchmark suite: {task_text!r}")
            benchmark_task_id = benchmark_task_by_text[task_text]
            if benchmark_task_id not in env_cache:
                task = task_suite.get_task(benchmark_task_id)
                env, task_description = _get_libero_env(task, int(args.seed))
                init_states = task_suite.get_task_init_states(benchmark_task_id)
                env_cache[benchmark_task_id] = (env, task_description, init_states)
            env, _, init_states = env_cache[benchmark_task_id]
            init_state_index = int(episode["init_state_index"])
            init_state_match_mse = None
            hdf5_match = None
            expert_actions = _get_expert_actions(dataset, int(episode["trajectory_id"]))
            if args.init_state_strategy == "image":
                if benchmark_task_id not in init_match_image_cache:
                    init_match_image_cache[benchmark_task_id] = _render_init_state_match_images(
                        env=env,
                        init_states=init_states,
                        num_steps_wait=int(args.num_steps_wait),
                    )
                init_state_index, init_state_match_mse = _match_init_state_by_first_frame(
                    candidate_images=init_match_image_cache[benchmark_task_id],
                    dataset=dataset,
                    trajectory_id=int(episode["trajectory_id"]),
                )
            if args.init_state_strategy == "hdf5_action":
                hdf5_match = _match_hdf5_demo_by_actions(
                    suite_name=args.task_suite_name,
                    task_text=task_text,
                    expert_actions=expert_actions,
                )
                if args.hdf5_reset_model_xml:
                    _reset_env_from_hdf5_model_file(env, hdf5_match["model_file"])
                init_state = hdf5_match["init_state"]
            else:
                if init_state_index >= len(init_states):
                    raise IndexError(
                        f"Episode ordinal {init_state_index} exceeds init_states for task {task_text!r}."
                    )
                init_state = init_states[init_state_index]

            if init_state_index >= len(init_states) and args.init_state_strategy != "hdf5_action":
                raise IndexError(
                    f"Episode ordinal {init_state_index} exceeds init_states for task {task_text!r}."
                )

            actions_by_mode = {}
            if "expert" in modes:
                actions_by_mode["expert"] = expert_actions
            if "recon" in modes:
                actions_by_mode["recon"] = _get_reconstructed_actions(
                    stage1_dataset=stage1_dataset,
                    dataset=dataset,
                    trajectory_id=int(episode["trajectory_id"]),
                    trajectory_length=int(episode["trajectory_length"]),
                    model=model,
                    device=device,
                )
            if "fast" in modes:
                actions_by_mode["fast"] = _get_fast_reconstructed_actions(
                    stage1_dataset=stage1_dataset,
                    dataset=dataset,
                    trajectory_id=int(episode["trajectory_id"]),
                    trajectory_length=int(episode["trajectory_length"]),
                    fast_tokenizer=fast_tokenizer,
                )

            episode_result = dict(episode)
            episode_result["benchmark_task_id"] = int(benchmark_task_id)
            episode_result["resolved_init_state_index"] = int(init_state_index)
            if init_state_match_mse is not None:
                episode_result["init_state_match_mse"] = float(init_state_match_mse)
            if hdf5_match is not None:
                episode_result["hdf5_match"] = {
                    key: value
                    for key, value in hdf5_match.items()
                    if key not in {"init_state", "model_file"}
                }
            episode_result["results"] = {}
            for mode in modes:
                video_path = None
                if video_root is not None:
                    suffix = f"task{benchmark_task_id:02d}_ep{init_state_index:03d}_{mode}.mp4"
                    video_path = video_root / args.task_suite_name / suffix
                result = _run_episode(
                    env=env,
                    init_state=init_state,
                    actions=actions_by_mode[mode],
                    max_steps=max_steps,
                    num_steps_wait=int(args.num_steps_wait),
                    gripper_mode=str(args.gripper_mode),
                    video_path=video_path,
                    reset_before_init=not bool(hdf5_match is not None and args.hdf5_reset_model_xml),
                )
                episode_result["results"][mode] = result
                report["summary"][mode]["episodes"] += 1
                report["summary"][mode]["successes"] += int(result["success"])

                task_bucket = report["per_task"].setdefault(
                    task_text,
                    {m: {"successes": 0, "episodes": 0, "success_rate": 0.0} for m in modes},
                )
                task_bucket[mode]["episodes"] += 1
                task_bucket[mode]["successes"] += int(result["success"])

            report["episodes"].append(episode_result)
            for mode in modes:
                item = report["summary"][mode]
                item["success_rate"] = item["successes"] / max(item["episodes"], 1)

        for task_bucket in report["per_task"].values():
            for mode in modes:
                item = task_bucket[mode]
                item["success_rate"] = item["successes"] / max(item["episodes"], 1)
    finally:
        for env, _, _ in env_cache.values():
            env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VAR Stage 1 tokenizer by LIBERO oracle replay.")
    parser.add_argument("--checkpoint", type=Path, default=Path("playground/Checkpoints/var_stage1_pi05_libero/best_recon.ckpt"))
    parser.add_argument("--output", type=Path, default=Path("playground/Checkpoints/var_stage1_pi05_libero/oracle_replay_eval.json"))
    parser.add_argument("--task_suite_name", type=str, default="libero_goal", choices=sorted(SUITE_TO_DATASET_PREFIX))
    parser.add_argument("--mode", type=str, default="both", choices=["expert", "recon", "fast", "both", "all"])
    parser.add_argument("--task_ids", type=str, default="", help="Comma-separated LeRobot/LIBERO task indices.")
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--num_episodes_per_task", type=int, default=1)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gripper_mode", type=str, default="open01", choices=["open01", "close01", "raw"])
    parser.add_argument("--init_state_strategy", type=str, default="ordinal", choices=["ordinal", "image", "hdf5_action"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fast_tokenizer_name", type=str, default="physical-intelligence/fast")
    parser.add_argument("--hdf5_reset_model_xml", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--video_out_dir", type=Path, default=Path("playground/Checkpoints/var_stage1_pi05_libero/oracle_replay_videos"))
    args = parser.parse_args()

    report = evaluate(args)
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
