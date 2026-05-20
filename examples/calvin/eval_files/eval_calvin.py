"""
Calvin Multi-Step Evaluation Script

Based on RoboFlamingo's evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

Evaluates a policy server on Calvin's long-horizon multi-task benchmark.
Measures success rate on chains of 1-5 consecutive tasks.

Usage:
    python examples/calvin/eval_calvin.py \
        --args.host 0.0.0.0 \
        --args.port 8000 \
        --args.dataset_path /path/to/calvin/task_D_D \
        --args.num_sequences 1000
"""

import copy
import contextlib
import dataclasses
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from math import pi
from pathlib import Path

import hydra
import numpy as np
import tyro

try:
    from moviepy.editor import ImageSequenceClip
except ModuleNotFoundError:
    ImageSequenceClip = None
from omegaconf import OmegaConf
try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *_args, **_kwargs):
        return text
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, *_args, **_kwargs):
        return iterable

from deployment.model_server.tools import image_tools
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# from calvin_env.envs.play_table_env import get_env

# Set OpenGL platform for headless rendering
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EP_LEN = 360  # Max steps per task
ACTION_DIM_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
TASK_OBJECT_TOKENS = {
    "red",
    "blue",
    "pink",
    "block",
    "drawer",
    "slider",
    "sliding",
    "door",
    "lightbulb",
    "led",
    "switch",
}
TASK_CATEGORY_PREFIXES = [
    "push_into",
    "rotate",
    "push",
    "lift",
    "place",
    "stack",
    "unstack",
    "move",
    "open",
    "close",
    "turn",
    "slide",
    "toggle",
    "pick",
    "grasp",
]


def collect_plan(model, plans, subtask):
    try:
        plans[subtask].append((model.plan.cpu(), model.latent_goal.cpu()))
    except AttributeError:
        return


def count_success(results):
    if not results:
        return [0.0] * 5
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        step_success.append(n_success / len(results))
    return step_success


def get_log_dir(log_dir):
    if log_dir is None:
        log_dir = Path("/tmp/evaluation")
    else:
        log_dir = Path(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _debug_status_limit(status):
    general = _env_int("CALVIN_DEBUG_MAX_GIFS_PER_TASK", -1)
    if status == "success":
        return _env_int("CALVIN_DEBUG_MAX_SUCCESS_GIFS_PER_TASK", general)
    if status == "fail":
        return _env_int("CALVIN_DEBUG_MAX_FAIL_GIFS_PER_TASK", general)
    return general


def _debug_slug(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text[:120] or "task"


def calvin_robot_obs_to_lerobot_state(obs):
    """Map CALVIN env robot_obs to the 8-D LeRobot CALVIN state layout.

    LeRobot CALVIN names the state vector as:
    x, y, z, roll, pitch, yaw, pad, gripper. In the official CALVIN env
    `robot_obs[:8]` corresponds to EEF pose plus the two gripper-state values.
    """
    robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32).reshape(-1)
    if robot_obs.size < 8:
        raise ValueError(f"Expected robot_obs with at least 8 values, got {robot_obs.shape}")
    return robot_obs[:8][None, :].astype(np.float32, copy=False)


class CalvinStatePerturber:
    """Deterministic eval-time state perturbations for state-path sanity checks."""

    def __init__(self, mode="normal", buffer_size=32, seed=0):
        self.mode = str(mode or "normal").lower().replace("-", "_")
        self.buffer_size = max(1, int(buffer_size))
        self.rng = np.random.default_rng(int(seed))
        self.buffer = []
        allowed = {"normal", "none", "off", "zero", "zeros", "shuffle", "temporal_shuffle"}
        if self.mode not in allowed:
            raise ValueError(f"Unsupported CALVIN state sanity mode `{mode}`. Expected one of {sorted(allowed)}")

    @property
    def enabled(self):
        return self.mode not in {"normal", "none", "off"}

    def reset(self):
        self.buffer.clear()

    def __call__(self, state):
        state = np.asarray(state, dtype=np.float32)
        if self.mode in {"normal", "none", "off"}:
            return state
        if self.mode in {"zero", "zeros"}:
            return np.zeros_like(state, dtype=np.float32)

        current = state.copy()
        if self.buffer:
            idx = int(self.rng.integers(0, len(self.buffer)))
            perturbed = self.buffer[idx].copy()
        else:
            perturbed = np.zeros_like(state, dtype=np.float32)
        self.buffer.append(current)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        return perturbed.astype(np.float32, copy=False)


def _debug_gif_root(eval_log_dir):
    return Path(os.environ.get("CALVIN_DEBUG_GIF_ROOT", eval_log_dir))


def _debug_counter_dir(eval_log_dir):
    default_dir = _debug_gif_root(eval_log_dir) / ".gif_counts"
    return Path(os.environ.get("CALVIN_DEBUG_GIF_COUNTER_DIR", default_dir))


def _debug_counter_value(eval_log_dir, subtask, status):
    count_path = _debug_counter_dir(eval_log_dir) / f"{_debug_slug(subtask)}.{status}.count"
    try:
        return int(count_path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _debug_should_collect_frames(eval_log_dir, subtask):
    if ImageSequenceClip is None:
        return False
    for status in ("success", "fail"):
        limit = _debug_status_limit(status)
        if limit < 0:
            return True
        if limit > _debug_counter_value(eval_log_dir, subtask, status):
            return True
    return False


@contextlib.contextmanager
def _debug_counter_lock(counter_dir, key):
    counter_dir.mkdir(parents=True, exist_ok=True)
    lock_path = counter_dir / f"{key}.lock"
    start = time.time()
    acquired = False
    while time.time() - start < 30:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.05)
    if not acquired:
        yield False
        return
    try:
        yield True
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def _debug_reserve_gif_path(eval_log_dir, subtask, status, sequence_i, subtask_i):
    limit = _debug_status_limit(status)
    if limit == 0:
        return None

    root = _debug_gif_root(eval_log_dir)
    root.mkdir(parents=True, exist_ok=True)
    task_slug = _debug_slug(subtask)
    status_slug = "succ" if status == "success" else "fail"

    if limit < 0:
        filename = f"{status_slug}__{task_slug}__seq{sequence_i}_sub{subtask_i}.gif"
        return root / filename

    counter_dir = _debug_counter_dir(eval_log_dir)
    key = f"{task_slug}.{status}"
    count_path = counter_dir / f"{key}.count"
    with _debug_counter_lock(counter_dir, key) as locked:
        if not locked:
            return None
        try:
            count = int(count_path.read_text().strip())
        except (FileNotFoundError, ValueError):
            count = 0
        if count >= limit:
            return None
        count += 1
        count_path.write_text(str(count))

    filename = f"{status_slug}__{task_slug}__{count:02d}__seq{sequence_i}_sub{subtask_i}.gif"
    return root / filename


def _write_debug_gif(img_queue, eval_log_dir, subtask, status, sequence_i, subtask_i, metadata):
    if not img_queue or ImageSequenceClip is None:
        return None
    target = _debug_reserve_gif_path(eval_log_dir, subtask, status, sequence_i, subtask_i)
    if target is None:
        return None
    try:
        img_clip = ImageSequenceClip(img_queue, fps=30)
        img_clip.write_gif(str(target), fps=30)
        target.with_suffix(".json").write_text(json.dumps(_json_safe(metadata), sort_keys=True, indent=2))
        print(f"debug gif saved: {target}")
        return str(target)
    except Exception as exc:
        print(f"failed to write debug gif for {subtask}: {exc}")
        return None


def print_and_save(results, sequences, log_dir, epoch=None):
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = float(np.mean(results)) if results else 0.0
    chain_sr = {i + 1: float(sr) for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()
    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            cnt_fail[sequence[result]] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": int(cnt_success[task]), "total": int(total[task])}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    current_data = {
        epoch: {
            "avg_seq_len": avg_seq_len,
            "chain_sr": chain_sr,
            "task_info": task_info,
            "results": [int(x) for x in results],
            "num_sequences": len(results),
        }
    }
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )


def _fnv1_32(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value = (value * 16777619) & 0xFFFFFFFF
        value ^= byte
    return value


def _json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value


def _basic_stats(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _parse_action_limits():
    raw_limits = os.environ.get("CALVIN_ACTION_SATURATION_LIMITS", "1,1,1,1,1,1,1")
    try:
        limits = np.asarray([float(item.strip()) for item in raw_limits.split(",")], dtype=np.float64)
    except ValueError:
        limits = np.ones(len(ACTION_DIM_NAMES), dtype=np.float64)
    if limits.size != len(ACTION_DIM_NAMES):
        limits = np.ones(len(ACTION_DIM_NAMES), dtype=np.float64)
    return np.maximum(np.abs(limits), 1e-12)


def _new_action_accumulator():
    dim = len(ACTION_DIM_NAMES)
    return {
        "count": 0,
        "sum_abs": np.zeros(dim, dtype=np.float64),
        "sum_sq": np.zeros(dim, dtype=np.float64),
        "max_abs": np.zeros(dim, dtype=np.float64),
        "saturation_count": np.zeros(dim, dtype=np.float64),
        "limits": _parse_action_limits(),
        "prev": None,
        "jitter_count": 0,
        "jitter_sum_abs": np.zeros(dim, dtype=np.float64),
        "jitter_max_abs": np.zeros(dim, dtype=np.float64),
        "jitter_sum_l2": 0.0,
        "jitter_max_l2": 0.0,
        "gripper_switches": 0,
    }


def _action_vector(action):
    array = np.asarray(action, dtype=np.float64).reshape(-1)
    if array.size == len(ACTION_DIM_NAMES):
        return array
    vector = np.zeros(len(ACTION_DIM_NAMES), dtype=np.float64)
    keep = min(array.size, len(ACTION_DIM_NAMES))
    if keep > 0:
        vector[:keep] = array[:keep]
    return vector


def _update_action_accumulator(accumulator, action):
    vector = _action_vector(action)
    abs_vector = np.abs(vector)
    accumulator["count"] += 1
    accumulator["sum_abs"] += abs_vector
    accumulator["sum_sq"] += vector * vector
    accumulator["max_abs"] = np.maximum(accumulator["max_abs"], abs_vector)
    accumulator["saturation_count"] += (abs_vector >= accumulator["limits"]).astype(np.float64)

    prev = accumulator["prev"]
    if prev is not None:
        diff = np.abs(vector - prev)
        diff_l2 = float(np.linalg.norm(vector - prev))
        accumulator["jitter_count"] += 1
        accumulator["jitter_sum_abs"] += diff
        accumulator["jitter_max_abs"] = np.maximum(accumulator["jitter_max_abs"], diff)
        accumulator["jitter_sum_l2"] += diff_l2
        accumulator["jitter_max_l2"] = max(accumulator["jitter_max_l2"], diff_l2)
        if np.sign(vector[-1]) != np.sign(prev[-1]):
            accumulator["gripper_switches"] += 1
    accumulator["prev"] = vector


def _finalize_action_accumulator(accumulator):
    count = int(accumulator["count"])
    dim = len(ACTION_DIM_NAMES)
    if count == 0:
        zeros = [0.0] * dim
        return {
            "count": 0,
            "dim_names": ACTION_DIM_NAMES,
            "saturation_limits": accumulator["limits"].tolist(),
            "mean_abs": zeros,
            "rms": zeros,
            "max_abs": zeros,
            "saturation_rate": zeros,
            "_sum_abs": zeros,
            "_sum_sq": zeros,
            "_saturation_count": zeros,
            "jitter": {
                "count": 0,
                "mean_abs": zeros,
                "max_abs": zeros,
                "mean_l2": 0.0,
                "max_l2": 0.0,
                "gripper_switches": 0,
                "gripper_switch_rate": 0.0,
                "_sum_abs": zeros,
                "_sum_l2": 0.0,
            },
        }

    jitter_count = int(accumulator["jitter_count"])
    if jitter_count:
        jitter_mean_abs = accumulator["jitter_sum_abs"] / jitter_count
        jitter_mean_l2 = accumulator["jitter_sum_l2"] / jitter_count
        gripper_switch_rate = accumulator["gripper_switches"] / jitter_count
    else:
        jitter_mean_abs = np.zeros(dim, dtype=np.float64)
        jitter_mean_l2 = 0.0
        gripper_switch_rate = 0.0

    return {
        "count": count,
        "dim_names": ACTION_DIM_NAMES,
        "saturation_limits": accumulator["limits"].tolist(),
        "mean_abs": (accumulator["sum_abs"] / count).tolist(),
        "rms": np.sqrt(accumulator["sum_sq"] / count).tolist(),
        "max_abs": accumulator["max_abs"].tolist(),
        "saturation_rate": (accumulator["saturation_count"] / count).tolist(),
        "_sum_abs": accumulator["sum_abs"].tolist(),
        "_sum_sq": accumulator["sum_sq"].tolist(),
        "_saturation_count": accumulator["saturation_count"].tolist(),
        "jitter": {
            "count": jitter_count,
            "mean_abs": jitter_mean_abs.tolist(),
            "max_abs": accumulator["jitter_max_abs"].tolist(),
            "mean_l2": float(jitter_mean_l2),
            "max_l2": float(accumulator["jitter_max_l2"]),
            "gripper_switches": int(accumulator["gripper_switches"]),
            "gripper_switch_rate": float(gripper_switch_rate),
            "_sum_abs": accumulator["jitter_sum_abs"].tolist(),
            "_sum_l2": float(accumulator["jitter_sum_l2"]),
        },
    }


def _merge_action_stats(stats_list):
    stats_list = [stats for stats in stats_list if stats and int(stats.get("count", 0)) > 0]
    dim = len(ACTION_DIM_NAMES)
    if not stats_list:
        return _finalize_action_accumulator(_new_action_accumulator())

    total_count = sum(int(stats["count"]) for stats in stats_list)
    sum_abs = np.zeros(dim, dtype=np.float64)
    sum_sq = np.zeros(dim, dtype=np.float64)
    max_abs = np.zeros(dim, dtype=np.float64)
    saturation_count = np.zeros(dim, dtype=np.float64)
    limits = np.asarray(stats_list[0].get("saturation_limits", [1.0] * dim), dtype=np.float64)
    jitter_count = 0
    jitter_sum_abs = np.zeros(dim, dtype=np.float64)
    jitter_max_abs = np.zeros(dim, dtype=np.float64)
    jitter_sum_l2 = 0.0
    jitter_max_l2 = 0.0
    gripper_switches = 0

    for stats in stats_list:
        count = int(stats["count"])
        sum_abs += np.asarray(stats.get("_sum_abs", np.asarray(stats["mean_abs"]) * count), dtype=np.float64)
        sum_sq += np.asarray(stats.get("_sum_sq", np.square(stats["rms"]) * count), dtype=np.float64)
        max_abs = np.maximum(max_abs, np.asarray(stats["max_abs"], dtype=np.float64))
        saturation_count += np.asarray(
            stats.get("_saturation_count", np.asarray(stats["saturation_rate"]) * count),
            dtype=np.float64,
        )
        jitter = stats.get("jitter", {})
        current_jitter_count = int(jitter.get("count", 0))
        jitter_count += current_jitter_count
        jitter_sum_abs += np.asarray(
            jitter.get("_sum_abs", np.asarray(jitter.get("mean_abs", [0.0] * dim)) * current_jitter_count),
            dtype=np.float64,
        )
        jitter_max_abs = np.maximum(jitter_max_abs, np.asarray(jitter.get("max_abs", [0.0] * dim), dtype=np.float64))
        jitter_sum_l2 += float(jitter.get("_sum_l2", float(jitter.get("mean_l2", 0.0)) * current_jitter_count))
        jitter_max_l2 = max(jitter_max_l2, float(jitter.get("max_l2", 0.0)))
        gripper_switches += int(jitter.get("gripper_switches", 0))

    mean_abs = sum_abs / total_count
    rms = np.sqrt(sum_sq / total_count)
    saturation_rate = saturation_count / total_count
    if jitter_count:
        jitter_mean_abs = jitter_sum_abs / jitter_count
        jitter_mean_l2 = jitter_sum_l2 / jitter_count
        gripper_switch_rate = gripper_switches / jitter_count
    else:
        jitter_mean_abs = np.zeros(dim, dtype=np.float64)
        jitter_mean_l2 = 0.0
        gripper_switch_rate = 0.0

    return {
        "count": int(total_count),
        "dim_names": ACTION_DIM_NAMES,
        "saturation_limits": limits.tolist(),
        "mean_abs": mean_abs.tolist(),
        "rms": rms.tolist(),
        "max_abs": max_abs.tolist(),
        "saturation_rate": saturation_rate.tolist(),
        "_sum_abs": sum_abs.tolist(),
        "_sum_sq": sum_sq.tolist(),
        "_saturation_count": saturation_count.tolist(),
        "jitter": {
            "count": int(jitter_count),
            "mean_abs": jitter_mean_abs.tolist(),
            "max_abs": jitter_max_abs.tolist(),
            "mean_l2": float(jitter_mean_l2),
            "max_l2": float(jitter_max_l2),
            "gripper_switches": int(gripper_switches),
            "gripper_switch_rate": float(gripper_switch_rate),
            "_sum_abs": jitter_sum_abs.tolist(),
            "_sum_l2": float(jitter_sum_l2),
        },
    }


def _task_set(task_info):
    if not task_info:
        return set()
    if isinstance(task_info, dict):
        return {str(key) for key in task_info.keys()}
    return {str(item) for item in task_info}


def _task_tokens(task):
    return {token for token in str(task).replace("-", "_").split("_") if token}


def _task_category(task):
    task = str(task)
    for prefix in TASK_CATEGORY_PREFIXES:
        if task.startswith(prefix):
            return prefix
    tokens = _task_tokens(task)
    for prefix in TASK_CATEGORY_PREFIXES:
        if prefix in tokens:
            return prefix
    return "other"


def _task_objects(task):
    tokens = _task_tokens(task)
    return sorted(tokens.intersection(TASK_OBJECT_TOKENS))


def _is_related_task(target_task, achieved_task):
    if target_task == achieved_task:
        return True
    if _task_category(target_task) == _task_category(achieved_task):
        return True
    return bool(set(_task_objects(target_task)).intersection(_task_objects(achieved_task)))


def summarize_eval_metrics(sequence_records):
    results = [int(record.get("success_count", 0)) for record in sequence_records]
    sequence_count = len(sequence_records)
    success_len_hist = {str(i): int(Counter(results).get(i, 0)) for i in range(6)}
    conditional = {}
    for position in range(1, 6):
        attempts = sum(
            1
            for record in sequence_records
            if len(record.get("tasks", [])) >= position and int(record.get("success_count", 0)) >= position - 1
        )
        successes = sum(1 for record in sequence_records if int(record.get("success_count", 0)) >= position)
        conditional[str(position)] = {
            "attempts": int(attempts),
            "successes": int(successes),
            "success_rate": float(successes / attempts) if attempts else None,
        }

    failure_position = Counter()
    failure_steps = []
    all_subtasks = []
    chain_accumulator = defaultdict(lambda: {"attempts": 0, "full_successes": 0, "success_len_sum": 0})
    per_task = defaultdict(
        lambda: {
            "attempts": 0,
            "successes": 0,
            "success_steps": [],
            "failure_steps": [],
            "near_miss_any_task": 0,
            "near_miss_related_task": 0,
            "achieved_other_tasks": Counter(),
        }
    )

    for record in sequence_records:
        tasks = [str(task) for task in record.get("tasks", [])]
        success_count = int(record.get("success_count", 0))
        chain_key = " -> ".join(tasks)
        chain_accumulator[chain_key]["attempts"] += 1
        chain_accumulator[chain_key]["full_successes"] += int(success_count >= len(tasks))
        chain_accumulator[chain_key]["success_len_sum"] += success_count

        failed_position = record.get("failed_subtask_position")
        failure_position[str(failed_position) if failed_position is not None else "complete"] += 1
        if record.get("failure_step") is not None:
            failure_steps.append(record["failure_step"])

        for subtask_record in record.get("subtasks", []):
            all_subtasks.append(subtask_record)
            task = str(subtask_record.get("task"))
            task_stats = per_task[task]
            task_stats["attempts"] += 1
            if subtask_record.get("success"):
                task_stats["successes"] += 1
                if subtask_record.get("success_step") is not None:
                    task_stats["success_steps"].append(subtask_record["success_step"])
            else:
                if subtask_record.get("failure_step") is not None:
                    task_stats["failure_steps"].append(subtask_record["failure_step"])
                task_stats["near_miss_any_task"] += int(bool(subtask_record.get("near_miss_any_task", False)))
                task_stats["near_miss_related_task"] += int(bool(subtask_record.get("near_miss_related_task", False)))
            for achieved_task in subtask_record.get("achieved_other_tasks", []):
                task_stats["achieved_other_tasks"][achieved_task] += 1

    failed_subtasks = [record for record in all_subtasks if not record.get("success")]
    near_miss_any = sum(int(bool(record.get("near_miss_any_task", False))) for record in failed_subtasks)
    near_miss_related = sum(int(bool(record.get("near_miss_related_task", False))) for record in failed_subtasks)

    per_task_summary = {}
    for task, stats in sorted(per_task.items()):
        attempts = int(stats["attempts"])
        successes = int(stats["successes"])
        failures = attempts - successes
        per_task_summary[task] = {
            "attempts": attempts,
            "successes": successes,
            "failures": failures,
            "success_rate": float(successes / attempts) if attempts else None,
            "success_step": _basic_stats(stats["success_steps"]),
            "failure_step": _basic_stats(stats["failure_steps"]),
            "near_miss_any_task_rate": float(stats["near_miss_any_task"] / failures) if failures else None,
            "near_miss_related_task_rate": float(stats["near_miss_related_task"] / failures) if failures else None,
            "achieved_other_tasks": {
                task_name: int(count) for task_name, count in stats["achieved_other_tasks"].most_common()
            },
        }

    chain_summary = {}
    for chain, stats in sorted(chain_accumulator.items()):
        attempts = int(stats["attempts"])
        chain_summary[chain] = {
            "attempts": attempts,
            "full_successes": int(stats["full_successes"]),
            "full_success_rate": float(stats["full_successes"] / attempts) if attempts else None,
            "avg_success_len": float(stats["success_len_sum"] / attempts) if attempts else None,
        }

    raw_action_stats = _merge_action_stats([record.get("raw_action_stats") for record in all_subtasks])
    env_action_stats = _merge_action_stats([record.get("env_action_stats") for record in all_subtasks])

    return {
        "num_sequences": int(sequence_count),
        "avg_seq_len": float(np.mean(results)) if results else 0.0,
        "chain_sr": {str(i + 1): float(sr) for i, sr in enumerate(count_success(results))},
        "success_len_histogram": success_len_hist,
        "conditional_success": conditional,
        "failure_position": {str(key): int(value) for key, value in sorted(failure_position.items())},
        "failure_step": _basic_stats(failure_steps),
        "near_miss": {
            "failed_subtasks": int(len(failed_subtasks)),
            "any_task_count": int(near_miss_any),
            "any_task_rate": float(near_miss_any / len(failed_subtasks)) if failed_subtasks else None,
            "related_task_count": int(near_miss_related),
            "related_task_rate": float(near_miss_related / len(failed_subtasks)) if failed_subtasks else None,
        },
        "per_atomic_task": per_task_summary,
        "task_chain": chain_summary,
        "action_stats": {
            "raw_model_action": raw_action_stats,
            "env_action_after_gripper_binarization": env_action_stats,
        },
    }


def write_metrics(sequence_records, log_dir, epoch=None):
    log_dir = Path(log_dir)
    epoch_key = str(epoch)
    records_name = f"metrics_sequences_epoch_{epoch_key}.jsonl"
    summary = summarize_eval_metrics(sequence_records)
    summary["sequence_records_path"] = records_name

    with open(log_dir / records_name, "w") as file:
        for record in sequence_records:
            file.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")

    current_data = {epoch_key: _json_safe(summary)}
    previous_data = {}
    try:
        with open(log_dir / "metrics.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    with open(log_dir / "metrics.json", "w") as file:
        json.dump({**previous_data, **current_data}, file, indent=2, sort_keys=True)
    results_path = log_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r") as file:
            results_data = json.load(file)
        epoch_results = results_data.setdefault(epoch_key, {})
        near_miss = summary.get("near_miss", {})
        epoch_results["near_miss"] = near_miss
        epoch_results["near_miss_rate"] = near_miss.get("any_task_rate")
        epoch_results["near_miss_related_rate"] = near_miss.get("related_task_rate")
        with open(results_path, "w") as file:
            json.dump(results_data, file, indent=2, sort_keys=True)
    print(f"Detailed metrics saved to {log_dir / 'metrics.json'}")


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]
    with temp_seed(_fnv1_32(str(initial_condition.values()))):
        np.random.shuffle(block_table)
        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]
        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)
        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)
        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)
    return robot_obs, scene_obs


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    pretrained_path: str = ""
    unnorm_key: str = ""

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = "/path/to/calvin/task_D_D"  # Path to Calvin dataset
    calvin_config_path: str = "/path/to/calvin/calvin_models/conf"
    eval_sequences_path: str = "/path/to/calvin/eval_sequences.json"
    num_sequences: int = 1000  # Number of evaluation sequences
    sequence_start: int = 0  # First eval sequence index for sharded evaluation
    sequence_stride: int = 1  # Stride between eval sequence indices for sharded evaluation
    num_workers: int = 1  # For future multi-process support
    seed: int = 0
    create_plan_tsne: bool = False

    #################################################################################################################
    # Evaluation settings
    #################################################################################################################
    debug: bool = False  # Save debug videos
    send_state: bool = False  # Send 8-D robot proprioception to state-aware checkpoints
    state_mode: str = "normal"  # normal | zero | shuffle, for state-path sanity checks
    state_shuffle_buffer: int = 32  # temporal buffer size used by state_mode=shuffle
    eval_log_dir: str = "tmp/calvin/eval_logs"  # Path to save evaluation logs and videos
    reset: bool = False  # If True, reset robot state between tasks (easier)
    diverse_inst: bool = False  # Use diverse instructions (zero-shot generalization)


class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""

    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 224,
        replan_steps: int = 5,
        pretrained_path: str = "",
        unnorm_key: str = "",
        send_state: bool = False,
        state_mode: str = "normal",
        state_shuffle_buffer: int = 32,
    ):
        self.send_state = send_state
        self.state_perturber = CalvinStatePerturber(
            mode=state_mode,
            buffer_size=state_shuffle_buffer,
            seed=int(os.environ.get("CALVIN_STATE_SHUFFLE_SEED", "0")),
        )
        self.client = ModelClient(
            unnorm_key=(unnorm_key or None),
            policy_setup="franka",
            horizon=0,
            action_ensemble=False,
            host=host,
            port=port,
        )
        server_meta = getattr(self.client, "_server_metadata", {})
        model_state_dim = int(server_meta.get("model_state_dim") or 0)
        if self.send_state and model_state_dim != 8:
            raise ValueError(
                "CALVIN_SEND_STATE requires a state-aware checkpoint with "
                f"model_state_dim=8, but server metadata reports model_state_dim={model_state_dim}. "
                "Use the state8 training config/checkpoint or disable CALVIN_SEND_STATE."
            )
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.step_count = 0

    def reset(self):
        """Reset action plan buffer."""
        self.step_count = 0
        self.state_perturber.reset()

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        """
        Query policy for action given observation and language instruction.

        Args:
            obs: Calvin observation dict with keys:
                - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
                - robot_obs: (15,) proprioceptive state [ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]
            lang_annotation: Natural language task description
            get_action: If True, query model for new action chunk

        Returns:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Preprocess images
        rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # (84, 84, 3) uint8

        # Resize and pad images
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size))
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        )

        # Prepare input for policy server (aligned with eval_libero)
        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
        }
        if self.send_state:
            example["state"] = self.state_perturber(calvin_robot_obs_to_lerobot_state(obs))

        # Query model
        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)

        action = np.concatenate([world_vector, rotation_delta, open_gripper], axis=0).astype(np.float32)
        self.step_count += 1
        return action


def make_env(dataset_path: str):
    """Initialize Calvin environment without tactile sensor (to avoid OpenGL issues)."""
    val_folder = Path(dataset_path) / "validation"

    # Load config and disable tactile sensor to avoid pyrender/OpenGL conflicts
    from omegaconf import OmegaConf

    config_path = val_folder / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(config_path)

    # Remove tactile sensor from camera list if it exists
    if hasattr(cfg.env, "cameras") and "tactile" in cfg.env.cameras:
        # Create a new camera dict without tactile
        new_cameras = OmegaConf.create({k: v for k, v in cfg.env.cameras.items() if k != "tactile"})
        cfg.env.cameras = new_cameras

    cfg.env.use_egl = os.environ.get("CALVIN_USE_EGL", "0") == "1"

    # Initialize environment with modified config
    import hydra

    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    return env


def load_lang_task(dataset_path: str) -> dict:
    """Load language annotations and task oracle for Calvin validation set."""
    conf_dir = Path(dataset_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return val_annotations, task_oracle


def evaluate_policy_ddp(
    policy,
    env,
    epoch,
    calvin_conf_path,
    eval_sequences_path,
    num_sequences,
    eval_log_dir=None,
    debug=False,
    create_plan_tsne=False,
    reset=False,
    diverse_inst=False,
    sequence_start=0,
    sequence_stride=1,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open("/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        all_eval_sequences = json.load(f)
    if sequence_start < 0:
        raise ValueError(f"sequence_start must be >= 0, got {sequence_start}")
    if sequence_stride < 1:
        raise ValueError(f"sequence_stride must be >= 1, got {sequence_stride}")
    sequence_entries = list(enumerate(all_eval_sequences))[sequence_start::sequence_stride]
    if num_sequences > 0:
        sequence_entries = sequence_entries[:num_sequences]
    print(
        f"Evaluating {len(sequence_entries)} sequences "
        f"(start={sequence_start}, stride={sequence_stride}, requested={num_sequences})"
    )
    results = []
    sequence_records = []
    plans = defaultdict(list)

    if not debug:
        sequence_iter = tqdm(sequence_entries, position=0, leave=True)
    else:
        sequence_iter = sequence_entries

    for original_sequence_i, (initial_state, eval_sequence) in sequence_iter:
        result = evaluate_sequence(
            env,
            policy,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
            eval_log_dir,
            original_sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
        )
        sequence_records.append(result)
        results.append(int(result["success_count"]))
        if not debug:
            sequence_iter.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    # if create_plan_tsne:
    #     create_tsne(plans, eval_log_dir, epoch)

    eval_sequences = [item for _, item in sequence_entries]
    print_and_save(results, eval_sequences, eval_log_dir, epoch)
    write_metrics(sequence_records, eval_log_dir, epoch)

    return results


def evaluate_sequence(
    env,
    policy,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    sequence_i=-1,
    reset=False,
    diverse_inst=False,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    sequence_record = {
        "sequence_index": int(sequence_i),
        "tasks": [str(task) for task in eval_sequence],
        "initial_state": _json_safe(initial_state),
        "subtasks": [],
        "success_count": 0,
        "failed_task": None,
        "failed_subtask_position": None,
        "failure_step": None,
        "completed": False,
    }
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            rollout_result = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                diverse_inst=diverse_inst,
            )
        else:
            rollout_result = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                diverse_inst=diverse_inst,
            )
        rollout_result["task"] = str(subtask)
        rollout_result["subtask_position"] = int(subtask_i + 1)
        sequence_record["subtasks"].append(rollout_result)
        if rollout_result["success"]:
            success_counter += 1
            sequence_record["success_count"] = int(success_counter)
        else:
            sequence_record["failed_task"] = str(subtask)
            sequence_record["failed_subtask_position"] = int(subtask_i + 1)
            sequence_record["failure_step"] = rollout_result.get("failure_step")
            return sequence_record
    sequence_record["completed"] = True
    return sequence_record


def rollout(
    env,
    policy,
    task_oracle,
    subtask,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    subtask_i=-1,
    sequence_i=-1,
    robot_obs=None,
    scene_obs=None,
    diverse_inst=False,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0]
    if "\u2019" in lang_annotation:
        lang_annotation.replace("\u2019", "'")
    policy.reset()
    start_info = env.get_info()
    raw_action_accumulator = _new_action_accumulator()
    env_action_accumulator = _new_action_accumulator()
    achieved_other_tasks = set()
    related_near_miss = False
    first_near_miss_step = None
    near_miss_interval = max(0, int(os.environ.get("CALVIN_NEAR_MISS_CHECK_INTERVAL", "30")))

    collect_debug_frames = debug and _debug_should_collect_frames(eval_log_dir, subtask)
    if collect_debug_frames:
        img_queue = []
    else:
        img_queue = None

    for step in range(EP_LEN):

        raw_action = policy.step(obs, lang_annotation)
        _update_action_accumulator(raw_action_accumulator, raw_action)

        # Calvin mutates the action in-place, so keep raw model output separate.
        action = np.array(raw_action, dtype=np.float32, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1
        _update_action_accumulator(env_action_accumulator, action)

        obs, _, _, current_info = env.step(action)
        if collect_debug_frames:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(policy, plans, subtask)

        should_check_near_miss = near_miss_interval and (step % near_miss_interval == 0 or step == EP_LEN - 1)
        if should_check_near_miss:
            achieved_tasks = _task_set(task_oracle.get_task_info(start_info, current_info))
            other_tasks = achieved_tasks.difference({subtask})
            if other_tasks and first_near_miss_step is None:
                first_near_miss_step = step + 1
            achieved_other_tasks.update(other_tasks)
            related_near_miss = related_near_miss or any(_is_related_task(subtask, task) for task in other_tasks)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                _write_debug_gif(
                    img_queue,
                    eval_log_dir,
                    subtask,
                    "success",
                    sequence_i,
                    subtask_i,
                    {
                        "status": "success",
                        "task": str(subtask),
                        "language": lang_annotation,
                        "sequence_index": int(sequence_i),
                        "subtask_index": int(subtask_i),
                        "success_step": int(step + 1),
                    },
                )
            return {
                "language": lang_annotation,
                "success": True,
                "success_step": int(step + 1),
                "failure_step": None,
                "final_step": int(step + 1),
                "achieved_other_tasks": sorted(achieved_other_tasks),
                "near_miss_any_task": False,
                "near_miss_related_task": False,
                "near_miss_first_step": first_near_miss_step,
                "raw_action_stats": _finalize_action_accumulator(raw_action_accumulator),
                "env_action_stats": _finalize_action_accumulator(env_action_accumulator),
            }
    if debug:
        print(colored("fail", "red"), end=" ")
        _write_debug_gif(
            img_queue,
            eval_log_dir,
            subtask,
            "fail",
            sequence_i,
            subtask_i,
            {
                "status": "fail",
                "task": str(subtask),
                "language": lang_annotation,
                "sequence_index": int(sequence_i),
                "subtask_index": int(subtask_i),
                "failure_step": int(EP_LEN),
                "achieved_other_tasks": sorted(achieved_other_tasks),
                "near_miss_any_task": bool(achieved_other_tasks),
                "near_miss_related_task": bool(related_near_miss),
            },
        )
    return {
        "language": lang_annotation,
        "success": False,
        "success_step": None,
        "failure_step": int(EP_LEN),
        "final_step": int(EP_LEN),
        "achieved_other_tasks": sorted(achieved_other_tasks),
        "near_miss_any_task": bool(achieved_other_tasks),
        "near_miss_related_task": bool(related_near_miss),
        "near_miss_first_step": first_near_miss_step,
        "raw_action_stats": _finalize_action_accumulator(raw_action_accumulator),
        "env_action_stats": _finalize_action_accumulator(env_action_accumulator),
    }


def main(args: Args):
    # args = tyro.cli(Args)

    policy = CalvinPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        args.replan_steps,
        pretrained_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        send_state=args.send_state,
        state_mode=args.state_mode,
        state_shuffle_buffer=args.state_shuffle_buffer,
    )
    env = make_env(args.dataset_path)

    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.eval_log_dir,
        args.debug,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
        args.sequence_start,
        args.sequence_stride,
    )


if __name__ == "__main__":
    tyro.cli(main)
