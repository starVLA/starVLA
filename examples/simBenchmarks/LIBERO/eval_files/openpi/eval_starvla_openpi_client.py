#!/usr/bin/env python3
"""LIBERO client eval for StarVLA PI0/PI05 served over websocket."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import tqdm

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from pi_libero_common import (
    DEFAULT_LIBERO_HOME,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    canonicalize_model_name,
    canonicalize_model_source,
    get_max_steps,
    normalize_openpi_value,
    OPENPI_MODEL_SOURCE,
    postprocess_openpi_actions,
    preprocess_env_obs,
    resolve_norm_stats_source,
    STARVLA_MODEL_SOURCE,
)

log = logging.getLogger("eval_starvla_openpi_client")


class StarVLAOpenPIClient:
    def __init__(
        self,
        host: str,
        port: int,
        model_name: str,
        norm_stats: dict[str, dict[str, np.ndarray]] | None = None,
        noise_seed: int | None = None,
        assets_checkpoint: Path | None = None,
        model_source: str = OPENPI_MODEL_SOURCE,
    ):
        self.policy = WebsocketClientPolicy(host=host, port=port)
        self.model_name = canonicalize_model_name(model_name)
        self.metadata = self.policy.get_server_metadata()
        checkpoint = self.metadata.get("checkpoint")
        server_source = self.metadata.get("model_source", OPENPI_MODEL_SOURCE)
        self.model_source = canonicalize_model_source(model_source or server_source)
        self.norm_stats = norm_stats or resolve_norm_stats_source(
            self.model_name,
            checkpoint=Path(checkpoint) if checkpoint else None,
            assets_checkpoint=assets_checkpoint,
        )
        self.use_quantile_norm = self.model_name != "PI0"
        self.last_action_stats: dict[str, Any] | None = None
        self.action_horizon = int(self.metadata.get("action_horizon", 10 if self.model_name == "PI05" else 50))
        self.action_dim = int(self.metadata.get("action_dim", 32))
        self.noise_rng = None if noise_seed is None else np.random.default_rng(noise_seed)
        self.last_noise: np.ndarray | None = None
        log.info("server metadata: %s", self.metadata)
        log.info(
            "using %s norm for %s with model_source=%s",
            "quantile" if self.use_quantile_norm else "zscore",
            self.model_name,
            self.model_source,
        )

    def predict_env_actions(self, example: dict[str, Any]) -> np.ndarray:
        raw_state = example["raw_state"]
        model_input = {
            "image": example["image"],
            "lang": example["lang"],
            "state": normalize_openpi_value(
                raw_state,
                self.norm_stats["state"],
                self.use_quantile_norm,
            ).astype(np.float32)[None],
        }
        payload = {"examples": [model_input]}
        self.last_noise = None
        if self.noise_rng is not None:
            self.last_noise = self.noise_rng.standard_normal(
                (self.action_horizon, self.action_dim),
                dtype=np.float32,
            )
            payload["noise"] = self.last_noise
        response = self.policy.predict_action(payload)
        if not response.get("ok", False):
            raise RuntimeError(f"policy server error: {response}")
        normalized_actions = np.asarray(response["data"]["normalized_actions"][0], dtype=np.float32)
        actions = postprocess_openpi_actions(
            normalized_actions=normalized_actions,
            raw_state=raw_state,
            norm_stats=self.norm_stats,
            model_name=self.model_name,
            model_source=self.model_source,
        )
        self.last_action_stats = {
            "normalized_actions": normalized_actions,
            "normalized_min": float(normalized_actions.min()),
            "normalized_max": float(normalized_actions.max()),
            "normalized_mean": float(normalized_actions.mean()),
            "normalized_std": float(normalized_actions.std()),
            "env_min": float(actions.min()),
            "env_max": float(actions.max()),
            "env_mean": float(actions.mean()),
            "env_std": float(actions.std()),
            "first_env_action": actions[0].tolist(),
        }
        return actions


@dataclasses.dataclass
class EvalArgs:
    model: str
    host: str = "127.0.0.1"
    port: int = 18000
    assets_checkpoint: Path | None = None
    libero_home: Path = DEFAULT_LIBERO_HOME
    task_suite: str = "libero_spatial"
    num_trials: int = 1
    max_tasks: int = 1
    task_start: int = 0
    task_end: int = -1
    num_steps_wait: int = 10
    replan_steps: int = 5
    resize_size: int = 224
    seed: int = 7
    result_json: Path | None = None
    log_action_stats: bool = True
    debug_dump_dir: Path | None = None
    noise_seed: int | None = None
    model_source: str = OPENPI_MODEL_SOURCE


def run(args: EvalArgs) -> dict[str, Any]:
    np.random.seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(args.libero_home / "libero"))
    if str(args.libero_home) not in sys.path:
        sys.path.insert(0, str(args.libero_home))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    model_name = canonicalize_model_name(args.model)
    client = StarVLAOpenPIClient(
        args.host,
        args.port,
        model_name,
        noise_seed=args.noise_seed,
        assets_checkpoint=args.assets_checkpoint,
        model_source=args.model_source,
    )

    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    num_tasks = task_suite.n_tasks
    task_end = num_tasks if args.task_end <= 0 else min(args.task_end, num_tasks)
    task_start = max(0, args.task_start)
    if args.max_tasks > 0:
        task_end = min(task_end, task_start + args.max_tasks)
    max_steps = get_max_steps(args.task_suite)
    replan_steps = int(args.replan_steps)

    total_episodes = 0
    total_successes = 0
    per_task: dict[str, dict[str, int]] = {}
    dumped_first_action = False

    task_iter = tqdm.tqdm(
        range(task_start, task_end),
        desc=f"{model_name} {args.task_suite}",
        dynamic_ncols=True,
    )
    for task_id in task_iter:
        task = task_suite.get_task(task_id)
        task_description = task.language
        initial_states = task_suite.get_task_init_states(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        task_episodes = 0
        task_successes = 0
        log.info("[task %s/%s] %s", task_id + 1, num_tasks, task_description)

        ep_iter = tqdm.tqdm(
            range(args.num_trials),
            desc=f"task {task_id + 1}/{num_tasks}",
            leave=False,
            dynamic_ncols=True,
        )
        for ep_idx in ep_iter:
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])
            action_plan: collections.deque[np.ndarray] = collections.deque()
            done = False
            t = 0
            logged_action_stats = False

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                example, _ = preprocess_env_obs(obs, task_description, args.resize_size)
                if not action_plan:
                    actions = client.predict_env_actions(example)
                    if args.debug_dump_dir is not None and not dumped_first_action:
                        args.debug_dump_dir.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(
                            args.debug_dump_dir / "first_policy_call.npz",
                            base_image=np.asarray(example["image"][0]),
                            wrist_image=np.asarray(example["image"][1]),
                            raw_state=np.asarray(example["raw_state"], dtype=np.float32),
                            normalized_state=normalize_openpi_value(
                                example["raw_state"],
                                client.norm_stats["state"],
                                client.use_quantile_norm,
                            ).astype(np.float32),
                            normalized_actions=np.asarray(
                                client.last_action_stats.get("normalized_actions", [])
                                if client.last_action_stats is not None
                                else [],
                                dtype=np.float32,
                            ),
                            noise=np.asarray([] if client.last_noise is None else client.last_noise, dtype=np.float32),
                            env_actions=actions,
                            first_env_action=actions[0],
                            task_description=np.asarray(str(task_description)),
                            model=np.asarray(model_name),
                            task_suite=np.asarray(args.task_suite),
                            task_id=np.asarray(task_id),
                            episode_idx=np.asarray(ep_idx),
                        )
                        log.info("  wrote debug dump: %s", args.debug_dump_dir / "first_policy_call.npz")
                        dumped_first_action = True
                    if args.log_action_stats and not logged_action_stats and client.last_action_stats is not None:
                        stats = client.last_action_stats
                        log.info(
                            "  ep=%s action_stats norm[min=%.3f max=%.3f mean=%.3f std=%.3f] "
                            "env[min=%.3f max=%.3f mean=%.3f std=%.3f] first=%s",
                            ep_idx,
                            stats["normalized_min"],
                            stats["normalized_max"],
                            stats["normalized_mean"],
                            stats["normalized_std"],
                            stats["env_min"],
                            stats["env_max"],
                            stats["env_mean"],
                            stats["env_std"],
                            np.array2string(np.asarray(stats["first_env_action"]), precision=4, suppress_small=True),
                        )
                        logged_action_stats = True
                    if len(actions) < replan_steps:
                        raise ValueError(f"replan_steps={replan_steps} but server returned {len(actions)} actions")
                    action_plan.extend(actions[:replan_steps])

                obs, _, done, _ = env.step(action_plan.popleft().tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            task_episodes += 1
            total_episodes += 1
            task_sr = task_successes / max(task_episodes, 1)
            total_sr = total_successes / max(total_episodes, 1)
            ep_iter.set_postfix(
                task_sr=f"{task_successes}/{task_episodes} ({100.0 * task_sr:.1f}%)",
                total_sr=f"{total_successes}/{total_episodes} ({100.0 * total_sr:.1f}%)",
            )
            task_iter.set_postfix(total_sr=f"{total_successes}/{total_episodes} ({100.0 * total_sr:.1f}%)")
            log.info(
                "  ep=%s %s task_sr=%s/%s %.1f%% total_sr=%s/%s %.1f%%",
                ep_idx,
                "SUCCESS" if done else "fail",
                task_successes,
                task_episodes,
                100.0 * task_sr,
                total_successes,
                total_episodes,
                100.0 * total_sr,
            )

        per_task[task_description] = {"success": task_successes, "total": task_episodes}
        log.info(
            "[task %s/%s done] task_sr=%s/%s %.1f%% total_sr=%s/%s %.1f%%",
            task_id + 1,
            num_tasks,
            task_successes,
            task_episodes,
            100.0 * task_successes / max(task_episodes, 1),
            total_successes,
            total_episodes,
            100.0 * total_successes / max(total_episodes, 1),
        )
        env.close()

    summary = {
        "model": model_name,
        "host": args.host,
        "port": args.port,
        "task_suite": args.task_suite,
        "task_start": task_start,
        "task_end": task_end,
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "success_rate": float(total_successes / max(total_episodes, 1)),
        "per_task": per_task,
    }
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("FINAL SR: %s/%s %.2f%%", total_successes, total_episodes, 100.0 * summary["success_rate"])
    return summary


def parse_args() -> EvalArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["PI0", "PI05", "pi0", "pi05"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--assets-checkpoint", type=Path, default=None)
    parser.add_argument("--libero-home", type=Path, default=DEFAULT_LIBERO_HOME)
    parser.add_argument("--task-suite", default="libero_spatial", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=-1)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--log-action-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-dump-dir", type=Path, default=None)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--model-source", default=OPENPI_MODEL_SOURCE, choices=[OPENPI_MODEL_SOURCE, STARVLA_MODEL_SOURCE])
    ns = parser.parse_args()
    return EvalArgs(**vars(ns))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
