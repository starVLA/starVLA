#!/usr/bin/env python3
"""Run a RoboCasa checkpoint eval with atomic status and video QA."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_ENV = "gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env"


class EvalFailed(RuntimeError):
    pass


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(2.0)
    raise EvalFailed(f"Policy server did not open {host}:{port} within {timeout_s}s: {last_error}")


def terminate_process(process: subprocess.Popen | None, timeout_s: float = 20.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


def run_process(cmd: list[str], *, env: dict, log_path: Path, timeout_s: float | None = None, cwd: str | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            terminate_process(process)
            raise EvalFailed(f"Command timed out after {timeout_s}s: {' '.join(cmd)}") from exc


def join_pythonpath(*paths: str | Path | None) -> str:
    values = [str(path) for path in paths if path]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        values.append(existing)
    return os.pathsep.join(values)


def conda_site_packages(python_bin: str) -> Path:
    env_root = Path(python_bin).resolve().parents[1]
    return env_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def _video_stats_cv2(video_path: Path, max_frames: int) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise EvalFailed(f"Could not open video: {video_path}")
    means = []
    stds = []
    frames = 0
    while frames < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        means.append(float(frame.mean()))
        stds.append(float(frame.std()))
        frames += 1
    capture.release()
    return {
        "path": str(video_path),
        "frames_sampled": frames,
        "mean_max": max(means) if means else 0.0,
        "std_max": max(stds) if stds else 0.0,
    }


def _video_stats_av(video_path: Path, max_frames: int) -> dict:
    import av

    means = []
    stds = []
    frames = 0
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            array = frame.to_ndarray(format="rgb24")
            means.append(float(array.mean()))
            stds.append(float(array.std()))
            frames += 1
            if frames >= max_frames:
                break
    return {
        "path": str(video_path),
        "frames_sampled": frames,
        "mean_max": max(means) if means else 0.0,
        "std_max": max(stds) if stds else 0.0,
    }


def video_stats(video_path: Path, max_frames: int) -> dict:
    try:
        return _video_stats_cv2(video_path, max_frames)
    except ModuleNotFoundError:
        return _video_stats_av(video_path, max_frames)


def validate_videos(video_dir: Path, *, min_frames: int, max_frames: int, min_mean: float, min_std: float) -> dict:
    videos = sorted(video_dir.rglob("*.mp4"))
    if not videos:
        raise EvalFailed(f"No mp4 videos found under {video_dir}")

    stats = [video_stats(path, max_frames) for path in videos]
    usable = [
        item
        for item in stats
        if item["frames_sampled"] >= min_frames and item["mean_max"] >= min_mean and item["std_max"] >= min_std
    ]
    if not usable:
        raise EvalFailed(
            "All videos look empty or black. "
            f"Thresholds: min_frames={min_frames}, min_mean={min_mean}, min_std={min_std}. Stats={stats[:3]}"
        )
    return {"videos": stats, "usable_videos": len(usable)}


def parse_success_rate(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        marker = "Success rate:"
        if marker in line:
            try:
                return float(line.split(marker, 1)[1].strip())
            except ValueError:
                return None
    return None


def parse_episode_results(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    pattern = re.compile(
        r"Episode (?P<episode>\d+): env=(?P<env>\d+), "
        r"success=(?P<success>True|False), "
        r"length=(?P<length>\d+), "
        r"reward=(?P<reward>[-+0-9.eE]+), "
        r"terminated=(?P<terminated>True|False), "
        r"truncated=(?P<truncated>True|False)"
    )
    results = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        item = match.groupdict()
        results.append(
            {
                "episode": int(item["episode"]),
                "env": int(item["env"]),
                "success": item["success"] == "True",
                "length": int(item["length"]),
                "reward": float(item["reward"]),
                "terminated": item["terminated"] == "True",
                "truncated": item["truncated"] == "True",
            }
        )
    return results


def run_attempt(args: argparse.Namespace, attempt_dir: Path) -> dict:
    logs_dir = attempt_dir / "logs"
    video_dir = attempt_dir / "videos"
    logs_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    server_env = os.environ.copy()
    server_env["CUDA_VISIBLE_DEVICES"] = str(args.server_gpu)
    server_env["PYTHONPATH"] = join_pythonpath(args.repo_root)
    if args.norm_action_stats_every > 0:
        server_env["STARVLA_NORM_ACTION_STATS_EVERY"] = str(args.norm_action_stats_every)

    sim_env = os.environ.copy()
    sim_env["CUDA_VISIBLE_DEVICES"] = str(args.sim_gpu)
    sim_env["PYTHONPATH"] = join_pythonpath(args.repo_root, conda_site_packages(args.starvla_python))
    if args.action_stats_every > 0:
        sim_env["ROBOCASA_ACTION_STATS_EVERY"] = str(args.action_stats_every)

    server_cmd = [
        args.starvla_python,
        "deployment/model_server/server_policy.py",
        "--ckpt_path",
        args.ckpt,
        "--port",
        str(args.port),
        "--idle_timeout",
        str(args.server_idle_timeout),
    ]
    if args.use_bf16:
        server_cmd.append("--use_bf16")
    sim_cmd = [
        args.robocasa_python,
        "examples/simBenchmarks/Robocasa_tabletop/eval_files/simulation_env.py",
        "--args.env_name",
        args.env_name,
        "--args.host",
        args.host,
        "--args.port",
        str(args.port),
        "--args.n_episodes",
        str(args.n_episodes),
        "--args.n_envs",
        str(args.n_envs),
        "--args.max_episode_steps",
        str(args.max_episode_steps),
        "--args.n_action_steps",
        str(args.n_action_steps),
        "--args.pretrained_path",
        args.ckpt,
    ]
    if args.no_video:
        sim_cmd.extend(["--args.video_out_path", "none"])
    else:
        sim_cmd.extend(["--args.video_out_path", str(video_dir)])

    server_log = logs_dir / "server.log"
    sim_log = logs_dir / "simulation.log"
    server = None
    with server_log.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT, env=server_env, cwd=args.repo_root)
    try:
        wait_for_port(args.host, args.port, args.server_ready_timeout)
        sim_rc = run_process(sim_cmd, env=sim_env, log_path=sim_log, timeout_s=args.sim_timeout, cwd=args.repo_root)  # type: ignore[arg-type]
        if sim_rc != 0:
            raise EvalFailed(f"simulation_env.py exited with code {sim_rc}. See {sim_log}")
        video_report = (
            {"skipped": True, "reason": "--no-video"}
            if args.no_video
            else validate_videos(
                video_dir,
                min_frames=args.video_min_frames,
                max_frames=args.video_sample_frames,
                min_mean=args.video_min_mean,
                min_std=args.video_min_std,
            )
        )
        return {
            "status": "complete",
            "ckpt": args.ckpt,
            "env_name": args.env_name,
            "success_rate": parse_success_rate(sim_log),
            "episodes": parse_episode_results(sim_log),
            "server_log": str(server_log),
            "simulation_log": str(sim_log),
            "video_dir": str(video_dir),
            "video_report": video_report,
        }
    finally:
        terminate_process(server)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="playground/Checkpoints/qwen_var_productvq_g16_s124816_robocasa_epoch027_100k_fullcache/checkpoints/steps_30000_pytorch_model.pt")
    parser.add_argument("--output-dir", default="playground/Checkpoints/qwen_var_productvq_g16_s124816_robocasa_epoch027_100k_fullcache/robocasa_eval/steps_30000_smoke")
    parser.add_argument("--env-name", default=DEFAULT_ENV)
    default_repo_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--repo-root", default=str(default_repo_root))
    parser.add_argument("--starvla-python", default=sys.executable)
    parser.add_argument("--robocasa-python", default=os.environ.get("ROBOCASA_PYTHON", "python"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6416)
    parser.add_argument("--server-gpu", default="6")
    parser.add_argument("--sim-gpu", default="6")
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--server-ready-timeout", type=float, default=600.0)
    parser.add_argument("--server-idle-timeout", type=int, default=1800)
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--action-stats-every", type=int, default=0)
    parser.add_argument("--norm-action-stats-every", type=int, default=0)
    parser.add_argument("--sim-timeout", type=float, default=3600.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-sample-frames", type=int, default=60)
    parser.add_argument("--video-min-frames", type=int, default=3)
    parser.add_argument("--video-min-mean", type=float, default=5.0)
    parser.add_argument("--video-min-std", type=float, default=2.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    complete_path = output_dir / "COMPLETE.json"
    invalid_path = output_dir / "INVALID.json"
    running_path = output_dir / "RUNNING.json"
    if complete_path.exists() and not args.force:
        print(f"Existing complete result: {complete_path}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(running_path, {"status": "running", "started_at": time.time(), "ckpt": args.ckpt})

    def mark_invalid(reason: str, attempt: int | None = None) -> None:
        write_json(
            invalid_path,
            {
                "status": "invalid",
                "reason": reason,
                "attempt": attempt,
                "ckpt": args.ckpt,
                "env_name": args.env_name,
                "updated_at": time.time(),
            },
        )
        running_path.unlink(missing_ok=True)

    def handle_signal(signum, _frame):
        mark_invalid(f"interrupted by signal {signum}")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    last_error = None
    for attempt in range(1, args.attempts + 1):
        attempt_dir = output_dir / f"attempt_{attempt:02d}"
        try:
            result = run_attempt(args, attempt_dir)
            result["attempt"] = attempt
            result["completed_at"] = time.time()
            write_json(complete_path, result)
            invalid_path.unlink(missing_ok=True)
            running_path.unlink(missing_ok=True)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            last_error = repr(exc)
            mark_invalid(last_error, attempt)
            if attempt < args.attempts:
                time.sleep(10.0)

    print(f"Eval failed: {last_error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
