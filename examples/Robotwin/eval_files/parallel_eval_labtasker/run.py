#!/usr/bin/env python3
"""
Worker script for RoboTwin parallel evaluation.

Launch one worker per GPU:

    CUDA_VISIBLE_DEVICES=0 \\
        ${STARVLA_PYTHON} examples/Robotwin/eval_files/parallel_eval_labtasker/run.py --env .env

Each task is fully self-contained: the worker starts a policy server for the
task's checkpoint, runs the eval, then shuts the server down before moving to
the next task.

Required env vars:
    ROBOTWIN_PATH        Path to the local RoboTwin repository
    STARVLA_PYTHON       Python interpreter with starVLA installed (runs the policy server)
    ROBOTWIN_PYTHON      Python interpreter with RoboTwin eval deps installed

Optional env vars:
    ROBOTWIN_SEED        Eval seed (default: 0)
    SERVER_TIMEOUT       Seconds to wait for the policy server (default: 600)
    CUDA_VISIBLE_DEVICES GPU(s) to use; worker uses the first device listed (default: 0)
    ROBOTWIN_LOG_ROOT    Override log root directory (default: <ckpt_dir>/robotwin_eval_logs/...)

Optional flags:
    --env <path>     Load a .env file before reading env vars (key=value, # comments ok)
                     Copy .env.example to .env, fill in your paths, then pass --env .env
"""

import argparse
import os
import pathlib
import re
import socket
import subprocess
import sys
import time

from dotenv import load_dotenv

try:
    import labtasker
    from labtasker import Required

    _has_labtasker = True
except ImportError:
    _has_labtasker = False

HERE = pathlib.Path(__file__).resolve().parent
EVAL_DIR = HERE.parent  # examples/Robotwin/eval_files/


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"[ERROR] {name} is not set. Export it before launching this worker.")
    return val


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(2)
    return False


def _stream(proc: subprocess.Popen, log_path: pathlib.Path) -> tuple[int, str]:
    """Tee proc stdout+stderr to our stdout and log_path; return (exit_code, full_output)."""
    lines: list[str] = []
    with open(log_path, "w") as lf:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)


def _parse_success_rate(output: str) -> float | None:
    # Strip ANSI escape codes (eval_policy.py emits them even to a pipe)
    clean = re.sub(r'\x1b\[[0-9;]*m', '', output)
    # Prefer the "=> Z%" form which gives the exact percentage
    matches = re.findall(r"[Ss]uccess\s+rate:\s*\d+/\d+\s*=>\s*([0-9]+\.?[0-9]*)%", clean)
    if matches:
        return float(matches[-1]) / 100.0
    # Fallback: plain "success rate: Z" (handles non-ANSI outputs)
    matches = re.findall(r"[Ss]uccess\s+rate[:\s]+([0-9]+\.?[0-9]*)", clean)
    if not matches:
        return None
    raw = float(matches[-1])
    return raw / 100.0 if raw > 1.0 else raw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", default=None)
    args, _ = parser.parse_known_args()
    if args.env:
        load_dotenv(args.env, override=False)

    robotwin_path = _require_env("ROBOTWIN_PATH")
    starvla_py = _require_env("STARVLA_PYTHON")
    robotwin_py = _require_env("ROBOTWIN_PYTHON")

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    seed = os.environ.get("ROBOTWIN_SEED", "0")
    server_timeout = int(os.environ.get("SERVER_TIMEOUT", "600"))
    log_root = os.environ.get("ROBOTWIN_LOG_ROOT", "")

    @labtasker.loop(extra_filter='metadata.benchmark == "RoboTwin"')
    def run_task(
            ckpt: str = Required(),
            task_name: str = Required(),
            mode: str = Required(),
            policy_name: str = Required(),
    ) -> None:
        base_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu,
            "STARVLA_PYTHON": starvla_py,
            "ROBOTWIN_PYTHON": robotwin_py,
            "ROBOTWIN_PATH": robotwin_path,
        }

        ckpt_path = pathlib.Path(ckpt)
        ckpt_stem = ckpt_path.stem
        port = _find_free_port()

        base_log_dir = (
            pathlib.Path(log_root)
            if log_root
            else ckpt_path.parent / "robotwin_eval_logs" / f"{policy_name}_{ckpt_stem}"
        )
        base_log_dir.mkdir(parents=True, exist_ok=True)

        # Start policy server for this task's checkpoint
        server_log = base_log_dir / f"{ckpt_stem}_server.log"
        server_proc = subprocess.Popen(
            [
                "bash",
                str(EVAL_DIR / "run_policy_server.sh"),
                ckpt,
                gpu,
                str(port),
            ],
            env=base_env,
            stdout=open(server_log, "w"),
            stderr=subprocess.STDOUT,
        )
        print(f"[INFO] Policy server pid={server_proc.pid}  port={port}  ckpt={ckpt_stem}")
        print(f"[INFO] Waiting for policy server (timeout={server_timeout}s)...")

        if not _wait_for_port(port, server_timeout):
            server_proc.terminate()
            server_proc.wait()
            sys.exit(f"[ERROR] Policy server not ready in {server_timeout}s. See {server_log}")
        print("[INFO] Policy server ready")

        try:
            eval_log = base_log_dir / f"{task_name}_{mode}_eval.log"
            print(f"[INFO] task={task_name}  mode={mode}  gpu={gpu}  port={port}")
            eval_proc = subprocess.Popen(
                [
                    "bash",
                    str(EVAL_DIR / "eval.sh"),
                    task_name,
                    mode,
                    "starvla_demo",  # ckpt_setting
                    seed,
                    gpu,
                    ckpt,
                    str(port),
                ],
                env=base_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            rc, output = _stream(eval_proc, eval_log)
            print(f"[INFO] eval exited with code {rc}")
        finally:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            print("[INFO] Policy server stopped")

        if rc != 0:
            sys.exit(rc)  # non-zero exit → labtasker marks task failed and retries

        success_rate = _parse_success_rate(output)
        if success_rate is None:
            sys.exit("[ERROR] Could not parse success rate from eval output")
        labtasker.finish(
            status="success",
            summary={"success_rate": success_rate, "task_name": task_name, "mode": mode},
        )

    run_task()


if __name__ == "__main__":
    main()
