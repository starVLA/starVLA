#!/usr/bin/env python3
"""
Worker script for LIBERO parallel evaluation.

Launch one worker per GPU:

    CUDA_VISIBLE_DEVICES=0 ${STARVLA_PYTHON} \\
        examples/LIBERO/eval_files/parallel_eval_labtasker/run.py --env .env

Each task is fully self-contained: the worker starts a policy server for the
task's checkpoint, runs the eval suite, then shuts the server down before
moving to the next task.

Required env vars:
    STARVLA_PYTHON   Python interpreter with starVLA installed (runs the policy server)
    LIBERO_PYTHON    Python interpreter with LIBERO installed (runs eval_libero.py)
    LIBERO_HOME      Root directory of the LIBERO repo

Optional env vars:
    LIBERO_NUM_TRIALS    Episodes per task (default: 50)
    SERVER_TIMEOUT       Seconds to wait for the policy server to become ready (default: 300)
    CUDA_VISIBLE_DEVICES GPU(s) to use; worker uses the first device listed (default: 0)

Optional flags:
    --env <path>     Load a .env file before reading env vars (key=value, # comments ok)
                     Copy .env.example to .env, fill in your paths, then pass --env .env
"""

import argparse
import os
import pathlib
import socket
import subprocess
import sys
import time

from dotenv import load_dotenv

try:
    import labtasker
    from labtasker import Required
except ImportError:
    print("[ERROR] Labtasker not installed. Install it with `pip install 'labtasker[plugins]'`")
    exit(1)

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]  # starVLA/


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


def _run(proc: subprocess.Popen, log_path: pathlib.Path) -> int:
    """Tee proc stdout+stderr to our stdout and to log_path; return exit code."""
    with open(log_path, "w") as lf:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
    proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env", default=None)
    args, _ = parser.parse_known_args()
    if args.env:
        load_dotenv(args.env, override=False)

    starvla_py = _require_env("STARVLA_PYTHON")
    libero_py = _require_env("LIBERO_PYTHON")
    libero_home = _require_env("LIBERO_HOME")

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    num_trials = os.environ.get("LIBERO_NUM_TRIALS", "50")
    server_timeout = int(os.environ.get("SERVER_TIMEOUT", "300"))

    @labtasker.loop(extra_filter='metadata.benchmark == "LIBERO"')
    def run_task(
            ckpt: str = Required(),
            task_suite: str = Required(),
    ) -> None:
        base_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": gpu,
            "LIBERO_CONFIG_PATH": str(pathlib.Path(libero_home) / "libero"),
            "PYTHONPATH": os.pathsep.join(filter(None, [
                str(PROJECT_ROOT),
                libero_home,
                os.environ.get("PYTHONPATH", ""),
            ])),
        }

        ckpt_path = pathlib.Path(ckpt)
        ckpt_stem = ckpt_path.stem
        ckpt_dir = ckpt_path.parent
        port = _find_free_port()

        video_out = ckpt_dir / "videos" / task_suite / ckpt_stem
        log_dir = ckpt_dir / "logs" / task_suite
        video_out.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Start policy server for this task's checkpoint
        server_log = log_dir / f"{ckpt_stem}_server.log"
        server_proc = subprocess.Popen(
            [
                starvla_py,
                str(PROJECT_ROOT / "deployment" / "model_server" / "server_policy.py"),
                "--ckpt_path", ckpt,
                "--port", str(port),
                "--use_bf16",
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

        rc = -1
        try:
            eval_log = log_dir / f"{ckpt_stem}.log"
            print(f"[INFO] suite={task_suite}  gpu={gpu}  port={port}")
            eval_proc = subprocess.Popen(
                [
                    libero_py,
                    str(PROJECT_ROOT / "examples" / "LIBERO" / "eval_files" / "eval_libero.py"),
                    "--args.pretrained-path", ckpt,
                    "--args.host", "127.0.0.1",
                    "--args.port", str(port),
                    "--args.task-suite-name", task_suite,
                    "--args.num-trials-per-task", num_trials,
                    "--args.video-out-path", str(video_out),
                ],
                env=base_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            rc = _run(eval_proc, eval_log)
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
            # labtasker.finish() is called inside eval_libero.py when labtasker is installed

    run_task()


if __name__ == "__main__":
    main()
