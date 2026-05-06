"""Background-friendly downloader: pull every `.tar.part-*` file from the
RoboChallenge/Table30v2 HF dataset, group + concatenate the parts back into
single tar files, untar them under <raw_root>/<task>/, and remove the .tar +
.tar.part-* shards once each task is fully extracted.

Skips tasks already extracted (i.e. ``<raw_root>/<task>/meta/task_info.json``
exists). Safe to re-run.

Usage::

    python examples/RoboChallenge_table30v2/train_files/download_table30v2.py \\
        --raw-root /project/vonneumann1/jye624/Datasets/RoboChallenge_table30v2/raw

Logs go to ``tmp/logs/download_table30v2.log`` when wrapped by
``download_table30v2.sh``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "RoboChallenge/Table30v2"


def list_task_parts() -> dict[str, list[str]]:
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    parts: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if ".tar.part-" not in f:
            continue
        task = f.split(".tar.part-")[0]
        parts[task].append(f)
    for task in parts:
        parts[task].sort()
    return parts


def download_task(task: str, part_files: list[str], raw_root: Path) -> None:
    out_task = raw_root / task
    if (out_task / "meta" / "task_info.json").is_file():
        print(f"[skip] {task} already extracted")
        return

    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[{task}] downloading {len(part_files)} parts ...", flush=True)
    local_paths: list[Path] = []
    for fname in part_files:
        p = hf_hub_download(REPO, fname, repo_type="dataset", local_dir=str(raw_root))
        local_paths.append(Path(p))

    # Concatenate (single part: just rename; multi: cat).
    tar_path = raw_root / f"{task}.tar"
    if tar_path.exists():
        tar_path.unlink()
    if len(local_paths) == 1:
        shutil.move(str(local_paths[0]), tar_path)
    else:
        print(f"[{task}] concatenating {len(local_paths)} parts -> {tar_path.name}", flush=True)
        with tar_path.open("wb") as out:
            for p in local_paths:
                with p.open("rb") as src:
                    shutil.copyfileobj(src, out, length=64 * 1024 * 1024)

    print(f"[{task}] extracting ...", flush=True)
    extract_dir = raw_root  # tar contains <task>/ as top-level prefix
    rc = subprocess.call(["tar", "-xf", str(tar_path), "-C", str(extract_dir)])
    if rc != 0:
        print(f"[{task}] extraction failed (rc={rc}); leaving tar in place", flush=True)
        return

    # Sanity: extracted dir exists.
    if not (out_task / "meta" / "task_info.json").is_file():
        print(f"[{task}] WARN: expected {out_task}/meta/task_info.json after extraction", flush=True)
        return

    # Cleanup: remove the tar AND any leftover .tar.part-* shards.
    tar_path.unlink(missing_ok=True)
    for p in local_paths:
        if p.exists():
            p.unlink()
    print(f"[{task}] DONE; extracted to {out_task}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=None, help="only download these tasks")
    ap.add_argument("--skip", nargs="*", default=None, help="skip these tasks")
    args = ap.parse_args()

    tasks = list_task_parts()
    print(f"[index] {len(tasks)} tasks total in {REPO}", flush=True)
    if args.only:
        tasks = {t: tasks[t] for t in args.only if t in tasks}
    if args.skip:
        tasks = {t: parts for t, parts in tasks.items() if t not in args.skip}

    for task, parts in sorted(tasks.items()):
        try:
            download_task(task, parts, args.raw_root)
        except Exception as exc:  # noqa: BLE001 — keep going on per-task failure
            print(f"[{task}] FAILED: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
