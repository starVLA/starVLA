#!/usr/bin/env python3
"""Resumable, auditable acquisition entry point for the StarVLA UMI sample set."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PLANS = HERE / "plans"
DEFAULT_ROOT = Path(os.environ.get("UMI_DATA_ROOT", "./umi_data")).expanduser()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_plan() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for filename, source_type in (
        ("hf_400case_selection.lock.json", "huggingface"),
        ("additional_hf_sources.lock.json", "huggingface"),
        ("direct_sources.lock.json", "direct"),
    ):
        for name, item in load_json(PLANS / filename).items():
            if name in result:
                raise ValueError(f"duplicate source in lock files: {name}")
            result[name] = {**item, "source_type": source_type}
    return result


def family_name(name: str, item: dict[str, Any]) -> str:
    if "family" in item:
        return str(item["family"])
    if name.startswith("LivUMI-"):
        return "LivUMI"
    return name.removesuffix("-Sample")


def selected_plan(plan: dict[str, dict[str, Any]], families: list[str]) -> dict[str, dict[str, Any]]:
    if not families:
        return plan
    wanted = {part.strip().lower() for value in families for part in value.split(",") if part.strip()}
    selected = {
        name: item
        for name, item in plan.items()
        if name.lower() in wanted or family_name(name, item).lower() in wanted
    }
    missing = wanted - {
        key.lower()
        for name, item in selected.items()
        for key in (name, family_name(name, item))
    }
    if missing:
        raise ValueError(f"unknown families: {', '.join(sorted(missing))}")
    return selected


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError


def source_dir(root: Path, name: str, item: dict[str, Any]) -> Path:
    if item["source_type"] == "direct":
        return root / "_archives"
    # Preserve the established samples_400 layout so an existing partial
    # download is adopted rather than duplicated under a new raw/ tree.
    return root / str(item["family"]) / name if "family" in item else root / name


def write_status(root: Path, status: dict[str, Any]) -> None:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    temporary = state / "download_status.json.tmp"
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(state / "download_status.json")


def disk_guard(root: Path, minimum_gib: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    minimum = int(minimum_gib * 1024**3)
    if free < minimum:
        raise RuntimeError(f"only {human_bytes(free)} free under {root}; required reserve is {minimum_gib:g} GiB")


def hf_download(name: str, item: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    destination = source_dir(root, name, item)
    if args.dry_run:
        return {"status": "planned", "destination": str(destination), "files": len(item["files"])}
    disk_guard(root, args.min_free_gib)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=item["repo"],
            repo_type="dataset",
            allow_patterns=item["files"],
            local_dir=destination,
            token=os.environ.get("HF_TOKEN") or None,
            max_workers=args.hf_workers,
        )
    except ImportError:
        hf_cli = shutil.which("hf")
        if not hf_cli:
            raise RuntimeError(
                "neither the huggingface_hub Python package nor the `hf` CLI is installed; "
                "install tools/requirements-download.txt"
            )
        command = [hf_cli, "download", item["repo"], "--repo-type", "dataset", "--local-dir", str(destination)]
        for pattern in item["files"]:
            command.extend(("--include", pattern))
        subprocess.run(command, check=True)
    count, missing = matches_exist(destination, item["files"])
    if missing:
        raise IOError(f"download finished with {len(missing)} missing patterns: {missing[:5]}")
    return {"status": "downloaded", "destination": str(destination), "files_on_disk": count}


def normal_direct_download(url: str, partial: Path, offset: int, timeout: int):
    import requests

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = requests.get(url, headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()
    mode = "ab" if offset and response.status_code == 206 else "wb"
    if mode == "wb":
        offset = 0
    with partial.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
    return offset


def google_drive_download(file_id: str, partial: Path, offset: int, timeout: int):
    import requests

    session = requests.Session()
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    response = session.get(url, params=params, headers=headers, stream=True, timeout=timeout)
    response.raise_for_status()
    mode = "ab" if offset and response.status_code == 206 else "wb"
    if mode == "wb":
        offset = 0
    with partial.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
    return offset


def direct_download(name: str, item: dict[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    destination = source_dir(root, name, item) / item["filename"]
    expected = int(item["expected_bytes"])
    if destination.exists() and destination.stat().st_size == expected:
        return {"status": "present", "destination": str(destination), "bytes": expected}
    if args.dry_run:
        return {"status": "planned", "destination": str(destination), "bytes": expected}
    disk_guard(root, args.min_free_gib)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists() and partial.stat().st_size == expected:
        partial.replace(destination)
        return {"status": "recovered", "destination": str(destination), "bytes": expected}
    if partial.exists() and partial.stat().st_size > expected:
        corrupt = partial.with_suffix(partial.suffix + f".oversize-{int(time.time())}")
        partial.replace(corrupt)
    for attempt in range(1, args.retries + 1):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if item.get("google_drive_id"):
                google_drive_download(item["google_drive_id"], partial, offset, args.timeout)
            else:
                normal_direct_download(item["url"], partial, offset, args.timeout)
            actual = partial.stat().st_size
            if actual != expected:
                raise IOError(f"size mismatch: expected {expected}, got {actual}")
            partial.replace(destination)
            return {"status": "downloaded", "destination": str(destination), "bytes": actual}
        except Exception:
            if attempt == args.retries:
                raise
            time.sleep(min(60, 2**attempt))
    raise AssertionError


def matches_exist(base: Path, patterns: list[str]) -> tuple[int, list[str]]:
    existing = [str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()] if base.exists() else []
    def matches(path: str, pattern: str) -> bool:
        if pattern.endswith("/**"):
            return path.startswith(pattern[:-2])
        return fnmatch.fnmatch(path, pattern)

    missing = [pattern for pattern in patterns if not any(matches(path, pattern) for path in existing)]
    return len(existing), missing


def verify_one(name: str, item: dict[str, Any], root: Path, deep: bool) -> dict[str, Any]:
    if item["source_type"] == "huggingface":
        base = source_dir(root, name, item)
        count, missing = matches_exist(base, item["files"])
        return {"status": "ok" if not missing else "incomplete", "files_on_disk": count, "missing": missing}
    path = source_dir(root, name, item) / item["filename"]
    if not path.exists():
        return {"status": "incomplete", "reason": "missing", "path": str(path)}
    actual = path.stat().st_size
    if actual != int(item["expected_bytes"]):
        return {"status": "incomplete", "reason": "size", "expected": item["expected_bytes"], "actual": actual}
    if deep and path.suffix == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
            if bad:
                return {"status": "incomplete", "reason": "bad zip member", "member": bad}
        except zipfile.BadZipFile:
            return {"status": "incomplete", "reason": "bad zip"}
    return {"status": "ok", "bytes": actual}


def summarize(results: dict[str, dict[str, Any]]) -> None:
    print(json.dumps(results, indent=2, ensure_ascii=False))
    counts: dict[str, int] = {}
    for result in results.values():
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print("summary:", ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


def command_doctor(plan: dict[str, dict[str, Any]], root: Path, args: argparse.Namespace) -> int:
    hf_count = sum(item["source_type"] == "huggingface" for item in plan.values())
    direct_count = len(plan) - hf_count
    families = {family_name(name, item) for name, item in plan.items()}
    report = {
        "root": str(root.resolve()),
        "sources": len(plan),
        "independent_families": len(families),
        "huggingface_sources": hf_count,
        "direct_sources": direct_count,
        "free_bytes": shutil.disk_usage(root.parent if not root.exists() else root).free,
        "hf_token_env": bool(os.environ.get("HF_TOKEN")),
        "deep_verify": args.deep,
    }
    print(json.dumps(report, indent=2))
    return 0


def command_download(plan: dict[str, dict[str, Any]], root: Path, args: argparse.Namespace) -> int:
    results: dict[str, dict[str, Any]] = {}
    status = {"started_at": time.time(), "root": str(root), "sources": results}
    workers = min(args.source_workers, max(1, len(plan)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for name, item in plan.items():
            function = hf_download if item["source_type"] == "huggingface" else direct_download
            futures[pool.submit(function, name, item, root, args)] = name
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                message = str(exc)
                gated = any(token in message.lower() for token in ("gated", "restricted", "agreement", "401", "403"))
                results[name] = {"status": "requires_acceptance" if gated else "failed", "error": message}
            write_status(root, status)
            print(f"[{results[name]['status']}] {name}", flush=True)
    status["finished_at"] = time.time()
    write_status(root, status)
    summarize(results)
    return 1 if any(result["status"] in {"failed", "requires_acceptance"} for result in results.values()) else 0


def command_verify(
    plan: dict[str, dict[str, Any]], root: Path, args: argparse.Namespace, *, full_plan: bool = False
) -> int:
    results = {name: verify_one(name, item, root, args.deep) for name, item in plan.items()}
    summarize(results)
    complete = all(result["status"] == "ok" for result in results.values())
    marker = root / ".all_available_400_sources_downloaded"
    if complete and full_plan:
        marker.touch()
    elif full_plan and marker.exists():
        marker.unlink()
    return 0 if complete else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("doctor", "download", "verify", "all"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--families", action="append", default=[], help="name or comma-separated names; repeatable")
    parser.add_argument("--source-workers", type=int, default=3)
    parser.add_argument("--hf-workers", type=int, default=4)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deep", action="store_true", help="also test ZIP members during verify")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.expanduser().resolve()
    try:
        complete_plan = load_plan()
        plan = selected_plan(complete_plan, args.families)
        full_plan = set(plan) == set(complete_plan)
        if args.command == "doctor":
            return command_doctor(plan, root, args)
        if args.command == "download":
            return command_download(plan, root, args)
        if args.command == "verify":
            return command_verify(plan, root, args, full_plan=full_plan)
        download_status = command_download(plan, root, args)
        if download_status and not args.dry_run:
            return download_status
        return 0 if args.dry_run else command_verify(plan, root, args, full_plan=full_plan)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
