"""Convert IndoorUAV VLA subset (Replica scenes) to starVLA LeRobot-v2.0 format.

Self-contained: only needs numpy / pyarrow / pillow / ffmpeg. No `lerobot` lib.

Input layout (under --raw_root):
  extracted/vla_ins/vla_ins/<scene_group>/<scene>/<traj>/vla_ins_<N>.json
      {"instruction": str, "source": [start_1based, end_1based]}
  replica_extracted/<scene_group>/<scene>/<traj>/
      posture.json:   list[[x, y, z, yaw_deg]]
      screenshots/<frame_1based>.png  (1280x720 RGBA)

Output (under --out_dir):
  meta/info.json
  meta/episodes.jsonl
  meta/tasks.jsonl
  meta/modality.json
  data/chunk-000/episode_{idx:06d}.parquet
  videos/chunk-000/observation.images.front/episode_{idx:06d}.mp4

State  = pose[t]                       (4 dims: x, y, z, yaw_deg)
Action = pose[t+1] - pose[t]           (yaw wrapped to (-180, 180])
         terminal frame action = 0
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ----------------------------- helpers -----------------------------

def safe_load_json(p: Path):
    for enc in ("utf-8", "latin-1", "gbk"):
        try:
            with open(p, encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode {p}")


def yaw_delta_deg(yaw_next: float, yaw_curr: float) -> float:
    d = (yaw_next - yaw_curr) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def encode_video_ffmpeg(frame_paths: list[Path], out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.with_suffix(".concat.txt")
    with open(list_file, "w") as f:
        for p in frame_paths:
            f.write(f"file '{p.resolve()}'\n")
            f.write(f"duration {1.0/fps}\n")
        f.write(f"file '{frame_paths[-1].resolve()}'\n")  # ffmpeg concat quirk
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-fps_mode", "cfr", "-r", str(fps),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        "-vf", "scale='trunc(iw/2)*2':'trunc(ih/2)*2'",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    list_file.unlink()


def build_episode(vla_path: Path, scene_group: str, raw_root: Path) -> dict | None:
    """Read one VLA json + its traj metadata, return episode dict or None on error."""
    vla = safe_load_json(vla_path)
    instruction = vla["instruction"]
    s, e = vla["source"]  # both 1-based inclusive
    # vla_path: .../vla_ins/<group>/<scene>/<traj>/vla_ins_N.json
    traj_rel = vla_path.parent.relative_to(
        raw_root / "extracted" / "vla_ins" / "vla_ins" / scene_group
    )
    img_dir = raw_root / "replica_extracted" / scene_group / traj_rel / "screenshots"
    posture_path = raw_root / "replica_extracted" / scene_group / traj_rel / "posture.json"

    if not posture_path.exists():
        print(f"  SKIP missing posture: {posture_path}")
        return None

    posture = safe_load_json(posture_path)  # list of [x, y, z, yaw]
    n_total = len(posture)
    if e > n_total or s < 1:
        print(f"  SKIP out-of-range {vla_path}: [{s},{e}] vs {n_total} frames")
        return None

    # State and action sequences. We need actions for frames s..e (1-based inclusive)
    # Action[t] = pose[t+1] - pose[t]. Terminal action (t=e) uses zeros.
    frame_paths: list[Path] = []
    states: list[list[float]] = []
    actions: list[list[float]] = []
    for f1 in range(s, e + 1):  # 1-based
        idx = f1 - 1
        pose = posture[idx]
        states.append([float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3])])
        if f1 < e and (idx + 1) < n_total:
            nxt = posture[idx + 1]
            dx = float(nxt[0]) - float(pose[0])
            dy = float(nxt[1]) - float(pose[1])
            dz = float(nxt[2]) - float(pose[2])
            dyaw = yaw_delta_deg(float(nxt[3]), float(pose[3]))
            actions.append([dx, dy, dz, dyaw])
        else:
            actions.append([0.0, 0.0, 0.0, 0.0])
        img_path = img_dir / f"{f1}.png"
        if not img_path.exists():
            print(f"  SKIP missing image: {img_path}")
            return None
        frame_paths.append(img_path)

    return {
        "instruction": instruction,
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "frame_paths": frame_paths,
        "traj_rel": str(traj_rel),
        "vla_name": vla_path.stem,
    }


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scene_group", default="replica")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--max_episodes", type=int, default=None)
    args = ap.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    vla_root = raw_root / "extracted" / "vla_ins" / "vla_ins" / args.scene_group

    vla_files = sorted(vla_root.glob("*/traj_*/vla_ins_*.json"))
    print(f"Found {len(vla_files)} VLA instances in {vla_root}")
    if args.max_episodes:
        vla_files = vla_files[: args.max_episodes]
        print(f"Truncated to first {len(vla_files)} for debug")

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos" / "chunk-000" / "observation.images.front"
    video_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: build episodes, dedupe instructions → task_index
    episodes = []
    task_to_index: dict[str, int] = {}
    for i, vla_path in enumerate(vla_files):
        print(f"[{i+1}/{len(vla_files)}] {vla_path.relative_to(vla_root)}")
        ep = build_episode(vla_path, args.scene_group, raw_root)
        if ep is None:
            continue
        if ep["instruction"] not in task_to_index:
            task_to_index[ep["instruction"]] = len(task_to_index)
        ep["task_index"] = task_to_index[ep["instruction"]]
        ep["episode_index"] = len(episodes)
        episodes.append(ep)

    print(f"\nValid episodes: {len(episodes)} | unique tasks: {len(task_to_index)}")

    # Pass 2: write parquet + mp4 per episode
    total_frames = 0
    h_w_c = None
    episodes_jsonl = []
    for ep in episodes:
        idx = ep["episode_index"]
        ep_len = len(ep["states"])
        global_indices = np.arange(total_frames, total_frames + ep_len, dtype=np.int64)

        # parquet: rows are timesteps. Columns: observation.state, action,
        # episode_index, frame_index, timestamp, index, task_index
        table = pa.table({
            "observation.state": pa.array(
                [list(map(float, row)) for row in ep["states"]],
                type=pa.list_(pa.float32(), 4),
            ),
            "action": pa.array(
                [list(map(float, row)) for row in ep["actions"]],
                type=pa.list_(pa.float32(), 4),
            ),
            "episode_index": pa.array([idx] * ep_len, type=pa.int64()),
            "frame_index": pa.array(np.arange(ep_len, dtype=np.int64)),
            "timestamp": pa.array(
                (np.arange(ep_len, dtype=np.float32) / args.fps).tolist(),
                type=pa.float32(),
            ),
            "index": pa.array(global_indices),
            "task_index": pa.array([ep["task_index"]] * ep_len, type=pa.int64()),
        })
        pq.write_table(table, data_dir / f"episode_{idx:06d}.parquet")

        # mp4
        video_path = video_dir / f"episode_{idx:06d}.mp4"
        encode_video_ffmpeg(ep["frame_paths"], video_path, args.fps)

        # peek size from first image once
        if h_w_c is None:
            from PIL import Image
            with Image.open(ep["frame_paths"][0]) as im:
                w, h = im.size
            h_w_c = (h, w, 3)

        episodes_jsonl.append({
            "episode_index": idx,
            "tasks": [ep["instruction"]],
            "length": ep_len,
        })
        total_frames += ep_len

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for r in episodes_jsonl:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task, ti in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": ti, "task": task}, ensure_ascii=False) + "\n")

    # modality.json
    modality = {
        "state": {
            "x":   {"start": 0, "end": 1},
            "y":   {"start": 1, "end": 2},
            "z":   {"start": 2, "end": 3},
            "yaw": {"start": 3, "end": 4},
        },
        "action": {
            "x":   {"start": 0, "end": 1},
            "y":   {"start": 1, "end": 2},
            "z":   {"start": 2, "end": 3},
            "yaw": {"start": 3, "end": 4},
        },
        "video": {
            "front": {"original_key": "observation.images.front"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"},
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2, ensure_ascii=False)

    # info.json (LeRobot v2.0)
    info = {
        "codebase_version": "v2.0",
        "robot_type": "indoor_uav",
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "total_tasks": len(task_to_index),
        "total_videos": len(episodes),
        "total_chunks": 1,
        "chunks_size": args.chunk_size,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [4],
                "names": ["x", "y", "z", "yaw"],
            },
            "action": {
                "dtype": "float32",
                "shape": [4],
                "names": ["x", "y", "z", "yaw"],
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [h_w_c[0], h_w_c[1], h_w_c[2]],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": float(args.fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64", "shape": [1], "names": None},
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "index":         {"dtype": "int64", "shape": [1], "names": None},
            "task_index":    {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Done. Wrote {len(episodes)} episodes / {total_frames} frames to {out_dir}")


if __name__ == "__main__":
    main()
