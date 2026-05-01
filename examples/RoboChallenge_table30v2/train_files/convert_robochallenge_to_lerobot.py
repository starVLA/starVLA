"""RoboChallenge Table30v2 — convert raw `<task>/data/episode_*` directories
into the LeRobot v2.1 layout consumed by ``starVLA.dataloader.gr00t_lerobot``.

Raw layout (per HF dataset README):
    <raw_root>/<task>/
        meta/task_info.json
        task_desc.json
        data/episode_NNNNNN/
            meta/episode_meta.json
            states/states.jsonl                  # single-arm
            videos/cam_global_rgb.mp4
            videos/cam_arm_rgb.mp4               # ARX5/UR5
            videos/cam_side_rgb.mp4              # ARX5 only

Output layout (LeRobot v2.1 + gr00t modality.json):
    <out_root>/<task>/
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        meta/modality.json
        meta/embodiment.json
        data/chunk-000/episode_NNNNNN.parquet
        videos/chunk-000/<video_key>/episode_NNNNNN.mp4   (symlinked from raw)

State / action conventions (kept identical to the official RoboChallenge
``convert_to_lerobot.py`` so EE-pose action heads transfer directly):

    UR5  / ARX5 / DOS-W1 (single-arm):
        observation.state  (7,)  = joint_positions(6) + gripper_width(1)
        action             (8,)  = ee_positions(7 quat) + gripper_width(1)
    ALOHA (dual-arm) — *not implemented yet* (left+right_states.jsonl).

The original script aligns ``state[t] = (joints,gripper) at t-1`` with
``action[t] = (ee_pose,gripper) at t``; for ``frame_interval=1`` we keep the
same alignment so a learned EE policy can replay the trajectory.

Usage::

    python examples/RoboChallenge_table30v2/train_files/convert_robochallenge_to_lerobot.py \\
        --raw-root /project/vonneumann1/jye624/Datasets/RoboChallenge_table30v2/raw \\
        --task shred_paper \\
        --out-root /project/vonneumann1/jye624/Datasets/RoboChallenge_table30v2/lerobot
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --- robot embodiments shipped by Table30v2 ---
ROBOT_TAGS = {
    "ARX5":   "arx5_robochallenge",
    "UR5":    "ur5_robochallenge",
    "DOS-W1": "dosw1_robochallenge",
    "ALOHA":  "aloha_robochallenge",
}

# Camera names per embodiment (raw filename, video_key suffix)
CAMERAS = {
    "ARX5":   [("cam_global_rgb.mp4", "cam_global"),
               ("cam_arm_rgb.mp4",    "cam_arm"),
               ("cam_side_rgb.mp4",   "cam_side")],
    "UR5":    [("cam_global_rgb.mp4", "cam_global"),
               ("cam_arm_rgb.mp4",    "cam_arm")],
    "DOS-W1": [("cam_high_rgb.mp4",          "cam_high"),
               ("cam_left_wrist_rgb.mp4",    "cam_left_wrist"),
               ("cam_right_wrist_rgb.mp4",   "cam_right_wrist")],
    "ALOHA":  [("cam_high_rgb.mp4",          "cam_high"),
               ("cam_left_wrist_rgb.mp4",    "cam_left_wrist"),
               ("cam_right_wrist_rgb.mp4",   "cam_right_wrist")],
}

SINGLE_ARM = {"ARX5", "UR5", "DOS-W1"}


def _detect_robot(task_info: dict) -> str:
    for tag in task_info["task_desc"].get("task_tag", []):
        if tag in ROBOT_TAGS:
            return tag
    raise ValueError(f"Could not infer robot from task_tag={task_info['task_desc'].get('task_tag')}")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _video_meta(mp4: Path) -> tuple[int, int, int]:
    import cv2
    v = cv2.VideoCapture(str(mp4))
    w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    v.release()
    return w, h, n


def _build_features(robot: str, fps: int, height: int, width: int,
                    state_dim: int, action_dim: int) -> dict:
    feats: dict = {}
    for _, cam in CAMERAS[robot]:
        feats[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": height, "video.width": width, "video.channels": 3,
                "video.fps": fps, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "has_audio": False,
            },
        }
    feats["observation.state"] = {"dtype": "float32", "shape": [state_dim], "names": ["state"]}
    feats["action"]            = {"dtype": "float32", "shape": [action_dim], "names": ["actions"]}
    feats["timestamp"]    = {"dtype": "float32", "shape": [1], "names": None}
    feats["frame_index"]  = {"dtype": "int64",   "shape": [1], "names": None}
    feats["episode_index"]= {"dtype": "int64",   "shape": [1], "names": None}
    feats["index"]        = {"dtype": "int64",   "shape": [1], "names": None}
    feats["task_index"]   = {"dtype": "int64",   "shape": [1], "names": None}
    return feats


def _process_episode_single_arm(states_path: Path, frame_interval: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return (state[N,7], action[N,8], timestamps[N], src_frame_indices[N])."""
    rows = _load_jsonl(states_path)
    n = len(rows)
    keep_idx: list[int] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    ts: list[float] = []
    for idx in range(frame_interval, n, frame_interval):
        prev = rows[idx - frame_interval]
        cur  = rows[idx]
        st  = np.concatenate([np.asarray(prev["joint_positions"], dtype=np.float32),
                              np.asarray([prev["gripper_width"]], dtype=np.float32)])
        act = np.concatenate([np.asarray(cur["ee_positions"], dtype=np.float32),
                              np.asarray([cur["gripper_width"]], dtype=np.float32)])
        states.append(st); actions.append(act); ts.append(float(cur.get("timestamp", idx)))
        keep_idx.append(idx)
    return np.stack(states), np.stack(actions), np.asarray(ts, dtype=np.float32), keep_idx


def _link_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def _write_parquet(df_path: Path, state: np.ndarray, action: np.ndarray,
                   timestamp: np.ndarray, episode_index: int, task_index: int,
                   global_offset: int) -> int:
    n = state.shape[0]
    table = pa.table({
        "observation.state": pa.array([row.tolist() for row in state], type=pa.list_(pa.float32(), state.shape[1])),
        "action":            pa.array([row.tolist() for row in action], type=pa.list_(pa.float32(), action.shape[1])),
        "timestamp":         pa.array(timestamp, type=pa.float32()),
        "frame_index":       pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        "episode_index":     pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
        "index":             pa.array(np.arange(global_offset, global_offset + n, dtype=np.int64), type=pa.int64()),
        "task_index":        pa.array(np.full(n, task_index, dtype=np.int64), type=pa.int64()),
    })
    df_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, df_path)
    return n


def convert_task(raw_root: Path, task: str, out_root: Path,
                 frame_interval: int = 1, overwrite: bool = False) -> None:
    raw_task = raw_root / task
    if not raw_task.is_dir():
        # Try the *_extracted/<task> layout produced by `tar -xf foo.tar -C foo_extracted`.
        alt = raw_root / f"{task}_extracted" / task
        if alt.is_dir():
            raw_task = alt
        else:
            raise FileNotFoundError(f"Raw task dir not found: {raw_task}")

    task_info = json.loads((raw_task / "meta/task_info.json").read_text())
    robot = _detect_robot(task_info)
    if robot not in SINGLE_ARM:
        raise NotImplementedError(f"Embodiment {robot} not yet supported by this converter.")
    prompt = task_info["task_desc"]["prompt"]
    fps = int(task_info["video_info"]["fps"])

    out_task = out_root / task
    if out_task.exists():
        if overwrite:
            print(f"[convert] removing existing output {out_task}")
            shutil.rmtree(out_task)
        else:
            raise FileExistsError(f"Output {out_task} exists (pass --overwrite).")
    out_task.mkdir(parents=True, exist_ok=True)
    (out_task / "meta").mkdir(parents=True, exist_ok=True)

    episodes_dirs = sorted([p for p in (raw_task / "data").iterdir() if p.is_dir()])
    if not episodes_dirs:
        raise RuntimeError(f"No episodes under {raw_task / 'data'}")

    # Probe video resolution from the first episode.
    first_video = episodes_dirs[0] / "videos" / CAMERAS[robot][0][0]
    width, height, _ = _video_meta(first_video)

    state_dim, action_dim = 7, 8  # single-arm convention

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot,
        "total_episodes": 0,        # filled below
        "total_frames": 0,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": "0:0"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _build_features(robot, fps, height, width, state_dim, action_dim),
    }

    # tasks.jsonl
    (out_task / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")

    # modality.json (gr00t format)
    state_subkeys = {
        "joint_positions": {"original_key": "observation.state", "start": 0, "end": 6},
        "gripper_width":   {"original_key": "observation.state", "start": 6, "end": 7},
    }
    action_subkeys = {
        "ee_positions":  {"original_key": "action", "start": 0, "end": 7},
        "gripper_width": {"original_key": "action", "start": 7, "end": 8},
    }
    video_subkeys = {cam: {"original_key": f"observation.images.{cam}"} for _, cam in CAMERAS[robot]}
    modality = {
        "state": state_subkeys,
        "action": action_subkeys,
        "video": video_subkeys,
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (out_task / "meta/modality.json").write_text(json.dumps(modality, indent=2))

    # embodiment.json
    (out_task / "meta/embodiment.json").write_text(json.dumps({
        "robot_name": robot,
        "robot_type": robot,
        "record_frequency": fps,
        "body_controller_frequency": fps,
        "hand_controller_frequency": fps,
        "embodiment_tag": ROBOT_TAGS[robot],
    }, indent=2))

    # Per-episode pass.
    episodes_meta_lines: list[str] = []
    total_frames = 0
    new_ep_idx = 0
    for src_dir in episodes_dirs:
        states_path = src_dir / "states/states.jsonl"
        if not states_path.is_file():
            print(f"[convert][skip] missing states.jsonl in {src_dir.name}")
            continue
        try:
            state, action, ts, src_idx = _process_episode_single_arm(states_path, frame_interval)
        except (KeyError, ValueError) as exc:
            print(f"[convert][skip] {src_dir.name}: {exc}")
            continue
        if state.shape[0] == 0:
            print(f"[convert][skip] empty episode {src_dir.name}")
            continue

        # Verify videos and link them.
        n_states = state.shape[0]
        ok = True
        for raw_name, cam_key in CAMERAS[robot]:
            mp4 = src_dir / "videos" / raw_name
            if not mp4.is_file():
                print(f"[convert][skip] {src_dir.name}: missing {raw_name}")
                ok = False; break
        if not ok:
            continue

        # Write parquet & link videos
        df = out_task / f"data/chunk-000/episode_{new_ep_idx:06d}.parquet"
        _write_parquet(df, state, action, ts, episode_index=new_ep_idx,
                       task_index=0, global_offset=total_frames)
        for raw_name, cam_key in CAMERAS[robot]:
            mp4 = src_dir / "videos" / raw_name
            dst = out_task / f"videos/chunk-000/observation.images.{cam_key}/episode_{new_ep_idx:06d}.mp4"
            _link_video(mp4, dst)

        episodes_meta_lines.append(json.dumps({
            "episode_index": new_ep_idx,
            "tasks": [prompt],
            "length": n_states,
        }))
        total_frames += n_states
        new_ep_idx += 1
        if new_ep_idx % 25 == 0:
            print(f"[convert][{task}] processed {new_ep_idx}/{len(episodes_dirs)} episodes ({total_frames} frames)")

    (out_task / "meta/episodes.jsonl").write_text("\n".join(episodes_meta_lines) + "\n")

    # Finalize info.json
    info["total_episodes"] = new_ep_idx
    info["total_frames"]   = total_frames
    info["total_videos"]   = new_ep_idx * len(CAMERAS[robot])
    info["splits"]["train"] = f"0:{new_ep_idx}"
    (out_task / "meta/info.json").write_text(json.dumps(info, indent=4))

    print(f"[convert][{task}] done. episodes={new_ep_idx}, frames={total_frames}, robot={robot}")
    print(f"[convert][{task}] output -> {out_task}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--task", required=True, type=str)
    p.add_argument("--out-root", required=True, type=Path)
    p.add_argument("--frame-interval", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    convert_task(args.raw_root, args.task, args.out_root,
                 frame_interval=args.frame_interval, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
