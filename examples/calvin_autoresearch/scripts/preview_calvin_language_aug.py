#!/usr/bin/env python
"""Preview CALVIN language paraphrase augmentation without decoding videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import (
    _parse_language_aug_cfg,
    apply_calvin_language_augmentation,
    canonicalize_calvin_task,
)


LANGUAGE_KEY = "annotation.human.action.task_description"


def _load_episodes(dataset: Path):
    with (dataset / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            episode = json.loads(line)
            task_text = str(episode.get("tasks", [""])[0])
            yield {
                "episode_index": int(episode["episode_index"]),
                "task_text": task_text,
                "canonical_task": canonicalize_calvin_task(task_text),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--max-per-task", type=int, default=5)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--probability", type=float, default=1.0, help="Preview override; default forces augmentation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if "language_augmentation" in cfg.datasets.vla_data and args.probability is not None:
        cfg.datasets.vla_data.language_augmentation.probability = args.probability
    lang_cfg = _parse_language_aug_cfg(cfg.datasets.vla_data)
    if not lang_cfg.get("enabled", False):
        raise SystemExit("language_augmentation.enabled is false in config")

    wanted = set(args.tasks or sorted(lang_cfg.get("paraphrases", {}).keys()))
    saved = {task: 0 for task in wanted}
    records = []
    for scanned, episode in enumerate(_load_episodes(args.dataset)):
        if scanned >= args.max_scan:
            break
        task = episode["canonical_task"]
        if task not in wanted or saved[task] >= args.max_per_task:
            continue

        rng = np.random.default_rng(args.seed + int(episode["episode_index"]))
        raw = {LANGUAGE_KEY: [episode["task_text"]]}
        aug = apply_calvin_language_augmentation(
            raw_data=raw,
            language_keys=[LANGUAGE_KEY],
            canonical_task=task,
            rng=rng,
            cfg=lang_cfg,
        )
        augmented = str(aug[LANGUAGE_KEY][0])
        records.append(
            {
                "episode_index": episode["episode_index"],
                "task": task,
                "original": episode["task_text"],
                "augmented": augmented,
                "changed": augmented != episode["task_text"],
            }
        )
        saved[task] += 1
        if all(count >= args.max_per_task for count in saved.values()):
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"saved {len(records)} language preview records to {args.output}")
    print(saved)


if __name__ == "__main__":
    main()
