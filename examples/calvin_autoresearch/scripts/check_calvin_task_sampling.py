#!/usr/bin/env python
"""Check CALVIN ABC hard-task sampling weights without decoding videos."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.datasets import canonicalize_calvin_task


def _plain_dict(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {k: value[k] for k in value.keys()}


def _load_episodes(dataset: Path):
    episodes_path = dataset / "meta" / "episodes.jsonl"
    with episodes_path.open() as f:
        for line in f:
            episode = json.loads(line)
            task_text = str(episode.get("tasks", [""])[0])
            yield {
                "episode_index": int(episode["episode_index"]),
                "length": int(episode["length"]),
                "task_text": task_text,
                "canonical_task": canonicalize_calvin_task(task_text),
            }


def _summarize(episodes, weights):
    mass = defaultdict(float)
    counts = Counter()
    for episode, weight in zip(episodes, weights):
        task = episode["canonical_task"]
        counts[task] += 1
        mass[task] += float(weight)
    total = sum(mass.values()) or 1.0
    return counts, {task: value / total for task, value in mass.items()}


def _print_summary(title, counts, probs, top_k):
    print(f"\n{title}")
    print(f"{'task':28s} {'episodes':>8s} {'prob':>10s}")
    for task, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        print(f"{task:28s} {counts[task]:8d} {prob:10.4f}")


def _print_on_off_report(raw_counts, raw_probs, balanced_probs):
    print("\non/off pair report")
    print(f"{'pair':20s} {'on_count':>8s} {'off_count':>9s} {'base_on':>9s} {'base_off':>9s} {'bal_on':>9s} {'bal_off':>9s}")
    pairs = [
        ("lightbulb", "turn_on_lightbulb", "turn_off_lightbulb"),
        ("led", "turn_on_led", "turn_off_led"),
    ]
    for label, on_task, off_task in pairs:
        print(
            f"{label:20s} {raw_counts[on_task]:8d} {raw_counts[off_task]:9d} "
            f"{raw_probs.get(on_task, 0.0):9.4f} {raw_probs.get(off_task, 0.0):9.4f} "
            f"{balanced_probs.get(on_task, 0.0):9.4f} {balanced_probs.get(off_task, 0.0):9.4f}"
        )


def _print_language_variants(episodes, tasks, max_per_task):
    if max_per_task <= 0:
        return
    wanted = set(tasks)
    seen = {task: [] for task in wanted}
    for episode in episodes:
        task = episode["canonical_task"]
        if task not in wanted:
            continue
        text = episode["task_text"]
        if text not in seen[task]:
            seen[task].append(text)
        if all(len(values) >= max_per_task for values in seen.values()):
            break

    print("\nlanguage variants")
    for task in sorted(wanted):
        print(f"[{task}]")
        for text in seen[task][:max_per_task]:
            print(f"  - {text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant-top-k", type=int, default=8)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    sampler_cfg = _plain_dict(cfg.datasets.vla_data.get("sampler", {}))
    oversample = _plain_dict(sampler_cfg.get("oversample_tasks", {}))
    oversample = {str(task): float(weight) for task, weight in oversample.items()}

    episodes = list(_load_episodes(args.dataset))
    lengths = np.asarray([episode["length"] for episode in episodes], dtype=np.float64)
    base_weights = lengths.copy()
    factors = np.asarray([oversample.get(episode["canonical_task"], 1.0) for episode in episodes], dtype=np.float64)
    balanced_weights = base_weights * factors

    raw_counts, raw_probs = _summarize(episodes, base_weights)
    balanced_counts, balanced_probs = _summarize(episodes, balanced_weights)
    _print_summary("base length-weighted distribution", raw_counts, raw_probs, args.top_k)
    _print_summary("balanced length-weighted distribution", balanced_counts, balanced_probs, args.top_k)
    _print_on_off_report(raw_counts, raw_probs, balanced_probs)

    if args.num_samples > 0:
        rng = np.random.default_rng(args.seed)
        probs = balanced_weights / balanced_weights.sum()
        sampled_indices = rng.choice(len(episodes), size=args.num_samples, replace=True, p=probs)
        sampled_counts = Counter(episodes[index]["canonical_task"] for index in sampled_indices)
        print(f"\nsampled distribution n={args.num_samples}")
        print(f"{'task':28s} {'samples':>8s} {'rate':>10s}")
        for task, count in sampled_counts.most_common(args.top_k):
            print(f"{task:28s} {count:8d} {count / args.num_samples:10.4f}")

    print("\noversample factors")
    for task, factor in sorted(oversample.items()):
        print(f"{task:28s} {factor:g}")

    _print_language_variants(
        episodes,
        [
            "turn_on_lightbulb",
            "turn_off_lightbulb",
            "turn_on_led",
            "turn_off_led",
            "close_drawer",
            "move_slider_left",
        ],
        args.variant_top_k,
    )


if __name__ == "__main__":
    main()
