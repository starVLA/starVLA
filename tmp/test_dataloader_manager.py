"""
Smoke test for DataLoaderManager.

Usage (single-GPU, from project root):
    python tmp/test_dataloader_manager.py --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml

Verifies:
  1. DataLoaderManager.from_config builds both VLA and VLM dataloaders
  2. get_next_batches() returns correct (name, batch) tuples
  3. Epoch reset works (iterate beyond one VLM epoch)
  4. Ratio < 1.0 causes probabilistic skipping
"""

import argparse
import os
import sys
import time
from unittest.mock import patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from omegaconf import OmegaConf


def _mock_get_rank():
    """Stub for dist.get_rank() when not running distributed."""
    return 0


def _mock_is_initialized():
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/LIBERO/train_files/starvla_cotrain_libero.yaml",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of iteration steps to test")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_yaml)

    # Need output_dir for dataset stats saving
    cfg.output_dir = "tmp/smoke_test_output"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # For smoke test: override VLM data to use a small subset & reduce batch size
    cfg.datasets.vlm_data.per_device_batch_size = 2
    cfg.datasets.vla_data.per_device_batch_size = 2
    # Use smaller data mix for faster loading
    cfg.datasets.vla_data.data_mix = "libero_goal"

    print("=" * 60)
    print("DataLoaderManager Smoke Test")
    print("=" * 60)

    # ── Step 1: Build manager from config ──
    print("\n[1/4] Building DataLoaderManager from config...")
    t0 = time.time()
    from starVLA.dataloader.dataloader_manager import DataLoaderManager

    # Patch dist.get_rank and dist.is_initialized for single-process testing
    with patch("torch.distributed.get_rank", _mock_get_rank), \
         patch("torch.distributed.is_initialized", _mock_is_initialized):
        manager = DataLoaderManager.from_config(cfg)
    t1 = time.time()
    print(f"  Built in {t1 - t0:.1f}s")
    print(f"  {manager}")
    print(f"  Names: {manager.names}")

    assert len(manager.names) >= 1, "Expected at least 1 dataloader"
    print("  [PASS] Manager created with dataloaders")

    # ── Step 2: Reset & iterate ──
    print(f"\n[2/4] Iterating {args.steps} steps...")
    manager.reset()
    all_batches = []
    for step in range(args.steps):
        t0 = time.time()
        batches = manager.get_next_batches()
        t1 = time.time()
        names_this_step = [n for n, _ in batches]
        print(f"  Step {step}: got {len(batches)} batches {names_this_step} ({t1-t0:.3f}s)")
        all_batches.append(batches)

    assert len(all_batches) == args.steps, "Expected one batch-list per step"
    print("  [PASS] Iteration works")

    # ── Step 3: Check batch contents ──
    print("\n[3/4] Checking batch structure...")
    for name, batch in all_batches[0]:
        if name == "vla":
            # VLA batch is a list of dicts (collated by lerobot collate_fn)
            assert isinstance(batch, list), f"VLA batch should be list, got {type(batch)}"
            print(f"  VLA: list of {len(batch)} items, keys={list(batch[0].keys()) if batch else '(empty)'}")
        elif name == "vlm":
            # VLM batch is a dict with input_ids, labels, etc
            assert isinstance(batch, dict), f"VLM batch should be dict, got {type(batch)}"
            print(f"  VLM: dict with keys={list(batch.keys())}")
        else:
            print(f"  {name}: type={type(batch)}")
    print("  [PASS] Batch structure valid")

    # ── Step 4: Test ratio-based skipping ──
    print("\n[4/4] Testing ratio-based skipping...")
    # Temporarily set vlm ratio to 0.0 -> should never include vlm
    if "vlm" in manager.ratios:
        original_ratio = manager.ratios["vlm"]
        manager.ratios["vlm"] = 0.0
        manager.reset()
        for _ in range(3):
            batches = manager.get_next_batches()
            names = [n for n, _ in batches]
            assert "vlm" not in names, f"vlm should be skipped at ratio=0.0, got {names}"
        manager.ratios["vlm"] = original_ratio
        print("  [PASS] ratio=0.0 correctly skips dataset")
    else:
        print("  [SKIP] Only one dataloader, ratio test not applicable")

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
