# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Implemented by DataLoader Manager design.

"""
DataLoaderManager: centralized management of heterogeneous dataloaders.

Sits between individual dataset modules (lerobot_datasets, vlm_datasets, ...)
and the trainer, providing:
  - Unified iteration over multiple dataloaders
  - Configurable per-dataset sampling ratios
  - Automatic epoch reset / StopIteration handling
  - accelerator.prepare() integration for distributed training
"""

import logging
import random
from typing import Any

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class DataLoaderManager:
    """
    Manages multiple named DataLoaders with per-dataset sampling ratios.

    Each call to ``get_next_batches()`` returns a list of ``(name, batch)``
    tuples.  The trainer can iterate over that list to forward each batch
    through the corresponding model head and accumulate gradients.

    Example::

        manager = DataLoaderManager.from_config(cfg)
        manager.prepare(accelerator)
        manager.reset()

        for step in range(max_steps):
            batches = manager.get_next_batches()
            for name, batch in batches:
                ...
    """

    def __init__(
        self,
        dataloaders: dict[str, DataLoader],
        ratios: dict[str, float] | None = None,
    ):
        """
        Args:
            dataloaders: ``{"vla": dl_vla, "vlm": dl_vlm, ...}``
            ratios: per-dataset inclusion probability per step.
                    ``{"vla": 1.0, "vlm": 0.5}`` means VLA is always
                    sampled while VLM is included with 50 % probability.
                    Defaults to 1.0 for every loader.
        """
        self.dataloaders: dict[str, DataLoader] = dataloaders
        self.ratios: dict[str, float] = ratios or {k: 1.0 for k in dataloaders}
        self._iterators: dict[str, Any] = {}
        self._epoch_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "DataLoaderManager":
        """Build from an OmegaConf config (compatible with existing YAML).

        Iterates over ``cfg.datasets.*_data`` entries, calls
        ``build_dataloader`` for each one that has a ``dataset_py`` field,
        and collects them into a single manager.
        """
        from starVLA.dataloader import build_dataloader

        dataloaders: dict[str, DataLoader] = {}
        ratios: dict[str, float] = {}

        for ds_key in cfg.datasets:
            ds_cfg = cfg.datasets[ds_key]
            # Skip entries set to null (e.g. ``datasets.vla_data=null`` via CLI)
            if ds_cfg is None:
                continue
            dataset_py = ds_cfg.get("dataset_py", None)
            if dataset_py is None:
                continue
            dl = build_dataloader(cfg=cfg, dataset_py=dataset_py)
            # "vla_data" -> "vla", "vlm_data" -> "vlm"
            name = ds_key.replace("_data", "")
            dataloaders[name] = dl
            ratios[name] = float(ds_cfg.get("ratio", 1.0))
            logger.info(f"DataLoaderManager: registered '{name}' (dataset_py={dataset_py}, ratio={ratios[name]})")

        return cls(dataloaders=dataloaders, ratios=ratios)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """(Re-)create iterators for all dataloaders and zero epoch counts."""
        for name, dl in self.dataloaders.items():
            self._iterators[name] = iter(dl)
            self._epoch_counts[name] = 0

    def prepare(self, accelerator) -> "DataLoaderManager":
        """Wrap every dataloader with ``accelerator.prepare``."""
        prepared: dict[str, DataLoader] = {}
        for name, dl in self.dataloaders.items():
            prepared[name] = accelerator.prepare(dl)
        self.dataloaders = prepared
        return self

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def _get_single_batch(self, name: str) -> Any:
        """Fetch next batch for *name*, handling epoch reset transparently."""
        try:
            return next(self._iterators[name])
        except StopIteration:
            self._epoch_counts[name] += 1
            dl = self.dataloaders[name]
            if hasattr(dl, "sampler") and callable(getattr(dl.sampler, "set_epoch", None)):
                dl.sampler.set_epoch(self._epoch_counts[name])
            self._iterators[name] = iter(dl)
            logger.info(f"DataLoaderManager: '{name}' epoch reset → epoch {self._epoch_counts[name]}")
            return next(self._iterators[name])

    def get_next_batches(self, allowed_names: set[str] | None = None) -> list[tuple[str, Any]]:
        """Return ``[(name, batch), ...]`` for this training step.

        Each dataset is included based on its configured ratio (probability).
        A ratio of 1.0 guarantees inclusion; 0.5 means ~50 % of steps.
        """
        batches: list[tuple[str, Any]] = []
        for name in self.dataloaders:
            if allowed_names is not None and name not in allowed_names:
                continue
            ratio = self.ratios.get(name, 1.0)
            if ratio < 1.0 and random.random() > ratio:
                continue
            batch = self._get_single_batch(name)
            batches.append((name, batch))
        return batches

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Ordered list of registered dataset names."""
        return list(self.dataloaders.keys())

    @property
    def epoch_counts(self) -> dict[str, int]:
        return dict(self._epoch_counts)

    def __repr__(self) -> str:
        items = ", ".join(f"{n}(ratio={self.ratios.get(n, 1.0)})" for n in self.names)
        return f"DataLoaderManager([{items}])"
