# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Streaming normalization statistics accumulator.

Uses Welford's online algorithm for mean/variance, running min/max,
and per-dimension t-digests for quantile estimation. Memory usage is
bounded regardless of dataset size.
"""

from __future__ import annotations

import numpy as np
from tdigest import TDigest


class StreamingStatsAccumulator:
    """Accumulates per-dimension statistics from arbitrarily many batches.

    Tracks mean, std (population), min, max, q01, and q99 for each feature
    dimension using streaming algorithms that never hold the full dataset
    in memory.

    Usage:
        acc = StreamingStatsAccumulator()
        for batch in data_source:          # batch: np.ndarray [N, D]
            acc.update(batch)
        stats = acc.finalize()             # {mean, std, min, max, q01, q99}
    """

    def __init__(self, digest_delta: float = 0.01, digest_k: int = 25) -> None:
        """
        Args:
            digest_delta: Compression parameter for t-digest. Smaller values
                give more accurate quantile estimates at the cost of memory.
            digest_k: Size parameter for t-digest centroid merging.
        """
        self._digest_delta = digest_delta
        self._digest_k = digest_k
        self._count: int = 0
        self._mean: np.ndarray | None = None
        self._m2: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._digests: list[TDigest] | None = None

    @property
    def count(self) -> int:
        """Total number of observations seen so far."""
        return self._count

    def update(self, batch: np.ndarray) -> None:
        """Incorporate a batch of observations.

        Args:
            batch: Array of shape [N, D] or [N]. If 1-D, treated as [N, 1].
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
        batch = batch.astype(np.float64)
        n_rows, n_dims = batch.shape
        if n_rows == 0:
            return

        if self._mean is not None and n_dims != len(self._mean):
            raise ValueError(
                f"Dimension mismatch: accumulator has {len(self._mean)} dims "
                f"but batch has {n_dims}"
            )

        # First batch: initialize all accumulators
        if self._mean is None:
            self._mean = np.zeros(n_dims, dtype=np.float64)
            self._m2 = np.zeros(n_dims, dtype=np.float64)
            self._min = np.full(n_dims, np.inf, dtype=np.float64)
            self._max = np.full(n_dims, -np.inf, dtype=np.float64)
            self._digests = [
                TDigest(delta=self._digest_delta, K=self._digest_k)
                for _ in range(n_dims)
            ]

        # --- Running min/max ---
        self._min = np.minimum(self._min, np.min(batch, axis=0))
        self._max = np.maximum(self._max, np.max(batch, axis=0))

        # --- Welford's parallel/batch merge (Chan et al. 1979) ---
        batch_count = n_rows
        batch_mean = np.mean(batch, axis=0)
        batch_m2 = np.sum((batch - batch_mean) ** 2, axis=0)

        new_count = self._count + batch_count
        delta = batch_mean - self._mean
        new_mean = self._mean + delta * (batch_count / new_count)
        new_m2 = (
            self._m2
            + batch_m2
            + delta ** 2 * (self._count * batch_count / new_count)
        )

        self._count = new_count
        self._mean = new_mean
        self._m2 = new_m2

        # --- t-digest quantile tracking ---
        for dim in range(n_dims):
            self._digests[dim].batch_update(batch[:, dim].tolist())

    def finalize(self) -> dict[str, list[float]]:
        """Compute final statistics and return as JSON-serializable dict.

        Returns:
            Dict with keys: mean, std, min, max, q01, q99.
            Each value is a list of floats with length D (feature dimensions).

        Raises:
            ValueError: If no data was ever fed via update().
        """
        if self._count == 0 or self._mean is None:
            raise ValueError("No data has been fed to the accumulator.")

        std = np.sqrt(self._m2 / self._count)
        n_dims = len(self._digests)

        return {
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "q01": [float(self._digests[d].percentile(1)) for d in range(n_dims)],
            "q99": [float(self._digests[d].percentile(99)) for d in range(n_dims)],
        }
