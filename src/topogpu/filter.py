"""Density-filter utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DensityFilter:
    nel: tuple[int, int, int]
    radius: float

    def matrix(self) -> np.ndarray:
        nelx, nely, nelz = self.nel
        ix, iy, iz = np.meshgrid(
            np.arange(nelx),
            np.arange(nely),
            np.arange(nelz),
            indexing="ij",
        )
        coords = np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1).astype(float)
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        weights = np.maximum(0.0, self.radius - dist)
        row_sums = weights.sum(axis=1)
        row_sums[row_sums == 0.0] = 1.0
        return weights / row_sums[:, None]

    def apply(self, rho: np.ndarray) -> np.ndarray:
        return self.matrix() @ rho
