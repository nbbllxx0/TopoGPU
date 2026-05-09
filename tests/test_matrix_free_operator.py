from __future__ import annotations

import numpy as np

import topogpu as tg


def test_density_filter_matrix_is_row_normalized() -> None:
    filt = tg.DensityFilter((3, 2, 2), radius=1.5)
    weights = filt.matrix()
    assert weights.shape == (12, 12)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-12)
