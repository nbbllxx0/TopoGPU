from __future__ import annotations

import numpy as np

import topogpu as tg


def test_density_filter_preserves_uniform_density() -> None:
    filt = tg.DensityFilter((4, 3, 2), radius=1.5)
    rho = filt.apply(np.ones(4 * 3 * 2))
    assert np.allclose(rho, 1.0, atol=1.0e-12)
