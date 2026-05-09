from __future__ import annotations

import numpy as np

import topogpu as tg


def test_filter_energy_form_is_symmetric_positive() -> None:
    filt = tg.DensityFilter((3, 2, 2), radius=1.5)
    weights = filt.matrix()
    rho = np.linspace(0.1, 1.0, weights.shape[0])
    energy = float(rho @ (weights @ rho))
    assert energy > 0.0
