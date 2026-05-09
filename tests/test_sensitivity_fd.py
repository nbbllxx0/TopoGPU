from __future__ import annotations

import numpy as np


def test_quadratic_sensitivity_finite_difference() -> None:
    rho = np.array([0.2, 0.4, 0.6], dtype=float)
    direction = np.array([1.0, -0.5, 0.25], dtype=float)
    eps = 1.0e-6
    fd = ((rho + eps * direction) @ (rho + eps * direction) - rho @ rho) / eps
    exact = 2.0 * float(rho @ direction)
    assert abs(fd - exact) < 1.0e-5
