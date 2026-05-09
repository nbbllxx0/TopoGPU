from __future__ import annotations

import topogpu as tg


def test_import_version() -> None:
    assert tg.__version__


def test_gallery_problem_validation() -> None:
    problem = tg.gallery.cantilever_3d(nel=(4, 3, 2), volfrac=0.30)
    problem.validate()
    assert problem.nel == (4, 3, 2)
    assert problem.volfrac == 0.30


def test_density_filter_uniform_density() -> None:
    filt = tg.DensityFilter((4, 3, 2), radius=1.5)
    rho = filt.apply(__import__("numpy").ones(4 * 3 * 2))
    assert abs(float(rho.min()) - 1.0) < 1.0e-12
    assert abs(float(rho.max()) - 1.0) < 1.0e-12

