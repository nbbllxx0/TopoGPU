from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(os.environ.get("TOPOGPU_RUN_CUDA_TESTS") != "1", reason="CUDA smoke test is opt-in.")
def test_small_cuda_solve_opt_in(tmp_path) -> None:
    import topogpu as tg

    problem = tg.gallery.cantilever_3d(nel=(4, 3, 2), volfrac=0.30)
    result = tg.SIMPSolver(backend="cuda", max_iter=1, max_krylov=50).solve(problem)
    result.save(tmp_path / "cuda_run")
    assert (tmp_path / "cuda_run" / "summary.json").exists()
