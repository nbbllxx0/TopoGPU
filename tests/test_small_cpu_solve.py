from __future__ import annotations

import topogpu as tg


def test_small_cpu_solve_writes_summary(tmp_path) -> None:
    problem = tg.gallery.cantilever_3d(nel=(4, 3, 2), volfrac=0.30)
    result = tg.SIMPSolver(backend="cpu", max_iter=1).solve(problem)
    bundle = result.save(tmp_path / "run")
    bundle.write_manifest()
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "render_metadata.json").exists()
