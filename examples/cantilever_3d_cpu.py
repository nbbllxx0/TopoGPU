from __future__ import annotations

from pathlib import Path

import topogpu as tg


def main() -> int:
    problem = tg.gallery.cantilever_3d(nel=(8, 4, 4), volfrac=0.30, filter_radius=1.5)
    result = tg.SIMPSolver(backend="cpu", max_iter=3).solve(problem)
    result.save(Path("runs/cantilever_3d_cpu"))
    print("final summary written to runs/cantilever_3d_cpu/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
