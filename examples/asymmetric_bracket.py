from __future__ import annotations

from pathlib import Path

import topogpu as tg


def main() -> int:
    problem = tg.gallery.tool_case("tool_high_volume_bracket_vf28", dims="16x8x6")
    result = tg.SIMPSolver(backend="cpu", max_iter=2).solve(problem)
    result.save(Path("runs/asymmetric_bracket"))
    print("final summary written to runs/asymmetric_bracket/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
