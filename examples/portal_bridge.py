from __future__ import annotations

from pathlib import Path

import topogpu as tg


def main() -> int:
    problem = tg.gallery.tool_case("tool_portal_bridge_vf30", dims="16x8x6")
    result = tg.SIMPSolver(backend="cpu", max_iter=2).solve(problem)
    result.save(Path("runs/portal_bridge"))
    print("final summary written to runs/portal_bridge/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
