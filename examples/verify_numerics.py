from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    script = ROOT / "experiments" / "tool_paper" / "verify_numerics.py"
    return subprocess.call(
        [
            sys.executable,
            str(script),
            "--out",
            "rerun_outputs/topogpu_verify",
            "--case-dim",
            "tool_long_cantilever_vf16=6x4x4",
        ],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
