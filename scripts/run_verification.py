from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(ROOT / "experiments" / "tool_paper" / "verify_numerics.py"),
                "--out",
                "rerun_outputs/topogpu_verify",
                "--case-dim",
                "tool_long_cantilever_vf16=6x4x4",
            ],
            cwd=ROOT,
        )
    )
