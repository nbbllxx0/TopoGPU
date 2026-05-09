from __future__ import annotations

from pathlib import Path

from topogpu.cli import main


def main_script() -> int:
    for run_dir in Path("runs").glob("*"):
        if run_dir.is_dir():
            main(["render", str(run_dir)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main_script())
