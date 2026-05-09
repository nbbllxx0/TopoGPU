from __future__ import annotations

from topogpu.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["benchmark", "cases/production_suite.yaml", "--out", "runs/production_suite"]))
