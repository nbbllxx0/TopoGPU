from __future__ import annotations

from topogpu.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "run",
                "cases/cantilever_3d.yaml",
                "--small",
                "--backend",
                "cpu",
                "--iters",
                "2",
                "--out",
                "runs/custom_yaml_case",
            ]
        )
    )
