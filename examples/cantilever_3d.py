from __future__ import annotations

import argparse
from pathlib import Path

import topogpu as tg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="Use a tiny CPU smoke-test mesh.")
    parser.add_argument("--backend", choices=["cpu", "cuda", "cupy"], default="cpu")
    parser.add_argument("--out", default="runs/cantilever_3d")
    args = parser.parse_args()

    nel = (8, 4, 4) if args.small else (24, 12, 6)
    problem = tg.gallery.cantilever_3d(nel=nel, volfrac=0.30, filter_radius=1.5)
    result = tg.SIMPSolver(backend=args.backend, max_iter=3 if args.small else 12).solve(problem)
    result.save(Path(args.out))
    print(f"final summary written to {Path(args.out) / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

