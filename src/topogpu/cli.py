"""TopoGPU command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import __version__


ROOT = Path(__file__).resolve().parents[2]


def _run_script(script: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(script), *args]
    return subprocess.call(cmd, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="topogpu")
    parser.add_argument("--version", action="store_true", help="Print package version and exit.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("cite", help="Print citation metadata.")
    sub.add_parser("list-cases", help="List bundled gallery cases.")

    verify_p = sub.add_parser("verify", help="Run numerical verification script.")
    verify_p.add_argument("--out", default="rerun_outputs/topogpu_verify")
    verify_p.add_argument("--case", default="tool_long_cantilever_vf16")
    verify_p.add_argument("--dims", default="6x4x4")

    run_p = sub.add_parser("run", help="Run a small gallery case through the public API.")
    run_p.add_argument("--case", default="cantilever_3d")
    run_p.add_argument("--backend", choices=["cpu", "cuda", "cupy"], default="cpu")
    run_p.add_argument("--iters", type=int, default=3)
    run_p.add_argument("--out", default="runs/cantilever_3d")

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "cite":
        print("TopoGPU: GPU-Accelerated 3D SIMP Topology Optimization in Python")
        print("Version: " + __version__)
        print("Repository: https://github.com/nbbllxx0/TopoGPU")
        print("DOI: pending Zenodo release")
        return 0
    if args.command == "list-cases":
        print("cantilever_3d")
        print("side_load_cantilever")
        print("tool_long_cantilever_vf16")
        print("tool_short_cantilever_vf25")
        print("tool_portal_bridge_vf30")
        print("tool_high_volume_bracket_vf28")
        return 0
    if args.command == "verify":
        script = ROOT / "experiments" / "tool_paper" / "verify_numerics.py"
        return _run_script(script, ["--out", args.out, "--case-dim", f"{args.case}={args.dims}"])
    if args.command == "run":
        from .gallery import CaseGallery
        from .solver import SIMPSolver

        if args.case == "cantilever_3d":
            problem = CaseGallery.cantilever_3d(nel=(8, 4, 4), volfrac=0.30)
        else:
            problem = CaseGallery.tool_case(args.case)
        result = SIMPSolver(backend=args.backend, max_iter=args.iters).solve(problem)
        result.save(args.out)
        print(f"wrote {args.out}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
