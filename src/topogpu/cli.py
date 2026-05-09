"""TopoGPU command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from . import __version__


ROOT = Path(__file__).resolve().parents[2]


def _run_script(script: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(script), *args]
    return subprocess.call(cmd, cwd=ROOT)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - packaging dependency guard
        raise SystemExit("PyYAML is required for YAML case files. Reinstall with `pip install -e .`.") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML file {path} must contain a mapping.")
    return data


def _problem_from_yaml(path: Path, small: bool = False):
    from .gallery import CaseGallery

    data = _load_yaml(path)
    name = str(data.get("name", path.stem))
    nel = tuple(int(v) for v in data.get("nel", [8, 4, 4] if small else [24, 12, 6]))
    if len(nel) != 3:
        raise SystemExit(f"Case {path} must define nel as three integers.")
    if small:
        nel = (min(nel[0], 8), min(nel[1], 4), min(nel[2], 4))
    if name == "cantilever_3d":
        return CaseGallery.cantilever_3d(
            nel=nel,
            volfrac=float(data.get("volfrac", 0.30)),
            filter_radius=float(data.get("filter_radius", data.get("rmin", 1.5))),
            support=str(data.get("support", "xmin")),
            load=str(data.get("load", "tip_patch_z")),
        )
    return CaseGallery.tool_case(name)


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
    verify_p.add_argument("--small", action="store_true", help="Use the default small verification mesh.")

    run_p = sub.add_parser("run", help="Run a small gallery case through the public API.")
    run_p.add_argument("target", nargs="?", help="Optional YAML case path, e.g. cases/cantilever_3d.yaml.")
    run_p.add_argument("--case", default="cantilever_3d")
    run_p.add_argument("--backend", choices=["cpu", "cuda", "cupy"], default="cpu")
    run_p.add_argument("--iters", type=int, default=3)
    run_p.add_argument("--out", default="runs/cantilever_3d")
    run_p.add_argument("--small", action="store_true", help="Clamp YAML mesh dimensions to a smoke-test size.")

    render_p = sub.add_parser("render", help="Write render metadata for a saved evidence bundle.")
    render_p.add_argument("run_dir")
    render_p.add_argument("--threshold", type=float, default=0.5)

    bench_p = sub.add_parser("benchmark", help="Create a suite manifest for a YAML benchmark suite.")
    bench_p.add_argument("suite")
    bench_p.add_argument("--out", default="runs/benchmark_suite")

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "cite":
        print("TopoGPU: GPU-Accelerated 3D SIMP Topology Optimization in Python")
        print("Version: " + __version__)
        print("Repository: https://github.com/nbbllxx0/TopoGPU")
        print("DOI: 10.5281/zenodo.20100693")
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
        dims = "6x4x4" if args.small else args.dims
        return _run_script(script, ["--out", args.out, "--case-dim", f"{args.case}={dims}"])
    if args.command == "run":
        from .gallery import CaseGallery
        from .solver import SIMPSolver

        if args.target:
            target = Path(args.target)
            if not target.is_absolute():
                target = ROOT / target
            problem = _problem_from_yaml(target, small=args.small)
        elif args.case == "cantilever_3d":
            problem = CaseGallery.cantilever_3d(nel=(8, 4, 4), volfrac=0.30)
        else:
            problem = CaseGallery.tool_case(args.case)
        result = SIMPSolver(backend=args.backend, max_iter=args.iters).solve(problem)
        result.save(args.out)
        print(f"wrote {args.out}")
        return 0
    if args.command == "render":
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "density_threshold": args.threshold,
            "scalar": "density",
            "scalar_normalization": "per_case_0_1",
            "source": "topogpu render",
        }
        path = run_dir / "render_metadata.json"
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"wrote {path}")
        return 0
    if args.command == "benchmark":
        suite = Path(args.suite)
        if not suite.is_absolute():
            suite = ROOT / suite
        data = _load_yaml(suite)
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "suite": data.get("name", suite.stem),
                "case": case,
                "role": data.get("role", "production_candidate"),
                "status": "declared",
            }
            for case in data.get("cases", [])
        ]
        csv_path = out / "benchmark_suite.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["suite", "case", "role", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
