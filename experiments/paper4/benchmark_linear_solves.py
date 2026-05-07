from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _prefer_pytorch_env() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_tmp = root / ".runtime_tmp"
    cupy_cache = root / ".cupy_cache"
    runtime_tmp.mkdir(exist_ok=True)
    cupy_cache.mkdir(exist_ok=True)
    os.environ["TMP"] = str(runtime_tmp)
    os.environ["TEMP"] = str(runtime_tmp)
    os.environ["CUPY_CACHE_DIR"] = str(cupy_cache)
    os.environ["CUPY_TEMPDIR"] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)



_prefer_pytorch_env()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from gpu_fem.bc_generator import generate_bc
from gpu_fem.presets import get_preset
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d
from gpu_fem.solver_v2 import SolverV2
from gpu_fem.solver_v4 import SolverV4


def _vram_gb() -> float:
    try:
        import cupy as cp

        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / 1024**3
    except Exception:
        return float("nan")


def build_problem(name: str) -> dict:
    spec = get_preset(name)
    bc = generate_bc(spec)
    return {
        "spec": spec,
        "fixed": bc.fixed_dofs.astype(np.int32),
        "free": bc.free_dofs.astype(np.int32),
        "F": bc.F,
        "ndof": bc.ndof,
    }


def make_solver(label: str, preset: str, prob: dict):
    spec = prob["spec"]
    edof = _edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    is_cantilever = preset.startswith("cantilever")
    hard_outer = "pcg" if is_cantilever else "fgmres"
    common = dict(
        edof=edof,
        row_idx=row_idx,
        col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D,
        free=prob["free"],
        F=prob["F"],
        ndof=prob["ndof"],
        backend="auto",
        grid_dims=(spec.nelx, spec.nely, spec.nelz),
        enable_warm_start=False,
        enable_matrix_free=True,
    )
    if label == "paper3_fused_fp32_jacobi":
        return SolverV2(
            **common,
            enable_mixed_precision=True,
            enable_fused_cuda=True,
            fused_dtype="fp32",
        )
    if label == "paper4_gmg_fp32":
        return SolverV4(
            **common,
            enable_matfree_gmg=True,
            enable_fused_cuda=True,
            gmg_fine_smoother="fp32",
            gmg_fine_degree=2,
            gmg_outer_solver=hard_outer,
        )
    if label == "paper4_gmg_bf16":
        return SolverV4(
            **common,
            enable_matfree_gmg=True,
            enable_fused_cuda=True,
            gmg_fine_smoother="bf16",
            gmg_fine_degree=2,
            gmg_outer_solver="fgmres",
        )
    raise ValueError(label)


def run_case(preset: str, label: str) -> dict:
    prob = build_problem(preset)
    solver = make_solver(label, preset, prob)
    n_elem = prob["spec"].nelx * prob["spec"].nely * prob["spec"].nelz
    rho = np.full(n_elem, 0.5)
    try:
        t0 = time.perf_counter()
        compliance, _ = solver.solve(rho, penal=3.0)
        wall_s = time.perf_counter() - t0
        error = ""
    except Exception as exc:
        wall_s = time.perf_counter() - t0
        compliance = float("nan")
        error = f"{type(exc).__name__}: {exc}"
    row = {
        "preset": preset,
        "path": label,
        "n_elem": n_elem,
        "wall_s": wall_s,
        "cg_iters": getattr(solver, "last_cg_iters", -1),
        "compliance": float(compliance),
        "vram_gb": _vram_gb(),
        "error": error,
    }
    if hasattr(solver, "last_outer_solver"):
        row["outer_solver"] = solver.last_outer_solver
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--presets",
        default="cantilever_gpu_medium,mbb_gpu_large,bridge_gpu_large",
        help="Comma-separated preset list.",
    )
    args = parser.parse_args()

    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    paths = ["paper3_fused_fp32_jacobi", "paper4_gmg_fp32", "paper4_gmg_bf16"]
    rows = []
    for preset in presets:
        print(f"\n[{preset}]")
        for label in paths:
            row = run_case(preset, label)
            rows.append(row)
            print(
                f"  {label:24s} wall={row['wall_s']:.3f}s cg={row['cg_iters']} "
                f"c={row['compliance']:.6f} vram={row['vram_gb']:.3f}GB"
            )

    out_dir = ROOT / "experiments" / "paper4"
    out_csv = out_dir / "benchmark_linear_solves.csv"
    out_json = out_dir / "benchmark_linear_solves.json"
    fieldnames = [
        "preset",
        "path",
        "n_elem",
        "wall_s",
        "cg_iters",
        "compliance",
        "vram_gb",
        "outer_solver",
        "error",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fieldnames} for row in rows])
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out_csv.relative_to(ROOT)}")
    print(f"Saved {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
