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
from gpu_fem.local_agents import PureFEMRouter
from gpu_fem.presets import get_preset
from gpu_fem.pub_baseline_controller import ScheduleOnlyController
from gpu_fem.simp_gpu import TO3DParams, run_simp_surrogate_gpu
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


def solver_config(label: str, preset: str, spec) -> tuple[type, dict]:
    is_cantilever = preset.startswith("cantilever")
    hard_outer = "pcg" if is_cantilever else "fgmres"
    base = dict(
        grid_dims=(spec.nelx, spec.nely, spec.nelz),
        enable_warm_start=True,
        enable_matrix_free=True,
        enable_fused_cuda=True,
    )
    if label == "paper3_fused_fp32_jacobi":
        return SolverV2, {**base, "enable_mixed_precision": True, "fused_dtype": "fp32"}
    if label == "paper4_gmg_fp32":
        return SolverV4, {
            **base,
            "enable_matfree_gmg": True,
            "gmg_fine_smoother": "fp32",
            "gmg_fine_degree": 2,
            "gmg_outer_solver": hard_outer,
        }
    if label == "paper4_gmg_bf16":
        return SolverV4, {
            **base,
            "enable_matfree_gmg": True,
            "gmg_fine_smoother": "bf16",
            "gmg_fine_degree": 2,
            "gmg_outer_solver": "fgmres",
        }
    raise ValueError(label)


def run_case(preset: str, label: str, n_iters: int) -> dict:
    prob = build_problem(preset)
    spec = prob["spec"]
    params = TO3DParams(
        nelx=spec.nelx,
        nely=spec.nely,
        nelz=spec.nelz,
        volfrac=spec.volfrac,
        rmin=spec.rmin if spec.rmin is not None else 1.5,
        max_iter=n_iters,
    )
    solver_class, solver_kwargs = solver_config(label, preset, spec)
    t0 = time.perf_counter()
    error = ""
    try:
        result = run_simp_surrogate_gpu(
            params=params,
            fixed=prob["fixed"],
            free=prob["free"],
            F=prob["F"],
            ndof=prob["ndof"],
            surrogate=None,
            router=PureFEMRouter(),
            device="auto",
            param_controller=ScheduleOnlyController(),
            verbose=False,
            solver_class=solver_class,
            solver_kwargs=solver_kwargs,
        )
        wall_s = time.perf_counter() - t0
    except Exception as exc:
        result = {}
        wall_s = time.perf_counter() - t0
        error = f"{type(exc).__name__}: {exc}"
    return {
        "preset": preset,
        "path": label,
        "n_iters": n_iters,
        "n_elem": spec.nelx * spec.nely * spec.nelz,
        "wall_s": wall_s,
        "best_compliance": float(result.get("best_compliance", result.get("final_compliance", float("nan")))),
        "final_compliance": float(result.get("final_compliance", float("nan"))),
        "vram_gb": _vram_gb(),
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", default="cantilever_gpu_medium,mbb_gpu_large,bridge_gpu_large")
    parser.add_argument("--iters", default="60,120")
    args = parser.parse_args()

    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    iters = [int(x.strip()) for x in args.iters.split(",") if x.strip()]
    paths = ["paper3_fused_fp32_jacobi", "paper4_gmg_fp32", "paper4_gmg_bf16"]

    rows = []
    for preset in presets:
        for n_iter in iters:
            print(f"\n[{preset} / SIMP-{n_iter}]")
            for label in paths:
                row = run_case(preset, label, n_iter)
                rows.append(row)
                print(
                    f"  {label:24s} wall={row['wall_s']:.3f}s "
                    f"best={row['best_compliance']:.6f} vram={row['vram_gb']:.3f}GB"
                )

    out_dir = ROOT / "experiments" / "paper4"
    out_csv = out_dir / "benchmark_simp_paper4.csv"
    out_json = out_dir / "benchmark_simp_paper4.json"
    fieldnames = [
        "preset",
        "path",
        "n_iters",
        "n_elem",
        "wall_s",
        "best_compliance",
        "final_compliance",
        "vram_gb",
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
