from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.tool_paper.fast_new_topology_probe import build_problem  # noqa: E402
from gpu_fem.fem_gpu import GPUFEMSolver  # noqa: E402
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d  # noqa: E402
from gpu_fem.solver_v2 import SolverV2  # noqa: E402
from gpu_fem.solver_v4 import SolverV4  # noqa: E402


DEFAULT_CASE_DIMS = {
    "tool_long_cantilever_vf16": "12x6x6",
    "tool_portal_bridge_vf18": "12x6x6",
    "tool_asymmetric_bracket_vf14": "12x8x6",
}

E0 = 1.0
EMIN = 1.0e-9
PENAL = 3.0


def _parse_dims(dims: str) -> tuple[int, int, int]:
    values = tuple(int(x) for x in dims.lower().split("x"))
    if len(values) != 3:
        raise ValueError(f"dims must have form nelx x nely x nelz; got {dims!r}")
    return values


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _density(n_elem: int, volfrac: float) -> np.ndarray:
    idx = np.arange(n_elem, dtype=np.float64)
    pattern = 0.5 + 0.5 * np.sin(0.37 * idx + 0.23 * np.cos(0.17 * idx))
    rho = volfrac + 0.10 * (pattern - pattern.mean())
    return np.clip(rho, 0.08, 0.92).astype(np.float64)


def _assemble_full_k(
    edof: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    ndof: int,
    rho: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    e_elem = EMIN + (E0 - EMIN) * rho**PENAL
    values = (e_elem[:, None] * KE_UNIT_3D.ravel()[None, :]).ravel()
    k_full = sp.csc_matrix((values, (row_idx, col_idx)), shape=(ndof, ndof))
    k_full.sum_duplicates()
    return k_full, e_elem


def _cpu_direct(
    edof: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    free: np.ndarray,
    force: np.ndarray,
    ndof: int,
    rho: np.ndarray,
) -> dict:
    t0 = time.perf_counter()
    k_full, _ = _assemble_full_k(edof, row_idx, col_idx, ndof, rho)
    t1 = time.perf_counter()
    u = np.zeros(ndof, dtype=np.float64)
    u[free] = spla.spsolve(k_full[free][:, free], force[free])
    t2 = time.perf_counter()
    residual = k_full[free][:, free] @ u[free] - force[free]
    rel_res = float(np.linalg.norm(residual) / max(np.linalg.norm(force[free]), 1.0e-300))
    return {
        "variant": "cpu_assembled_direct",
        "backend": "scipy_spsolve",
        "compliance": float(force @ u),
        "relative_residual": rel_res,
        "setup_ms": (t1 - t0) * 1.0e3,
        "solve_ms": (t2 - t1) * 1.0e3,
        "total_ms": (t2 - t0) * 1.0e3,
        "outer_iters": 0,
        "used_gmg": False,
        "matrix_free": False,
        "status": "ok",
        "_u_free": u[free],
    }


def _make_solver(cls, *, edof, row_idx, col_idx, free, force, ndof, dims, variant: str):
    base = dict(
        edof=edof,
        row_idx=row_idx,
        col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D,
        free=free.astype(np.int32),
        F=force,
        ndof=ndof,
        backend="cupy",
        cg_tol=1.0e-8,
        cg_maxiter=1000,
        grid_dims=dims,
        enable_warm_start=True,
        enable_profiling=True,
    )
    if variant == "gpu_assembled_csr_pcg":
        return SolverV2(**base, enable_matrix_free=False)
    if variant == "gpu_matrixfree_jacobi":
        return SolverV2(**base, enable_matrix_free=True, enable_mixed_precision=False)
    if variant == "gpu_matrixfree_gmg":
        return SolverV4(
            **base,
            enable_matrix_free=True,
            enable_fused_cuda=False,
            enable_matfree_gmg=True,
            matfree_gmg_levels=3,
            gmg_fine_smoother="fp64",
            gmg_outer_solver="pcg",
            gmg_smoother_type="chebyshev",
        )
    raise KeyError(variant)


def _gpu_residual(solver, rho: np.ndarray, force_norm: float) -> float | str:
    try:
        import cupy as cp

        u_free = solver._u_prev_cupy
        if u_free is None:
            return ""
        rho_gpu = cp.asarray(rho, dtype=cp.float64)
        e_elem = solver.Emin + (solver.E0 - solver.Emin) * rho_gpu**PENAL
        if getattr(solver, "_enable_matrix_free", False):
            if getattr(solver, "_matfree_op", None) is None and hasattr(solver, "_ensure_matfree_components"):
                solver._ensure_matfree_components()
            if getattr(solver, "_matfree_op", None) is None:
                return ""
            ku = solver._matfree_op.matvec(u_free, e_elem)
        else:
            # Reuse the assembled canonical sparse matrix from the solve path.
            if getattr(solver, "_Kff_gpu", None) is None:
                return ""
            ku = solver._Kff_gpu @ u_free
        r = ku - solver._F_free_gpu
        return float(cp.linalg.norm(r).get() / max(force_norm, 1.0e-300))
    except Exception as exc:
        return f"unavailable: {exc}"


def _gpu_variant(
    variant: str,
    edof: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    free: np.ndarray,
    force: np.ndarray,
    ndof: int,
    dims: tuple[int, int, int],
    rho: np.ndarray,
) -> dict:
    solver = _make_solver(
        SolverV4 if variant == "gpu_matrixfree_gmg" else SolverV2,
        edof=edof,
        row_idx=row_idx,
        col_idx=col_idx,
        free=free,
        force=force,
        ndof=ndof,
        dims=dims,
        variant=variant,
    )
    try:
        t0 = time.perf_counter()
        compliance, _ = solver.solve(rho, PENAL)
        t1 = time.perf_counter()
        force_norm = float(np.linalg.norm(force[free]))
        residual = _gpu_residual(solver, rho, force_norm)
        timing = getattr(solver, "last_timing", {}) or {}
        return {
            "variant": variant,
            "backend": getattr(solver, "_backend", "cupy"),
            "compliance": float(compliance),
            "relative_residual": residual,
            "setup_ms": timing.get("assemble_ms", timing.get("diag_ms", "")),
            "solve_ms": timing.get("cg_ms", ""),
            "total_ms": timing.get("total_ms", (t1 - t0) * 1.0e3),
            "outer_iters": getattr(solver, "last_cg_iters", ""),
            "used_gmg": timing.get("used_gmg", variant == "gpu_matrixfree_gmg"),
            "matrix_free": variant.startswith("gpu_matrixfree"),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "variant": variant,
            "backend": getattr(solver, "_backend", "cupy"),
            "compliance": "",
            "relative_residual": "",
            "setup_ms": "",
            "solve_ms": "",
            "total_ms": "",
            "outer_iters": "",
            "used_gmg": variant == "gpu_matrixfree_gmg",
            "matrix_free": variant.startswith("gpu_matrixfree"),
            "status": f"failed: {exc}",
        }
    finally:
        try:
            solver.free_gpu_cache()
        except Exception:
            pass


def _case_rows(case: str, dims_text: str) -> list[dict]:
    built = build_problem(case, dims_text)
    spec = built["spec"]
    bc = built["bc"]
    dims = _parse_dims(dims_text)
    edof = _edof_table_3d(*dims)
    row_idx, col_idx = _build_sparse_indices(edof)
    free = bc.free_dofs.astype(np.int32)
    rho = _density(edof.shape[0], float(spec.volfrac))
    force = built["F"]
    cpu = _cpu_direct(edof, row_idx, col_idx, free, force, bc.ndof, rho)
    ref_compliance = cpu["compliance"]
    rows = []
    for result in [
        cpu,
        _gpu_variant("gpu_assembled_csr_pcg", edof, row_idx, col_idx, free, force, bc.ndof, dims, rho),
        _gpu_variant("gpu_matrixfree_jacobi", edof, row_idx, col_idx, free, force, bc.ndof, dims, rho),
        _gpu_variant("gpu_matrixfree_gmg", edof, row_idx, col_idx, free, force, bc.ndof, dims, rho),
    ]:
        compliance = result.get("compliance", "")
        if isinstance(compliance, float):
            rel_comp = abs(compliance - ref_compliance) / max(abs(ref_compliance), 1.0e-300)
        else:
            rel_comp = ""
        result.update(
            {
                "case": case,
                "dims": dims_text,
                "n_elem": edof.shape[0],
                "ndof": bc.ndof,
                "n_free": len(free),
                "reference_compliance": ref_compliance,
                "relative_compliance_error": rel_comp,
            }
        )
        result.pop("_u_free", None)
        rows.append(result)
    return rows


def _case_dims_from_args(items: list[str]) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_CASE_DIMS)
    case_dims: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--case-dim entries must have form case=dims")
        case, dims = item.split("=", 1)
        _parse_dims(dims)
        case_dims[case] = dims
    return case_dims


def run(out_dir: Path, case_dims: dict[str, str]) -> dict:
    rows = []
    for case, dims in case_dims.items():
        rows.extend(_case_rows(case, dims))
    fields = [
        "case",
        "dims",
        "n_elem",
        "ndof",
        "n_free",
        "variant",
        "backend",
        "matrix_free",
        "used_gmg",
        "outer_iters",
        "setup_ms",
        "solve_ms",
        "total_ms",
        "reference_compliance",
        "compliance",
        "relative_compliance_error",
        "relative_residual",
        "status",
    ]
    _write_csv(out_dir / "TABLE_BASELINE_ABLATION.csv", rows, fields)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    comp_errors = [
        float(row["relative_compliance_error"])
        for row in ok_rows
        if row["relative_compliance_error"] != ""
    ]
    residuals = [
        float(row["relative_residual"])
        for row in ok_rows
        if isinstance(row["relative_residual"], float)
    ]
    summary = {
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "case_dims": case_dims,
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "failed_rows": len(rows) - len(ok_rows),
        "max_relative_compliance_error": max(comp_errors) if comp_errors else "",
        "max_relative_residual": max(residuals) if residuals else "",
        "variants": sorted({row["variant"] for row in rows}),
    }
    (out_dir / "baseline_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small-mesh CPU/GPU baseline and solver-ablation checks.")
    parser.add_argument("--out", default="rerun_outputs/tool_paper_baselines")
    parser.add_argument("--case-dim", action="append", default=[])
    args = parser.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run(out_dir, _case_dims_from_args(args.case_dim))
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
