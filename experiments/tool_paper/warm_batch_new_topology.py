from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpu_fem.problem_spec import ProblemSpec  # noqa: E402
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d  # noqa: E402
from gpu_fem.solver_v4 import SolverV4  # noqa: E402

from experiments.tool_paper.fast_new_topology_probe import (  # noqa: E402
    CASES,
    _gpu_snapshot,
    _grayness,
    _linear_relative_residual,
    _oc_update,
    _plot_outputs,
    _write_history,
    build_problem,
)


DEFAULT_BATCH = [
    ("tool_long_cantilever_vf16", "96x48x48"),
    ("tool_portal_bridge_vf18", "112x56x40"),
    ("tool_asymmetric_bracket_vf14", "96x64x40"),
]


def _parse_case_spec(text: str) -> tuple[str, str | None]:
    if "@" not in text:
        return text, None
    case, dims = text.split("@", 1)
    return case.strip(), dims.strip()


def _dims_to_text(dims: tuple[int, int, int]) -> str:
    return f"{dims[0]}x{dims[1]}x{dims[2]}"


def _free_cupy_pool() -> None:
    try:
        import cupy as cp

        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        return


def _save_displacement_scalar(
    solver: SolverV4,
    bc,
    edof: np.ndarray,
    dims: tuple[int, int, int],
    out_dir: Path,
) -> str | None:
    """Save element-averaged displacement magnitude for colored surface renders."""

    u_free = getattr(solver, "_u_prev_cupy", None)
    if u_free is None:
        return None
    try:
        u_free_np = np.asarray(u_free.get(), dtype=np.float64)
    except AttributeError:
        u_free_np = np.asarray(u_free, dtype=np.float64)

    full = np.zeros(int(bc.ndof), dtype=np.float64)
    free = np.asarray(bc.free_dofs, dtype=np.int64)
    full[free] = u_free_np
    node_mag = np.linalg.norm(full.reshape(-1, 3), axis=1)
    elem_nodes = (edof.reshape(edof.shape[0], 8, 3)[:, :, 0] // 3).astype(np.int64)
    elem_disp = node_mag[elem_nodes].mean(axis=1).reshape(dims)
    out_path = out_dir / "disp_elem.npy"
    np.save(out_path, elem_disp)
    return str(out_path.relative_to(ROOT)).replace("\\", "/")


def _make_solver(spec: ProblemSpec, bc, F, edof: np.ndarray, cg_tol: float, cg_maxiter: int) -> SolverV4:
    row_idx, col_idx = _build_sparse_indices(edof)
    return SolverV4(
        edof=edof,
        row_idx=row_idx,
        col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D,
        free=bc.free_dofs.astype(np.int32),
        F=F,
        ndof=bc.ndof,
        backend="cupy",
        enable_warm_start=True,
        enable_matrix_free=True,
        enable_fused_cuda=True,
        enable_matfree_gmg=True,
        matfree_gmg_levels=4,
        gmg_fine_smoother="fp32",
        gmg_outer_solver=_MAKE_SOLVER_OUTER_SOLVER,
        gmg_restart=_MAKE_SOLVER_RESTART,
        gmg_smoother_type="chebyshev",
        grid_dims=(spec.nelx, spec.nely, spec.nelz),
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        enable_profiling=True,
    )


_MAKE_SOLVER_OUTER_SOLVER = "auto"
_MAKE_SOLVER_RESTART = 50


def run_case(
    case_name: str,
    dims_text: str | None,
    out_root: Path,
    iters: int,
    move: float,
    cg_tol: float,
    cg_maxiter: int,
    max_gpu_mem_gb: float,
    render: bool,
    outer_solver: str,
    restart: int,
    rho_min: float,
) -> dict:
    global _MAKE_SOLVER_OUTER_SOLVER, _MAKE_SOLVER_RESTART
    _MAKE_SOLVER_OUTER_SOLVER = outer_solver
    _MAKE_SOLVER_RESTART = restart
    built = build_problem(case_name, dims_text)
    spec: ProblemSpec = built["spec"]
    bc = built["bc"]
    F = built["F"]
    dims = (spec.nelx, spec.nely, spec.nelz)
    n_elem = spec.nelx * spec.nely * spec.nelz
    out_dir = out_root / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    edof = _edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    solver = _make_solver(spec, bc, F, edof, cg_tol=cg_tol, cg_maxiter=cg_maxiter)
    rho = np.full(n_elem, spec.volfrac, dtype=np.float64)
    history: list[dict] = []

    print(
        f"case={case_name} dims={_dims_to_text(dims)} n_elem={n_elem} "
        f"vf={spec.volfrac} iters={iters} cg_tol={cg_tol:g} cg_maxiter={cg_maxiter} "
        f"outer_solver={outer_solver}",
        flush=True,
    )
    gpu_start = _gpu_snapshot()
    print(f"gpu_start={gpu_start}", flush=True)
    total0 = time.perf_counter()

    for iteration in range(1, iters + 1):
        t0 = time.perf_counter()
        compliance, dc = solver.solve(rho, penal=3.0)
        linear_residual = _linear_relative_residual(solver, rho, penal=3.0)
        wall_s = time.perf_counter() - t0
        rho = _oc_update(rho, dc, spec.volfrac, move, rho_min=rho_min)
        gpu = _gpu_snapshot()
        row = {
            "iteration": iteration,
            "compliance": float(compliance),
            "outer_iters": int(getattr(solver, "last_cg_iters", -1)),
            "linear_relative_residual": linear_residual,
            "linear_residual_available": linear_residual is not None,
            "rho_mean": float(rho.mean()),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "grayness": _grayness(rho),
            "wall_s": wall_s,
            "outer_solver": getattr(solver, "last_outer_solver", ""),
            "gpu_mem_used_mb": gpu.get("gpu_mem_used_mb"),
            "gpu_mem_total_mb": gpu.get("gpu_mem_total_mb"),
            "gpu_util_pct": gpu.get("gpu_util_pct"),
            "gpu_temp_c": gpu.get("gpu_temp_c"),
        }
        history.append(row)
        _write_history(out_dir / "history.csv", history)
        cap = " CAP" if row["outer_iters"] >= cg_maxiter else ""
        print(
            f"iter {iteration:03d}: C={row['compliance']:.6g} "
            f"iters={row['outer_iters']}{cap} gray={row['grayness']:.4f} "
            f"relres={linear_residual if linear_residual is not None else 'NA'} "
            f"wall={wall_s:.2f}s gpu_mem={row['gpu_mem_used_mb']}MB "
            f"gpu_util={row['gpu_util_pct']}%",
            flush=True,
        )
        used_mb = row.get("gpu_mem_used_mb")
        if used_mb is not None and used_mb > max_gpu_mem_gb * 1024:
            raise RuntimeError(
                f"GPU memory guard tripped for {case_name}: "
                f"{used_mb} MB > {max_gpu_mem_gb:.1f} GB"
            )

    rho_path = out_dir / "rho_final.npy"
    np.save(rho_path, rho)
    disp_path = _save_displacement_scalar(solver, bc, edof, dims, out_dir)
    render_meta = {} if not render else _plot_outputs(out_dir, rho, history, dims)
    warm_walls = [float(r["wall_s"]) for r in history[1:]]
    cap_count = sum(1 for r in history if int(r["outer_iters"]) >= cg_maxiter)
    summary = {
        "case": case_name,
        "dims": list(dims),
        "n_elem": n_elem,
        "volfrac": spec.volfrac,
        "iters": iters,
        "total_wall_s": time.perf_counter() - total0,
        "first_iter_wall_s": float(history[0]["wall_s"]),
        "warm_iter_mean_wall_s_excluding_first": float(np.mean(warm_walls)) if warm_walls else None,
        "warm_iter_max_wall_s_excluding_first": float(np.max(warm_walls)) if warm_walls else None,
        "linear_solve_cap_count": cap_count,
        "final": history[-1],
        "rho_final_npy": str(rho_path.relative_to(ROOT)).replace("\\", "/"),
        "disp_elem_npy": disp_path,
        "history_csv": str((out_dir / "history.csv").relative_to(ROOT)).replace("\\", "/"),
        "render_png": str((out_dir / "render.png").relative_to(ROOT)).replace("\\", "/") if render else None,
        "projection_png": str((out_dir / "density_projections.png").relative_to(ROOT)).replace("\\", "/") if render else None,
        "render_meta": render_meta,
        "backend": {
            "solver": "SolverV4",
            "enable_matrix_free": True,
            "enable_fused_cuda": True,
            "enable_matfree_gmg": True,
            "matfree_gmg_levels": 4,
            "gmg_fine_smoother": "fp32",
            "gmg_smoother_type": "chebyshev",
            "gmg_outer_solver": outer_solver,
            "gmg_restart": restart,
            "cg_tol": cg_tol,
            "cg_maxiter": cg_maxiter,
            "grid_dims": list(dims),
            "move_limit": move,
            "rho_min": rho_min,
        },
        "gpu_start": gpu_start,
        "gpu_end": _gpu_snapshot(),
        "note": (
            "Single-process warm-batch SolverV4 fused CUDA + matrix-free GMG run. "
            "The first case in the Python process may still include per-problem GMG construction; "
            "warm iteration fields exclude the first optimization iteration."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rerun_outputs/tool_paper_new_topology_solverv4_warm_large")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--move", type=float, default=0.10)
    parser.add_argument("--rho-min", type=float, default=1.0e-3)
    parser.add_argument("--cg-tol", type=float, default=1e-6)
    parser.add_argument("--cg-maxiter", type=int, default=800)
    parser.add_argument("--outer-solver", choices=["auto", "pcg", "fgmres"], default="auto")
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--max-gpu-mem-gb", type=float, default=24.0)
    parser.add_argument("--warmup-case", default="tool_long_cantilever_vf16@48x24x24")
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--case", action="append", help="Case spec, e.g. tool_long_cantilever_vf16@96x48x48")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--free-between", action="store_true", default=True)
    args = parser.parse_args()

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    if args.case:
        batch = [_parse_case_spec(item) for item in args.case]
    else:
        batch = DEFAULT_BATCH

    run_meta = {
        "out": str(out_root.relative_to(ROOT)).replace("\\", "/"),
        "started_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "available_cases": sorted(CASES),
        "requested_cases": [{"case": c, "dims": d} for c, d in batch],
        "warmup_case": args.warmup_case,
        "settings": {
            "iters": args.iters,
            "move": args.move,
            "rho_min": args.rho_min,
            "cg_tol": args.cg_tol,
            "cg_maxiter": args.cg_maxiter,
            "max_gpu_mem_gb": args.max_gpu_mem_gb,
            "render": not args.no_render,
            "free_between": args.free_between,
        },
        "gpu_before": _gpu_snapshot(),
        "warmup": None,
        "summaries": [],
    }

    if args.warmup_case:
        warm_case, warm_dims = _parse_case_spec(args.warmup_case)
        warm_root = out_root / "_warmup"
        print(f"warmup={warm_case}@{warm_dims}", flush=True)
        run_meta["warmup"] = run_case(
            warm_case,
            warm_dims,
            warm_root,
            args.warmup_iters,
            args.move,
            args.cg_tol,
            args.cg_maxiter,
            args.max_gpu_mem_gb,
            render=False,
            outer_solver=args.outer_solver,
            restart=args.restart,
            rho_min=args.rho_min,
        )
        if args.free_between:
            gc.collect()
            _free_cupy_pool()

    for case_name, dims_text in batch:
        summary = run_case(
            case_name,
            dims_text,
            out_root,
            args.iters,
            args.move,
            args.cg_tol,
            args.cg_maxiter,
            args.max_gpu_mem_gb,
            render=not args.no_render,
            outer_solver=args.outer_solver,
            restart=args.restart,
            rho_min=args.rho_min,
        )
        run_meta["summaries"].append(summary)
        (out_root / "batch_summary.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        if args.free_between:
            gc.collect()
            _free_cupy_pool()

    run_meta["gpu_after"] = _gpu_snapshot()
    run_meta["completed_wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_root / "batch_summary.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
