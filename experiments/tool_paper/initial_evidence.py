from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpu_fem.bc_generator import generate_bc
from gpu_fem.presets import get_preset
from gpu_fem.problem_spec import DistributedLoad, EdgeSupport, PointLoad, PointSupport
from gpu_fem.multigrid_v4 import _cupy_fgmres
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _edof_table_3d
from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
from gpu_fem.solver_v2 import MatrixFreeKff, _cupy_pcg


DEFAULT_SOLVE_CASES = ["cantilever_3d", "cantilever_gpu_medium"]
DEFAULT_GEOMETRY_CASES = [
    "cantilever_3d",
    "bridge_3d",
    "mbb_3d",
    "bracket_3d",
    "torsion_3d",
    "cantilever_gpu_medium",
]


def _safe_label(text: str) -> str:
    return (
        text.replace("@", "_")
        .replace("x", "x")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _case_spec(case: str):
    if "@" not in case:
        return _safe_label(case), get_preset(case)
    base, dims_text = case.split("@", 1)
    dims = [int(part) for part in dims_text.lower().split("x")]
    if len(dims) != 3:
        raise ValueError(f"Expected case override like bridge_gpu_medium@96x32x16, got {case}")
    spec = replace(get_preset(base), nelx=dims[0], nely=dims[1], nelz=dims[2])
    return _safe_label(f"{base}_{dims[0]}x{dims[1]}x{dims[2]}"), spec


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _n_levels_for_dims(nelx: int, nely: int, nelz: int) -> int:
    levels = 1
    dims = [nelx, nely, nelz]
    while levels < 4 and all(d % 2 == 0 and d >= 4 for d in dims):
        dims = [d // 2 for d in dims]
        levels += 1
    return levels


def environment_report() -> dict:
    report = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import numpy

        report["numpy"] = numpy.__version__
    except Exception as exc:
        report["numpy_error"] = repr(exc)
    try:
        import scipy

        report["scipy"] = scipy.__version__
    except Exception as exc:
        report["scipy_error"] = repr(exc)
    try:
        import cupy as cp

        dev = cp.cuda.runtime.getDeviceProperties(0)
        name = dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"]
        free, total = cp.cuda.runtime.memGetInfo()
        report.update(
            {
                "cupy": cp.__version__,
                "gpu": name,
                "compute_capability": f"{dev['major']}.{dev['minor']}",
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "vram_total_gb": total / 1024**3,
                "vram_used_gb": (total - free) / 1024**3,
            }
        )
    except Exception as exc:
        report["cupy_error"] = repr(exc)
    return report


def run_uniform_probe(case: str, out_dir: Path, outer: str = "pcg") -> dict:
    import cupy as cp

    case_label, spec = _case_spec(case)
    if not spec.is_3d:
        raise ValueError(f"{case} is not a 3D preset")

    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    free_gpu = cp.asarray(free)
    edof_gpu = cp.asarray(
        _edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32)
    )
    F_free_gpu = cp.asarray(bc.F[free].astype(np.float64))
    n_elem = spec.nelx * spec.nely * spec.nelz
    rho = cp.full(n_elem, 0.5, dtype=cp.float64)
    E_e = 1e-9 + (1.0 - 1e-9) * rho**3.0

    mf_op = MatrixFreeKff(
        edof_gpu=edof_gpu,
        KE_unit_gpu=cp.asarray(KE_UNIT_3D),
        free_gpu=free_gpu,
        n_free=len(free),
        ndof=bc.ndof,
    )
    n_levels = _n_levels_for_dims(spec.nelx, spec.nely, spec.nelz)
    gmg = GalerkinMatFreeGMG(
        mf_op=mf_op,
        free=free,
        free_gpu=free_gpu,
        nelx=spec.nelx,
        nely=spec.nely,
        nelz=spec.nelz,
        KE_UNIT=KE_UNIT_3D,
        n_levels=n_levels,
        fine_smoother="fp64",
        smoother_type="chebyshev",
    )

    cp.cuda.Stream.null.synchronize()
    t_setup0 = time.perf_counter()
    gmg.setup(E_e)
    cp.cuda.Stream.null.synchronize()
    setup_s = time.perf_counter() - t_setup0

    def A_op(v):
        return mf_op.matvec(v, E_e)

    history: list[float] = []
    t_solve0 = time.perf_counter()
    if outer == "fgmres":
        history.append(1.0)
        x, iters, converged = _cupy_fgmres(
            A_op, F_free_gpu, gmg.apply, tol=1e-10, maxiter=300, restart=50
        )
    elif outer == "pcg":
        x, iters, converged = _cupy_pcg(
            A_op, F_free_gpu, gmg.apply, tol=1e-10, maxiter=300, history=history
        )
    else:
        raise ValueError(f"Unsupported outer solver '{outer}'")
    cp.cuda.Stream.null.synchronize()
    solve_s = time.perf_counter() - t_solve0
    rel_res = float(cp.linalg.norm(F_free_gpu - A_op(x))) / float(cp.linalg.norm(F_free_gpu))
    if outer == "fgmres":
        history.append(rel_res)
    compliance = float(cp.dot(F_free_gpu, x))
    free_mem, total_mem = cp.cuda.runtime.memGetInfo()

    hist_path = out_dir / f"{case_label}_residual_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "rel_residual"])
        writer.writeheader()
        for i, value in enumerate(history):
            writer.writerow({"iteration": i, "rel_residual": value})

    level_stats = [
        {
            "level": s.level,
            "kind": s.kind,
            "n_elem": s.n_elem,
            "n_free": s.n_free,
            "nnz": s.nnz,
            "vram_mb": s.estimated_vram_bytes / 1024**2,
        }
        for s in gmg.level_stats
    ]
    return {
        "case": case,
        "case_label": case_label,
        "nelx": spec.nelx,
        "nely": spec.nely,
        "nelz": spec.nelz,
        "n_elem": n_elem,
        "ndof": bc.ndof,
        "n_free": len(free),
        "volfrac": spec.volfrac,
        "n_levels": n_levels,
        "outer": outer,
        "setup_s": setup_s,
        "solve_s": solve_s,
        "iters": iters,
        "converged": bool(converged),
        "rel_residual": rel_res,
        "compliance": compliance,
        "vram_used_gb": (total_mem - free_mem) / 1024**3,
        "residual_history_csv": hist_path.relative_to(ROOT).as_posix(),
        "level_stats": level_stats,
    }


def write_probe_tables(rows: list[dict], out_dir: Path) -> None:
    fieldnames = [
        "case",
        "case_label",
        "nelx",
        "nely",
        "nelz",
        "n_elem",
        "ndof",
        "n_free",
        "volfrac",
        "n_levels",
        "outer",
        "setup_s",
        "solve_s",
        "iters",
        "converged",
        "rel_residual",
        "compliance",
        "vram_used_gb",
        "residual_history_csv",
    ]
    with (out_dir / "uniform_probe_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_residual_histories(rows: list[dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for row in rows:
        path = ROOT / row["residual_history_csv"]
        data = np.genfromtxt(path, delimiter=",", names=True)
        if data.size == 0:
            continue
        iterations = np.atleast_1d(data["iteration"])
        residuals = np.atleast_1d(data["rel_residual"])
        ax.semilogy(iterations, residuals, marker="o", linewidth=1.8, label=row["case"])
    ax.axhline(1e-10, color="0.25", linestyle="--", linewidth=1.0, label="1e-10 gate")
    ax.set_xlabel("PCG iteration")
    ax.set_ylabel("Relative residual")
    ax.set_title("Uniform-density FP64 GMG-PCG residual histories")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "residual_histories.png", dpi=180)
    fig.savefig(out_dir / "residual_histories.pdf")
    plt.close(fig)


def _plot_box(ax, spec):
    x = [0, spec.Lx]
    y = [0, spec.Ly]
    z = [0, spec.Lz]
    corners = np.array(
        [
            [x[0], y[0], z[0]],
            [x[1], y[0], z[0]],
            [x[1], y[1], z[0]],
            [x[0], y[1], z[0]],
            [x[0], y[0], z[1]],
            [x[1], y[0], z[1]],
            [x[1], y[1], z[1]],
            [x[0], y[1], z[1]],
        ]
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color="0.45", linewidth=0.8)


def _support_point(spec, support):
    if isinstance(support, EdgeSupport):
        if support.edge == "left":
            return 0.0, spec.Ly / 2, spec.Lz / 2
        if support.edge == "right":
            return spec.Lx, spec.Ly / 2, spec.Lz / 2
        if support.edge == "bottom":
            return spec.Lx / 2, 0.0, spec.Lz / 2
        if support.edge == "top":
            return spec.Lx / 2, spec.Ly, spec.Lz / 2
        if support.edge == "front":
            return spec.Lx / 2, spec.Ly / 2, 0.0
        if support.edge == "back":
            return spec.Lx / 2, spec.Ly / 2, spec.Lz
    if isinstance(support, PointSupport):
        return support.x, support.y, support.z
    return None


def _load_vector(load):
    if isinstance(load, PointLoad):
        return load.x, load.y, load.z, load.fx, load.fy, load.fz
    if isinstance(load, DistributedLoad):
        return None
    return None


def plot_geometry_matrix(cases: list[str], out_dir: Path) -> list[dict]:
    fig = plt.figure(figsize=(12, 7.2))
    meta = []
    for idx, case in enumerate(cases, start=1):
        case_label, spec = _case_spec(case)
        if not spec.is_3d:
            continue
        bc = generate_bc(spec)
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        _plot_box(ax, spec)
        for support in spec.supports:
            point = _support_point(spec, support)
            if point is not None:
                ax.scatter(*point, marker="s", s=36, color="#1f77b4")
        for load in spec.loads:
            vec = _load_vector(load)
            if vec is not None:
                x, y, z, fx, fy, fz = vec
                ax.quiver(x, y, z, fx, fy, fz, length=0.25, normalize=True, color="#d62728")
            elif isinstance(load, DistributedLoad):
                ax.text(spec.Lx / 2, spec.Ly, spec.Lz, "dist. load", color="#d62728", fontsize=8)
        ax.set_title(f"{case_label}\n{spec.nelx}x{spec.nely}x{spec.nelz}, {spec.nelx * spec.nely * spec.nelz:,} elems", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect((spec.Lx, spec.Ly, spec.Lz))
        meta.append(
            {
                "case": case,
                "case_label": case_label,
                "nelx": spec.nelx,
                "nely": spec.nely,
                "nelz": spec.nelz,
                "n_elem": spec.nelx * spec.nely * spec.nelz,
                "volfrac": spec.volfrac,
                "n_fixed_dofs": int(len(bc.fixed_dofs)),
                "n_free_dofs": int(len(bc.free_dofs)),
                "n_load_entries": int(np.count_nonzero(bc.F)),
            }
        )
    fig.suptitle("Candidate 3D sample matrix: domains, supports, and loads", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "geometry_sample_matrix.png", dpi=180)
    fig.savefig(out_dir / "geometry_sample_matrix.pdf")
    plt.close(fig)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rerun_outputs/tool_paper_initial")
    parser.add_argument("--solve-cases", default=",".join(DEFAULT_SOLVE_CASES))
    parser.add_argument("--geometry-cases", default=",".join(DEFAULT_GEOMETRY_CASES))
    parser.add_argument("--outer", choices=["pcg", "fgmres"], default="pcg")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment_report(),
        "probes": [],
        "geometry_cases": [],
    }
    (out_dir / "environment.json").write_text(
        json.dumps(report["environment"], indent=2, default=_json_default),
        encoding="utf-8",
    )

    solve_cases = [item.strip() for item in args.solve_cases.split(",") if item.strip()]
    for case in solve_cases:
        print(f"[probe] {case}", flush=True)
        row = run_uniform_probe(case, out_dir, outer=args.outer)
        report["probes"].append(row)
        (out_dir / "latest_probe.json").write_text(
            json.dumps(row, indent=2, default=_json_default), encoding="utf-8"
        )
        write_probe_tables(report["probes"], out_dir)
        print(
            f"  n={row['n_elem']:,} setup={row['setup_s']:.2f}s solve={row['solve_s']:.2f}s "
            f"iters={row['iters']} rel_res={row['rel_residual']:.2e}",
            flush=True,
        )

    plot_residual_histories(report["probes"], out_dir)
    geometry_cases = [item.strip() for item in args.geometry_cases.split(",") if item.strip()]
    report["geometry_cases"] = plot_geometry_matrix(geometry_cases, out_dir)

    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "initial_evidence_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"[done] wrote {out_dir.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
