from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from figures.make_3d_renders import density_to_polydata, render_panel  # noqa: E402
from gpu_fem.bc_generator import generate_bc, _face_nodes_3d, _node_coords_3d  # noqa: E402
from gpu_fem.problem_spec import EdgeSupport, PointLoad, ProblemSpec  # noqa: E402
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d  # noqa: E402
from gpu_fem.solver_v4 import SolverV4  # noqa: E402


CASES = {
    "tool_long_cantilever_vf16": {
        "dims": (72, 36, 36),
        "volfrac": 0.16,
        "rmin": 2.4,
        "kind": "patch_cantilever",
        "load_rel": (1.0, 0.50, 0.18),
        "patch_radius_rel": 0.18,
        "load_vector": (0.0, 0.0, -1.0),
    },
    "tool_portal_bridge_vf18": {
        "dims": (80, 40, 32),
        "volfrac": 0.18,
        "rmin": 2.8,
        "kind": "portal_bridge",
    },
    "tool_asymmetric_bracket_vf14": {
        "dims": (72, 48, 32),
        "volfrac": 0.14,
        "rmin": 2.4,
        "kind": "asym_bracket",
        "load_rel": (1.0, 0.72, 0.28),
        "patch_radius_rel": 0.16,
        "load_vector": (0.0, -0.7, -0.7),
    },
    "tool_short_cantilever_vf25": {
        "dims": (48, 32, 24),
        "volfrac": 0.25,
        "rmin": 2.2,
        "kind": "patch_cantilever",
        "load_rel": (1.0, 0.50, 0.50),
        "patch_radius_rel": 0.20,
        "load_vector": (0.0, 0.0, -1.0),
        "role": "production_timing_candidate",
    },
    "tool_deep_cantilever_vf20": {
        "dims": (64, 48, 32),
        "volfrac": 0.20,
        "rmin": 2.4,
        "kind": "patch_cantilever",
        "load_rel": (1.0, 0.50, 0.35),
        "patch_radius_rel": 0.18,
        "load_vector": (0.0, 0.0, -1.0),
        "role": "production_timing_candidate",
    },
    "tool_oblique_cantilever_vf22": {
        "dims": (64, 40, 32),
        "volfrac": 0.22,
        "rmin": 2.4,
        "kind": "patch_cantilever",
        "load_rel": (1.0, 0.55, 0.35),
        "patch_radius_rel": 0.18,
        "load_vector": (0.0, -0.6, -0.8),
        "role": "production_timing_candidate",
    },
    "tool_side_load_cantilever_vf24": {
        "dims": (56, 36, 28),
        "volfrac": 0.24,
        "rmin": 2.3,
        "kind": "patch_cantilever",
        "load_rel": (1.0, 0.62, 0.50),
        "patch_radius_rel": 0.20,
        "load_vector": (0.0, -1.0, 0.0),
        "role": "production_timing_candidate",
    },
    "tool_dual_load_cantilever_vf26": {
        "dims": (56, 36, 28),
        "volfrac": 0.26,
        "rmin": 2.3,
        "kind": "patch_cantilever",
        "patch_loads": [
            {
                "load_rel": (1.0, 0.35, 0.65),
                "patch_radius_rel": 0.16,
                "load_vector": (0.0, -0.45, -0.55),
            },
            {
                "load_rel": (1.0, 0.72, 0.30),
                "patch_radius_rel": 0.14,
                "load_vector": (0.0, 0.35, -0.65),
            },
        ],
        "role": "production_timing_candidate",
    },
    "tool_high_volume_bracket_vf28": {
        "dims": (72, 48, 32),
        "volfrac": 0.28,
        "rmin": 2.4,
        "kind": "asym_bracket",
        "load_rel": (1.0, 0.72, 0.28),
        "patch_radius_rel": 0.18,
        "load_vector": (0.0, -0.55, -0.45),
        "role": "non_cantilever_candidate",
    },
    "tool_portal_bridge_vf30": {
        "dims": (80, 40, 32),
        "volfrac": 0.30,
        "rmin": 2.8,
        "kind": "portal_bridge",
        "role": "non_cantilever_candidate",
    },
}


def _override_dims(case: dict, dims: str | None) -> dict:
    case = dict(case)
    if dims:
        parsed = tuple(int(x) for x in dims.lower().split("x"))
        if len(parsed) != 3:
            raise ValueError("--dims must have form nelx x nely x nelz, e.g. 72x36x36")
        case["dims"] = parsed
    return case


def _patch_load_problem(cfg: dict, kind: str) -> dict:
    nelx, nely, nelz = cfg["dims"]
    if kind == "asym_bracket":
        Lx, Ly, Lz = 2.0, 1.25, 0.75
    else:
        Lx, Ly, Lz = 2.4, 1.0, 1.0
    first_patch = cfg.get("patch_loads", [{}])[0]
    load_rel_for_spec = first_patch.get("load_rel", cfg.get("load_rel", (1.0, 0.5, 0.5)))
    center_y_for_spec = float(load_rel_for_spec[1]) * Ly
    center_z_for_spec = float(load_rel_for_spec[2]) * Lz
    spec = ProblemSpec(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=cfg["volfrac"],
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=Lx, y=center_y_for_spec, z=center_z_for_spec, fz=-1e-12)],
        rmin=cfg["rmin"],
    )
    bc = generate_bc(spec)
    coords = _node_coords_3d(nelx, nely, nelz, Lx, Ly, Lz)
    face_nodes = _face_nodes_3d("right", nelx, nely, nelz)

    F = np.zeros_like(bc.F)
    yz = coords[face_nodes][:, 1:3]
    patch_loads = cfg.get(
        "patch_loads",
        [
            {
                "load_rel": cfg.get("load_rel", (1.0, 0.5, 0.5)),
                "patch_radius_rel": cfg.get("patch_radius_rel", 0.18),
                "load_vector": cfg.get("load_vector", (0.0, 0.0, -1.0)),
            }
        ],
    )
    for patch in patch_loads:
        load_rel = patch.get("load_rel", (1.0, 0.5, 0.5))
        center_y = float(load_rel[1]) * Ly
        center_z = float(load_rel[2]) * Lz
        radius = float(patch.get("patch_radius_rel", 0.18)) * min(Ly, Lz)
        d = np.linalg.norm(yz - np.array([center_y, center_z])[None, :], axis=1)
        patch_nodes = face_nodes[d <= radius]
        if patch_nodes.size == 0:
            patch_nodes = face_nodes[np.argsort(d)[:12]]
        load = np.array(patch.get("load_vector", (0.0, 0.0, -1.0)), dtype=float)
        load = load / float(len(patch_nodes))
        for node in patch_nodes:
            base = 3 * int(node)
            F[base + 0] += load[0]
            F[base + 1] += load[1]
            F[base + 2] += load[2]
    return {"spec": spec, "bc": bc, "F": F}


def _portal_bridge_problem(cfg: dict) -> dict:
    nelx, nely, nelz = cfg["dims"]
    Lx, Ly, Lz = float(nelx), float(nely), float(nelz)
    spec = ProblemSpec(
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=cfg["volfrac"],
        supports=[
            EdgeSupport(edge="left", constraint="fixed"),
            EdgeSupport(edge="right", constraint="roller_x"),
        ],
        loads=[PointLoad(x=Lx * 0.52, y=Ly, z=Lz * 0.5, fy=-1.0)],
        rmin=cfg["rmin"],
    )
    bc = generate_bc(spec)
    return {"spec": spec, "bc": bc, "F": bc.F}


def build_problem(case_name: str, dims: str | None = None) -> dict:
    if case_name not in CASES:
        raise KeyError(f"Unknown case {case_name}. Available: {', '.join(CASES)}")
    cfg = _override_dims(CASES[case_name], dims)
    if cfg["kind"] in {"patch_cantilever", "asym_bracket"}:
        built = _patch_load_problem(cfg, cfg["kind"])
    elif cfg["kind"] == "portal_bridge":
        built = _portal_bridge_problem(cfg)
    else:
        raise ValueError(cfg["kind"])
    built["cfg"] = cfg
    return built


def _oc_update(
    rho: np.ndarray,
    dc: np.ndarray,
    volfrac: float,
    move: float,
    rho_min: float = 1.0e-3,
) -> np.ndarray:
    dc_safe = np.minimum(dc, -1e-12)
    lam_lo, lam_hi = 0.0, 1.0

    def candidate(lam: float) -> np.ndarray:
        return np.clip(
            rho * np.sqrt(np.maximum(-dc_safe / max(lam, 1e-40), 0.0)),
            np.maximum(rho - move, rho_min),
            np.minimum(rho + move, 1.0),
        )

    while candidate(lam_hi).mean() > volfrac and lam_hi < 1e40:
        lam_lo = lam_hi
        lam_hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lam_lo + lam_hi)
        rho_new = candidate(mid)
        if rho_new.mean() > volfrac:
            lam_lo = mid
        else:
            lam_hi = mid
        if abs(float(rho_new.mean()) - volfrac) < 1e-8:
            break
    return rho_new


def _grayness(rho: np.ndarray) -> float:
    return float(4.0 * np.mean(rho * (1.0 - rho)))


def _gpu_snapshot() -> dict:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        first = proc.stdout.strip().splitlines()[0]
        used, total, util, temp = [x.strip() for x in first.split(",")]
        return {
            "gpu_mem_used_mb": int(float(used)),
            "gpu_mem_total_mb": int(float(total)),
            "gpu_util_pct": int(float(util)),
            "gpu_temp_c": int(float(temp)),
        }
    except Exception as exc:
        return {"gpu_monitor_error": str(exc)}


def _write_history(path: Path, rows: list[dict]) -> None:
    fields = [
        "iteration",
        "compliance",
        "outer_iters",
        "linear_relative_residual",
        "linear_residual_available",
        "rho_mean",
        "rho_min",
        "rho_max",
        "grayness",
        "wall_s",
        "outer_solver",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "gpu_util_pct",
        "gpu_temp_c",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _linear_relative_residual(solver, rho: np.ndarray, penal: float) -> float | None:
    try:
        import cupy as cp

        u_free = getattr(solver, "_u_prev_cupy", None)
        mf_op = getattr(solver, "_matfree_op", None)
        if u_free is None or mf_op is None:
            return None
        rho_gpu = cp.asarray(rho, dtype=cp.float64)
        e_elem = solver.Emin + (solver.E0 - solver.Emin) * rho_gpu**penal
        residual = mf_op.matvec(u_free, e_elem) - solver._F_free_gpu
        denom = max(float(cp.linalg.norm(solver._F_free_gpu).get()), 1.0e-300)
        return float(cp.linalg.norm(residual).get()) / denom
    except Exception:
        return None


def _plot_outputs(out_dir: Path, rho: np.ndarray, rows: list[dict], dims: tuple[int, int, int]) -> dict:
    nelx, nely, nelz = dims
    volume = rho.reshape(nelx, nely, nelz)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot([r["iteration"] for r in rows], [r["compliance"] for r in rows], marker="o")
    axes[0].set_title("Compliance")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot([r["iteration"] for r in rows], [r["grayness"] for r in rows], marker="o")
    axes[1].set_title("Grayness")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=180)
    plt.close(fig)

    projections = [volume.max(axis=2).T, volume.max(axis=1).T, volume.max(axis=0).T]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, image in zip(axes, projections):
        ax.imshow(image, origin="lower", cmap="gray_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "density_projections.png", dpi=180)
    plt.close(fig)

    level = 0.5
    if not (float(volume.min()) < level < float(volume.max())):
        level = float(volume.min()) + 0.65 * (float(volume.max()) - float(volume.min()))
    mesh = density_to_polydata(volume, level=level, taubin_iters=5, taubin_pass_band=0.15)
    render_panel(mesh, nelx, nely, nelz, out_dir / "render.png")
    return {"iso_level": level, "mesh_points": int(mesh.n_points), "mesh_cells": int(mesh.n_cells)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--dims", default=None)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--move", type=float, default=0.10)
    parser.add_argument("--rho-min", type=float, default=1.0e-3)
    parser.add_argument("--out", default="rerun_outputs/tool_paper_new_topology_fast")
    parser.add_argument("--cg-tol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=400)
    parser.add_argument("--max-gpu-mem-gb", type=float, default=28.0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    out_dir = ROOT / args.out / args.case
    out_dir.mkdir(parents=True, exist_ok=True)

    built = build_problem(args.case, args.dims)
    spec: ProblemSpec = built["spec"]
    bc = built["bc"]
    F = built["F"]
    dims = (spec.nelx, spec.nely, spec.nelz)
    n_elem = spec.nelx * spec.nely * spec.nelz

    edof = _edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    solver = SolverV4(
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
        gmg_smoother_type="chebyshev",
        grid_dims=dims,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        enable_profiling=True,
    )

    rho = np.full(n_elem, spec.volfrac, dtype=np.float64)
    history: list[dict] = []
    total0 = time.perf_counter()
    print(
        f"case={args.case} dims={spec.nelx}x{spec.nely}x{spec.nelz} "
        f"n_elem={n_elem} vf={spec.volfrac} iters={args.iters} "
        "backend=SolverV4+fused_cuda+matfree_gmg",
        flush=True,
    )
    gpu0 = _gpu_snapshot()
    print(f"gpu_start={gpu0}", flush=True)
    for iteration in range(1, args.iters + 1):
        t0 = time.perf_counter()
        compliance, dc = solver.solve(rho, penal=3.0)
        linear_residual = _linear_relative_residual(solver, rho, penal=3.0)
        wall_s = time.perf_counter() - t0
        rho = _oc_update(rho, dc, spec.volfrac, args.move, rho_min=args.rho_min)
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
        print(
            f"iter {iteration:03d}: C={row['compliance']:.6g} "
            f"iters={row['outer_iters']} gray={row['grayness']:.3f} "
            f"relres={linear_residual if linear_residual is not None else 'NA'} "
            f"rho=[{row['rho_min']:.3g},{row['rho_max']:.3g}] wall={wall_s:.2f}s "
            f"gpu_mem={row['gpu_mem_used_mb']}MB gpu_util={row['gpu_util_pct']}%",
            flush=True,
        )
        used_mb = row.get("gpu_mem_used_mb")
        if used_mb is not None and used_mb > args.max_gpu_mem_gb * 1024:
            raise RuntimeError(
                f"GPU memory guard tripped: {used_mb} MB > {args.max_gpu_mem_gb:.1f} GB"
            )

    rho_path = out_dir / "rho_final.npy"
    np.save(rho_path, rho)
    render_meta = {} if args.no_render else _plot_outputs(out_dir, rho, history, dims)
    warm_walls = [r["wall_s"] for r in history[1:]]
    summary = {
        "case": args.case,
        "dims": list(dims),
        "n_elem": n_elem,
        "volfrac": spec.volfrac,
        "iters": args.iters,
        "total_wall_s": time.perf_counter() - total0,
        "first_iter_wall_s": history[0]["wall_s"],
        "warm_iter_mean_wall_s_excluding_first": float(np.mean(warm_walls)) if warm_walls else None,
        "warm_iter_max_wall_s_excluding_first": float(np.max(warm_walls)) if warm_walls else None,
        "final": history[-1],
        "rho_final_npy": str(rho_path.relative_to(ROOT)).replace("\\", "/"),
        "render_png": None if args.no_render else str((out_dir / "render.png").relative_to(ROOT)).replace("\\", "/"),
        "projection_png": None if args.no_render else str((out_dir / "density_projections.png").relative_to(ROOT)).replace("\\", "/"),
        "history_csv": str((out_dir / "history.csv").relative_to(ROOT)).replace("\\", "/"),
        "render_meta": render_meta,
        "backend": {
            "solver": "SolverV4",
            "enable_matrix_free": True,
            "enable_fused_cuda": True,
            "enable_matfree_gmg": True,
            "matfree_gmg_levels": 4,
            "gmg_fine_smoother": "fp32",
            "gmg_smoother_type": "chebyshev",
            "cg_tol": args.cg_tol,
            "cg_maxiter": args.cg_maxiter,
            "grid_dims": list(dims),
            "move_limit": args.move,
            "rho_min": args.rho_min,
        },
        "gpu_start": gpu0,
        "gpu_end": _gpu_snapshot(),
        "note": "New tool-paper topology exploration run using the Phase 4-style SolverV4 fused CUDA + matrix-free GMG path. Timing is exploratory unless a separate cold/warm protocol is used.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
