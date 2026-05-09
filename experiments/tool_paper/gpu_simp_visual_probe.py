from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
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
from gpu_fem.bc_generator import generate_bc  # noqa: E402
from gpu_fem.presets import get_preset  # noqa: E402
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d  # noqa: E402
from gpu_fem.solver_v4 import SolverV4  # noqa: E402


def _case_spec(case: str):
    if "@" not in case:
        return case, get_preset(case)
    base, dims_text = case.split("@", 1)
    dims = [int(part) for part in dims_text.lower().split("x")]
    if len(dims) != 3:
        raise ValueError(f"Expected case override like cantilever_gpu_medium@96x48x24, got {case}")
    return f"{base}_{dims[0]}x{dims[1]}x{dims[2]}", replace(
        get_preset(base), nelx=dims[0], nely=dims[1], nelz=dims[2]
    )


def _oc_update(rho: np.ndarray, dc: np.ndarray, volfrac: float, move: float = 0.12) -> np.ndarray:
    dc_safe = np.minimum(dc, -1e-12)
    lam_lo, lam_hi = 0.0, 1.0

    def candidate(lam: float) -> np.ndarray:
        return np.clip(
            rho * np.sqrt(np.maximum(-dc_safe / max(lam, 1e-40), 0.0)),
            np.maximum(rho - move, 1e-3),
            np.minimum(rho + move, 1.0),
        )

    while candidate(lam_hi).mean() > volfrac and lam_hi < 1e40:
        lam_lo = lam_hi
        lam_hi *= 2.0

    for _ in range(100):
        lmid = 0.5 * (lam_lo + lam_hi)
        rho_new = candidate(lmid)
        if rho_new.mean() > volfrac:
            lam_lo = lmid
        else:
            lam_hi = lmid
        if abs(rho_new.mean() - volfrac) < 1e-8:
            break
    return rho_new


def _grayness(rho: np.ndarray) -> float:
    return float(4.0 * np.mean(rho * (1.0 - rho)))


def _write_history(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "iteration",
        "compliance",
        "outer_iters",
        "rho_mean",
        "rho_min",
        "rho_max",
        "grayness",
        "wall_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(out_dir: Path, rows: list[dict]) -> None:
    it = [row["iteration"] for row in rows]
    compliance = [row["compliance"] for row in rows]
    grayness = [row["grayness"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(it, compliance, marker="o", linewidth=1.6)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Compliance")
    axes[0].set_title("GPU SIMP compliance")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[1].plot(it, grayness, marker="o", linewidth=1.6)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Grayness")
    axes[1].set_title("Density grayness")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_dir / "gpu_simp_history.png", dpi=180)
    fig.savefig(out_dir / "gpu_simp_history.pdf")
    plt.close(fig)


def _plot_density_projections(out_dir: Path, rho: np.ndarray, nelx: int, nely: int, nelz: int) -> None:
    volume = rho.reshape(nelx, nely, nelz)
    projections = [
        ("max over z", volume.max(axis=2).T),
        ("max over y", volume.max(axis=1).T),
        ("max over x", volume.max(axis=0).T),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (title, image) in zip(axes, projections):
        im = ax.imshow(image, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.68])
    fig.colorbar(im, cax=cax)
    fig.suptitle(f"GPU SIMP density projections ({nelx}x{nely}x{nelz})")
    fig.savefig(out_dir / "gpu_simp_density_projections.png", dpi=180)
    fig.savefig(out_dir / "gpu_simp_density_projections.pdf")
    plt.close(fig)


def _render_density(out_dir: Path, rho: np.ndarray, nelx: int, nely: int, nelz: int) -> dict:
    volume = rho.reshape(nelx, nely, nelz)
    level = 0.5
    lo, hi = float(volume.min()), float(volume.max())
    if not (lo < level < hi):
        level = lo + 0.65 * (hi - lo)
    mesh = density_to_polydata(volume, level=level, taubin_iters=6, taubin_pass_band=0.15)
    render_panel(mesh, nelx, nely, nelz, out_dir / "gpu_simp_render.png")
    return {
        "iso_level": level,
        "mesh_points": int(mesh.n_points),
        "mesh_cells": int(mesh.n_cells),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="cantilever_gpu_medium")
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--out", default="rerun_outputs/tool_paper_gpu_simp_visual_64k")
    parser.add_argument("--move", type=float, default=0.12)
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    case_label, spec = _case_spec(args.case)
    if not spec.is_3d:
        raise ValueError(f"{args.case} is not a 3D case")

    bc = generate_bc(spec)
    edof = _edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    free = bc.free_dofs.astype(np.int32)
    n_elem = spec.nelx * spec.nely * spec.nelz
    common_kwargs = dict(
        edof=edof,
        row_idx=row_idx,
        col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D,
        free=free,
        F=bc.F,
        ndof=bc.ndof,
        backend="cupy",
        enable_warm_start=True,
    )
    solver = SolverV4(
        enable_matrix_free=True,
        enable_fused_cuda=True,
        enable_matfree_gmg=True,
        matfree_gmg_levels=4,
        gmg_fine_smoother="fp32",
        gmg_smoother_type="chebyshev",
        grid_dims=(spec.nelx, spec.nely, spec.nelz),
        **common_kwargs,
    )

    rho = np.full(n_elem, spec.volfrac, dtype=np.float64)
    history = []
    t_total0 = time.perf_counter()
    for k in range(1, args.iters + 1):
        t0 = time.perf_counter()
        compliance, dc = solver.solve(rho, penal=3.0)
        wall_s = time.perf_counter() - t0
        rho = _oc_update(rho, dc, spec.volfrac, move=args.move)
        row = {
            "iteration": k,
            "compliance": float(compliance),
            "outer_iters": int(getattr(solver, "last_cg_iters", -1)),
            "rho_mean": float(rho.mean()),
            "rho_min": float(rho.min()),
            "rho_max": float(rho.max()),
            "grayness": _grayness(rho),
            "wall_s": wall_s,
        }
        history.append(row)
        _write_history(out_dir / "gpu_simp_history.csv", history)
        print(
            f"iter {k:03d}: C={row['compliance']:.6g} "
            f"iters={row['outer_iters']} gray={row['grayness']:.3f} "
            f"rho=[{row['rho_min']:.3g},{row['rho_max']:.3g}] wall={wall_s:.2f}s",
            flush=True,
        )

    rho_path = out_dir / "gpu_simp_rho_final.npy"
    np.save(rho_path, rho)
    _plot_history(out_dir, history)
    _plot_density_projections(out_dir, rho, spec.nelx, spec.nely, spec.nelz)
    render_meta = _render_density(out_dir, rho, spec.nelx, spec.nely, spec.nelz)

    summary = {
        "case": args.case,
        "case_label": case_label,
        "nelx": spec.nelx,
        "nely": spec.nely,
        "nelz": spec.nelz,
        "n_elem": n_elem,
        "volfrac": spec.volfrac,
        "iters": args.iters,
        "total_wall_s": time.perf_counter() - t_total0,
        "final_compliance_solve": history[-1]["compliance"],
        "final_outer_iters": history[-1]["outer_iters"],
        "final_grayness": history[-1]["grayness"],
        "final_rho_mean": history[-1]["rho_mean"],
        "final_rho_min": history[-1]["rho_min"],
        "final_rho_max": history[-1]["rho_max"],
        "rho_final_npy": str(rho_path.relative_to(ROOT)).replace("\\", "/"),
        "history_csv": str((out_dir / "gpu_simp_history.csv").relative_to(ROOT)).replace("\\", "/"),
        "render_png": str((out_dir / "gpu_simp_render.png").relative_to(ROOT)).replace("\\", "/"),
        "projection_png": str((out_dir / "gpu_simp_density_projections.png").relative_to(ROOT)).replace("\\", "/"),
        "render_meta": render_meta,
        "note": "GPU SolverV4 SIMP visual probe using the paper4 OC loop; no density filter/projection polish.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
