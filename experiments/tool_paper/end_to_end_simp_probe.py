from __future__ import annotations

import argparse
import csv
import json
import math
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
from gpu_fem.pub_simp_solver import SIMPParams, run_simp  # noqa: E402


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _write_history(path: Path, history: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "compliance"])
        writer.writeheader()
        for idx, compliance in enumerate(history, start=1):
            writer.writerow({"iteration": idx, "compliance": compliance})


def _plot_history(path: Path, history: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(range(1, len(history) + 1), history, marker="o", linewidth=1.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Compliance")
    ax.set_title("Fresh small 3D SIMP probe")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_density_projections(path: Path, rho: np.ndarray, nelx: int, nely: int, nelz: int) -> None:
    volume = rho.reshape(nelx, nely, nelz)
    projections = [
        ("max over z", volume.max(axis=2).T),
        ("max over y", volume.max(axis=1).T),
        ("max over x", volume.max(axis=0).T),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for ax, (title, image) in zip(axes, projections):
        im = ax.imshow(image, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.68])
    fig.colorbar(im, cax=cax)
    fig.suptitle("Fresh small 3D SIMP density projections")
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _render_density(path: Path, rho: np.ndarray, nelx: int, nely: int, nelz: int) -> dict:
    volume = rho.reshape(nelx, nely, nelz)
    lo = float(volume.min())
    hi = float(volume.max())
    if hi - lo < 1e-8:
        raise ValueError("Density field is nearly uniform; marching-cubes render is not meaningful.")
    level = 0.5
    if not (lo < level < hi):
        level = lo + 0.65 * (hi - lo)
    mesh = density_to_polydata(volume, level=level, taubin_iters=4, taubin_pass_band=0.2)
    render_panel(mesh, nelx, nely, nelz, path)
    return {
        "iso_level": level,
        "density_min": lo,
        "density_max": hi,
        "mesh_points": int(mesh.n_points),
        "mesh_cells": int(mesh.n_cells),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rerun_outputs/tool_paper_end_to_end_simp")
    parser.add_argument("--nelx", type=int, default=24)
    parser.add_argument("--nely", type=int, default=12)
    parser.add_argument("--nelz", type=int, default=6)
    parser.add_argument("--volfrac", type=float, default=0.3)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--rmin", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    params = SIMPParams(
        nelx=args.nelx,
        nely=args.nely,
        nelz=args.nelz,
        volfrac=args.volfrac,
        rmin=args.rmin,
        max_iter=args.max_iter,
        min_iter=args.max_iter + 1,
        seed=args.seed,
        checkpoint_dir=None,
    )

    started = time.perf_counter()
    result = run_simp(params, verbose=True, problem="cantilever")
    wall_s = time.perf_counter() - started

    rho_final = np.asarray(result["rho_final"], dtype=np.float64)
    rho_path = out_dir / "fresh_cantilever_3d_rho_final.npy"
    np.save(rho_path, rho_final)
    _write_history(out_dir / "compliance_history.csv", result["compliance_history"])
    _plot_history(out_dir / "compliance_history", result["compliance_history"])
    _plot_density_projections(
        out_dir / "density_projection_gallery", rho_final, args.nelx, args.nely, args.nelz
    )

    render_meta = {}
    render_error = None
    try:
        render_meta = _render_density(
            out_dir / "fresh_cantilever_3d_render.png",
            rho_final,
            args.nelx,
            args.nely,
            args.nelz,
        )
    except Exception as exc:
        render_error = repr(exc)

    summary = {
        "case": "fresh_cantilever_3d_simp_probe",
        "nelx": args.nelx,
        "nely": args.nely,
        "nelz": args.nelz,
        "n_elem": args.nelx * args.nely * args.nelz,
        "volfrac_target": args.volfrac,
        "max_iter": args.max_iter,
        "n_iter_reported": result["n_iter"],
        "wall_s": wall_s,
        "final_compliance": result["final_compliance"],
        "final_grayness": result["final_grayness"],
        "final_volume_fraction": float(rho_final.mean()),
        "density_min": float(rho_final.min()),
        "density_max": float(rho_final.max()),
        "best_is_valid": bool(result["best_is_valid"]),
        "best_compliance": result["best_compliance"]
        if math.isfinite(float(result["best_compliance"]))
        else None,
        "best_iteration": result["best_iteration"],
        "rho_final_npy": str(rho_path.relative_to(ROOT)).replace("\\", "/"),
        "history_csv": str((out_dir / "compliance_history.csv").relative_to(ROOT)).replace("\\", "/"),
        "projection_png": str((out_dir / "density_projection_gallery.png").relative_to(ROOT)).replace("\\", "/"),
        "render_png": str((out_dir / "fresh_cantilever_3d_render.png").relative_to(ROOT)).replace("\\", "/")
        if render_error is None
        else "",
        "render_error": render_error,
        "render_meta": render_meta,
        "note": "Bounded small 3D SIMP provenance probe; not a final optimized topology or benchmark result.",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0 if render_error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
