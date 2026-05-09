from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.make_3d_renders import density_to_polydata  # noqa: E402


def _load_summary(case_dir: Path) -> dict:
    path = case_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_point_scalars(points: np.ndarray, scalar_volume: np.ndarray) -> np.ndarray:
    idx = np.rint(points).astype(np.int64)
    idx[:, 0] = np.clip(idx[:, 0], 0, scalar_volume.shape[0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, scalar_volume.shape[1] - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, scalar_volume.shape[2] - 1)
    return scalar_volume[idx[:, 0], idx[:, 1], idx[:, 2]]


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    lo = float(np.percentile(finite, 2.0))
    hi = float(np.percentile(finite, 98.0))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _render_mesh(mesh, dims: tuple[int, int, int], out_png: Path, scalar_name: str, show_scalar_bar: bool) -> None:
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(1500, 950))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars=scalar_name,
        cmap="cividis",
        smooth_shading=True,
        specular=0.22,
        specular_power=18.0,
        diffuse=0.92,
        ambient=0.18,
        pbr=False,
        show_edges=False,
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={
            "title": "norm. disp.",
            "title_font_size": 28,
            "label_font_size": 22,
            "vertical": True,
            "position_x": 0.86,
            "position_y": 0.18,
            "width": 0.055,
            "height": 0.62,
        },
    )
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    lx, ly, lz = xmax - xmin, ymax - ymin, zmax - zmin
    r = 2.15 * max(lx, ly, lz)
    # Slightly lower and more frontal than the grey diagnostic render so the
    # new low-volume members read as structural surfaces rather than slabs.
    plotter.camera.position = (cx + 0.80 * r, cy - 0.70 * r, cz + 0.48 * r)
    plotter.camera.focal_point = (cx, cy, cz)
    plotter.camera.up = (0.0, 0.0, 1.0)
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.18)

    plotter.remove_all_lights()
    key = pv.Light(
        position=(cx + 1.0 * r, cy - 0.55 * r, cz + 1.1 * r),
        focal_point=(cx, cy, cz),
        color="white",
        intensity=0.95,
        light_type="scene light",
    )
    fill = pv.Light(
        position=(cx - 1.0 * r, cy + 0.45 * r, cz + 0.5 * r),
        focal_point=(cx, cy, cz),
        color="white",
        intensity=0.48,
        light_type="scene light",
    )
    rim = pv.Light(
        position=(cx - 0.4 * r, cy - 1.1 * r, cz + 1.3 * r),
        focal_point=(cx, cy, cz),
        color="white",
        intensity=0.38,
        light_type="scene light",
    )
    for light in (key, fill, rim):
        plotter.add_light(light)

    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(out_png, transparent_background=False, return_img=False)
    plotter.close()


def render_case(case_dir: Path, out_dir: Path) -> dict:
    summary = _load_summary(case_dir)
    dims = tuple(int(x) for x in summary["dims"])
    rho_path = ROOT / summary["rho_final_npy"]
    rho = np.load(rho_path).reshape(dims)
    disp_rel = summary.get("disp_elem_npy")
    if disp_rel:
        scalar = np.load(ROOT / disp_rel).reshape(dims)
        scalar_kind = "element_displacement_magnitude"
    else:
        scalar = rho
        scalar_kind = "density_fallback"

    level = 0.5
    if not (float(rho.min()) < level < float(rho.max())):
        level = float(rho.min()) + 0.65 * (float(rho.max()) - float(rho.min()))
    mesh = density_to_polydata(rho, level=level, taubin_iters=7, taubin_pass_band=0.12)
    sampled = _sample_point_scalars(np.asarray(mesh.points), scalar)
    mesh.point_data["scalar_norm"] = _normalize(sampled).astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{summary['case']}_colored.png"
    clean_png = out_dir / f"{summary['case']}_colored_clean.png"
    _render_mesh(mesh, dims, out_png, "scalar_norm", show_scalar_bar=True)
    _render_mesh(mesh, dims, clean_png, "scalar_norm", show_scalar_bar=False)
    meta = {
        "case": summary["case"],
        "source_summary": str((case_dir / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "rho_final_npy": summary["rho_final_npy"],
        "scalar_source": disp_rel or summary["rho_final_npy"],
        "scalar_kind": scalar_kind,
        "iso_level": level,
        "mesh_points": int(mesh.n_points),
        "mesh_cells": int(mesh.n_cells),
        "colored_render_png": str(out_png.relative_to(ROOT)).replace("\\", "/"),
        "colored_clean_png": str(clean_png.relative_to(ROOT)).replace("\\", "/"),
    }
    (out_dir / f"{summary['case']}_colored_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def _trim_white(img: Image.Image, pad: int = 8) -> Image.Image:
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < 248 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return img
    left = max(bbox[0] - pad, 0)
    top = max(bbox[1] - pad, 0)
    right = min(bbox[2] + pad, img.width)
    bottom = min(bbox[3] + pad, img.height)
    return img.crop((left, top, right, bottom))


def make_panel(metas: list[dict], out_png: Path) -> None:
    labels = {
        "tool_long_cantilever_vf16": "(a) Long cantilever, $V_f=0.16$",
        "tool_portal_bridge_vf18": "(b) Portal bridge, $V_f=0.18$",
        "tool_asymmetric_bracket_vf14": "(c) Asymmetric bracket, $V_f=0.14$",
    }
    fig, axes = plt.subplots(1, len(metas), figsize=(5.2 * len(metas), 3.8), facecolor="white")
    axes = np.atleast_1d(axes)
    for ax, meta in zip(axes, metas):
        img = _trim_white(Image.open(ROOT / meta.get("colored_clean_png", meta["colored_render_png"])).convert("RGB"))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(labels.get(meta["case"], meta["case"]), fontsize=12, fontfamily="serif", pad=4)
    fig.tight_layout(pad=0.2, w_pad=0.35)
    fig.savefig(out_png, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_shared_colorbar_panel(metas: list[dict], out_png: Path) -> None:
    labels = {
        "tool_long_cantilever_vf16": "(a) Long cantilever, $V_f=0.16$",
        "tool_portal_bridge_vf18": "(b) Portal bridge, $V_f=0.18$",
        "tool_asymmetric_bracket_vf14": "(c) Asymmetric bracket, $V_f=0.14$",
    }
    fig = plt.figure(figsize=(13.8, 3.8), facecolor="white")
    gs = fig.add_gridspec(1, len(metas) + 1, width_ratios=[1, 1, 1, 0.045], wspace=0.04)
    for idx, meta in enumerate(metas):
        ax = fig.add_subplot(gs[0, idx])
        img = _trim_white(Image.open(ROOT / meta.get("colored_clean_png", meta["colored_render_png"])).convert("RGB"))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(labels.get(meta["case"], meta["case"]), fontsize=11.5, fontfamily="serif", pad=4)
    cax = fig.add_subplot(gs[0, -1])
    sm = plt.cm.ScalarMappable(cmap="cividis", norm=plt.Normalize(0.0, 1.0))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("normalized displacement", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="rerun_outputs/tool_paper_new_topology_solverv4_warm_large")
    parser.add_argument("--out", default="rerun_outputs/tool_paper_new_topology_solverv4_warm_large_colored")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()

    suite = ROOT / args.suite
    out_dir = ROOT / args.out
    cases = args.case or [
        "tool_long_cantilever_vf16",
        "tool_portal_bridge_vf18",
        "tool_asymmetric_bracket_vf14",
    ]
    metas = []
    for case in cases:
        print(f"rendering colored case {case}", flush=True)
        metas.append(render_case(suite / case, out_dir))
    panel = out_dir / "new_toolpaper_solverv4_colored_panel.png"
    shared_panel = out_dir / "new_toolpaper_solverv4_colored_panel_shared_colorbar.png"
    make_panel(metas, panel)
    make_shared_colorbar_panel(metas, shared_panel)
    summary = {
        "suite": str(suite.relative_to(ROOT)).replace("\\", "/"),
        "out": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "colored_panel_png": str(panel.relative_to(ROOT)).replace("\\", "/"),
        "colored_panel_shared_colorbar_png": str(shared_panel.relative_to(ROOT)).replace("\\", "/"),
        "renders": metas,
    }
    (out_dir / "colored_render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
