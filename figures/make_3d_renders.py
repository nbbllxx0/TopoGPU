"""
Publication-grade 3D renders for the qualitative density-field gallery (grey-matte,
smooth-shaded, soft 3-point light).

Pipeline:
    rho (npy)  ->  marching_cubes (skimage)  ->  pyvista.PolyData
                                              ->  PBR off-screen render to PNG

Output: figs/F11_gallery.pdf  (2x3 grid)
        figs/F11a-F11d.pdf    (individual panels)

Run from the figures directory:
    python make_3d_renders.py --renders-dir ../rerun_outputs/topology_renders --figs-dir ../rerun_outputs/paper4_figs
    python make_3d_renders.py --renders-dir ../rerun_outputs/topology_renders --figs-dir ../tmp_figs
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
from skimage import measure
import pyvista as pv
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RENDERS = BASE_DIR.parent / "rerun_outputs" / "topology_renders"
DEFAULT_FIGS = BASE_DIR.parent / "rerun_outputs" / "paper4_figs"
RENDERS = DEFAULT_RENDERS
FIGS = DEFAULT_FIGS
TMP = FIGS / "_render_tmp"


def configure_paths(renders_dir=None, figs_dir=None):
    global RENDERS, FIGS, TMP
    RENDERS = Path(
        renders_dir
        or os.environ.get("PAPER4_RENDERS_DIR")
        or DEFAULT_RENDERS
    )
    FIGS = Path(
        figs_dir
        or os.environ.get("PAPER4_FIGS_DIR")
        or DEFAULT_FIGS
    )
    TMP = FIGS / "_render_tmp"
    FIGS.mkdir(exist_ok=True, parents=True)
    TMP.mkdir(exist_ok=True, parents=True)

# (label, npy stem, nelx, nely, nelz, iso threshold, expected volume fraction)
STRUCTURES = [
    ("(a) Cantilever 216k",      "cantilever_216k_rho_best",     120,  60,  30, 0.5, 0.3),
    ("(b) Bridge 216k",          "bridge_216k_rho_best",         120,  60,  30, 0.5, 0.3),
    ("(c) Double-clamped 216k",  "doubleclamp_216k_rho_best",    120,  60,  30, 0.5, 0.1),
    ("(d) Torsion 499k",         "torsion_500k_rho_best",        165,  55,  55, 0.5, 0.25),
    (r"(e) Cantilever 1M ($V_f=0.3$)", "cantilever_1m_rho_best",   200, 100,  50, 0.5, 0.3),
    # The retained vf=0.1 `rho_final` and `rho_best` arrays are byte-identical in
    # this bundle, so the panel is labeled as a qualitative snapshot rather than
    # as evidence of a final-vs-best distinction.
    (r"(f) Cantilever 1M low-volume ($V_f=0.1$)", "cantilever_1m_vf01_rho_final", 200, 100, 50, 0.5, 0.1),
]

IMG_SIZE   = (1400, 900)
GREY_MATTE = (0.62, 0.62, 0.64)   # slightly cool mid-grey; matches reference aesthetic


def _meta_path_from_stem(stem):
    base = stem
    for suffix in ("_rho_best", "_rho_final", "_rho_run1"):
        if stem.endswith(suffix):
            base = stem[:-len(suffix)]
            break
    return RENDERS / (base + "_meta.json")


def load_density(label, stem, nelx, nely, nelz, expected_vf):
    arr = np.load(RENDERS / (stem + ".npy"))
    expected_size = nelx * nely * nelz
    if arr.size != expected_size:
        raise ValueError(
            f"{label}: expected {expected_size} voxels for {nelx}x{nely}x{nelz}, got {arr.size}"
        )
    rho = arr.reshape(nelx, nely, nelz)
    measured_vf = float(rho.mean())
    if abs(measured_vf - expected_vf) > 0.05:
        print(
            f"WARNING: {label} mean density {measured_vf:.3f} differs from caption vf={expected_vf:.3f}",
            flush=True,
        )

    meta_path = _meta_path_from_stem(stem)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        source_density = meta.get("source_density")
        stem_density = stem.split("_")[-2] + "_" + stem.split("_")[-1] if "_rho_" in stem else None
        if source_density and stem_density and source_density != stem_density:
            print(
                f"WARNING: {label} metadata source_density={source_density} but render stem uses {stem_density}",
                flush=True,
            )
    return rho


def density_to_polydata(rho, level=0.5, taubin_iters=10, taubin_pass_band=0.1):
    """Marching-cubes -> PolyData with light Taubin smoothing.

    We use few iterations (10) and a wide pass band (0.1) so the staircase
    from the voxel grid is removed but fine features in high-resolution
    meshes (1M+ elements) are preserved.  Heavy smoothing erases the very
    detail that distinguishes a 1M-element result from a 216k one.
    """
    padded = np.pad(rho, 1, mode="constant", constant_values=0.0)
    verts, faces, normals, _ = measure.marching_cubes(
        padded, level=level, allow_degenerate=False)
    verts -= 1.0
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64),
                          faces.astype(np.int64)]).ravel()
    mesh = pv.PolyData(verts.astype(np.float32), faces_pv)
    mesh = mesh.smooth_taubin(n_iter=taubin_iters, pass_band=taubin_pass_band,
                               feature_smoothing=False,
                               boundary_smoothing=False,
                               non_manifold_smoothing=False)
    mesh = mesh.compute_normals(auto_orient_normals=True,
                                 consistent_normals=True,
                                 feature_angle=30.0)
    return mesh


def render_panel(mesh, nelx, nely, nelz, out_png):
    plotter = pv.Plotter(off_screen=True, window_size=IMG_SIZE)
    plotter.set_background("white")

    plotter.add_mesh(
        mesh,
        color=GREY_MATTE,
        smooth_shading=True,
        specular=0.18,
        specular_power=14.0,
        diffuse=0.95,
        ambient=0.22,
        pbr=False,
        show_edges=False,
    )

    # Camera: hero 3/4 view. Place camera off the +x / -y / +z octant and point
    # at the mesh centre; parallel projection avoids fisheye distortion in long
    # aspect-ratio beams.
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    lx, ly, lz = xmax - xmin, ymax - ymin, zmax - zmin
    r = 2.2 * max(lx, ly, lz)
    cam_pos = (cx + 0.70 * r, cy - 0.85 * r, cz + 0.55 * r)
    plotter.camera.position     = cam_pos
    plotter.camera.focal_point  = (cx, cy, cz)
    plotter.camera.up           = (0.0, 0.0, 1.0)
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.camera.zoom(1.15)

    # Three-point lighting (key / fill / rim); remove PyVista's default headlight.
    plotter.remove_all_lights()
    key  = pv.Light(position=(cx + 1.0 * r, cy - 0.5 * r, cz + 1.2 * r),
                    focal_point=(cx, cy, cz),
                    color="white", intensity=0.95, light_type="scene light")
    fill = pv.Light(position=(cx - 1.2 * r, cy + 0.4 * r, cz + 0.6 * r),
                    focal_point=(cx, cy, cz),
                    color="white", intensity=0.45, light_type="scene light")
    rim  = pv.Light(position=(cx - 0.3 * r, cy - 1.1 * r, cz + 1.4 * r),
                    focal_point=(cx, cy, cz),
                    color="white", intensity=0.35, light_type="scene light")
    for L in (key, fill, rim):
        plotter.add_light(L)

    plotter.enable_anti_aliasing("ssaa")
    plotter.screenshot(out_png, transparent_background=False,
                        return_img=False)
    plotter.close()


def make_figure_from_png(png_path, pdf_path, title=None):
    img = Image.open(png_path)
    fig_w = img.width / 200.0
    fig_h = img.height / 200.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=11, fontfamily="serif", pad=3)
    fig.savefig(pdf_path, dpi=200, bbox_inches="tight", pad_inches=0.02,
                 facecolor="white")
    plt.close(fig)


def make_gallery_pdf(panels, out_pdf, ncols=3):
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 3.9 * nrows),
                              facecolor="white")
    axes = np.atleast_2d(axes)
    for idx, (label, png) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.imshow(Image.open(png))
        ax.set_axis_off()
        ax.set_title(label, fontsize=14, fontfamily="serif", pad=4)
    for j in range(len(panels), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_axis_off()
    fig.suptitle("Representative qualitative topology examples",
                 fontsize=16, fontfamily="serif", y=0.995)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(renders_dir=None, figs_dir=None):
    configure_paths(renders_dir=renders_dir, figs_dir=figs_dir)
    # 1. render each structure to PNG
    panel_pngs = []
    for (label, stem, nx, ny, nz, thr, expected_vf) in STRUCTURES:
        print(f"  rendering {label} ...", flush=True)
        rho  = load_density(label, stem, nx, ny, nz, expected_vf)
        mesh = density_to_polydata(rho, level=thr)
        png  = TMP / (stem + ".png")
        render_panel(mesh, nx, ny, nz, png)
        panel_pngs.append((label, png))

    # 2. 2x3 gallery
    gallery = FIGS / "F11_gallery.pdf"
    make_gallery_pdf(panel_pngs, gallery, ncols=3)
    print(f"  wrote {gallery}")

    # 3. individual panels used in the body text
    PRIMARY = [
        ("cantilever_216k_rho_best", "F11a_cant216k",    "Cantilever 216k"),
        ("bridge_216k_rho_best",     "F11b_bridge216k",  "Bridge 216k"),
        ("torsion_500k_rho_best",    "F11c_torsion500k", "Torsion 499k"),
        ("cantilever_1m_rho_best",   "F11d_cant1m",      "Cantilever 1M"),
    ]
    # Map stem -> png path produced above
    png_by_stem = {s: p for (lbl, s, *_), (_, p) in zip(STRUCTURES, panel_pngs)}
    for stem, name, lbl in PRIMARY:
        png = png_by_stem[stem]
        out_pdf = FIGS / (name + ".pdf")
        make_figure_from_png(png, out_pdf, title=lbl)
        print(f"  wrote {out_pdf}")

    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renders-dir",
        default=None,
        help="Directory containing qualitative density-field arrays and metadata",
    )
    parser.add_argument(
        "--figs-dir",
        default=None,
        help="Directory where regenerated render figures should be written",
    )
    args = parser.parse_args()
    main(renders_dir=args.renders_dir, figs_dir=args.figs_dir)
