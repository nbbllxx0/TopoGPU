from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RENDERS = ROOT / "rerun_outputs" / "topology_renders"
DEFAULT_OUT = ROOT / "rerun_outputs" / "tool_paper_topology_figs"

PANELS = [
    ("Cantilever 216k", "cantilever_216k", "cantilever_216k_rho_best.npy"),
    ("Bridge 216k", "bridge_216k", "bridge_216k_rho_best.npy"),
    ("Double-clamped 216k", "doubleclamp_216k", "doubleclamp_216k_rho_best.npy"),
    ("Torsion 499k", "torsion_500k", "torsion_500k_rho_best.npy"),
    ("Cantilever 1M", "cantilever_1m", "cantilever_1m_rho_best.npy"),
    ("Cantilever 1M low-volume", "cantilever_1m_vf01", "cantilever_1m_vf01_rho_final.npy"),
]


def load_case(renders_dir: Path, meta_stem: str, density_name: str) -> tuple[dict, np.ndarray]:
    meta = json.loads((renders_dir / f"{meta_stem}_meta.json").read_text(encoding="utf-8"))
    arr = np.load(renders_dir / density_name)
    expected = int(meta["nelx"]) * int(meta["nely"]) * int(meta["nelz"])
    if arr.size != expected:
        raise ValueError(f"{density_name}: expected {expected} values, found {arr.size}")
    rho = arr.reshape(int(meta["nelx"]), int(meta["nely"]), int(meta["nelz"]))
    return meta, rho


def make_gallery(renders_dir: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), constrained_layout=True)
    summary: list[dict] = []
    for ax, (title, meta_stem, density_name) in zip(axes.flat, PANELS):
        meta, rho = load_case(renders_dir, meta_stem, density_name)
        solid_projection = rho.max(axis=2).T
        mid_slice = rho[:, :, rho.shape[2] // 2].T
        composite = 0.70 * solid_projection + 0.30 * mid_slice
        ax.imshow(composite, origin="lower", cmap="gray_r", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(
            f"{title}\n{meta['nelx']}x{meta['nely']}x{meta['nelz']}, "
            f"V={float(meta['volfrac']):.2f}, C={float(meta['best_compliance']):.3g}",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        summary.append(
            {
                "label": title,
                "density_file": str((renders_dir / density_name).relative_to(ROOT)).replace("\\", "/"),
                "nelx": int(meta["nelx"]),
                "nely": int(meta["nely"]),
                "nelz": int(meta["nelz"]),
                "n_elem": int(meta["n_elem"]),
                "volfrac": float(meta["volfrac"]),
                "mean_density": float(rho.mean()),
                "best_compliance": float(meta["best_compliance"]),
                "best_grayness": float(meta["best_grayness"]),
                "source_density": meta.get("source_density", ""),
                "best_iteration": meta.get("best_iteration"),
            }
        )
    fig.suptitle("Copied topology-density snapshots: max-depth projection plus mid-plane context", fontsize=13)
    fig.savefig(out_dir / "topology_density_snapshot_gallery.png", dpi=180)
    fig.savefig(out_dir / "topology_density_snapshot_gallery.pdf")
    plt.close(fig)

    (out_dir / "topology_density_snapshot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", default=str(DEFAULT_RENDERS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    summary = make_gallery(Path(args.renders_dir), Path(args.out_dir))
    for row in summary:
        print(
            f"{row['label']}: n={row['n_elem']:,}, mean={row['mean_density']:.4f}, "
            f"gray={row['best_grayness']:.2e}, C={row['best_compliance']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
