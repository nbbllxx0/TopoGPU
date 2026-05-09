"""Render one optimized 3D density field with the paper-style renderer.

This is intentionally narrow: it reuses the existing publication render
helpers without requiring the full six-case retained-density gallery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from figures.make_3d_renders import (
    density_to_polydata,
    make_figure_from_png,
    render_panel,
    trim_white,
)


def _dims_from_meta(meta: dict[str, object]) -> tuple[int, int, int]:
    for key in ("dims", "shape", "nel"):
        value = meta.get(key)
        if isinstance(value, list) and len(value) == 3:
            return int(value[0]), int(value[1]), int(value[2])
        if isinstance(value, tuple) and len(value) == 3:
            return int(value[0]), int(value[1]), int(value[2])

    problem = meta.get("problem")
    if isinstance(problem, dict):
        for keys in (("nelx", "nely", "nelz"), ("nx", "ny", "nz")):
            if all(k in problem for k in keys):
                return tuple(int(problem[k]) for k in keys)  # type: ignore[return-value]

    if all(k in meta for k in ("nelx", "nely", "nelz")):
        return int(meta["nelx"]), int(meta["nely"]), int(meta["nelz"])

    raise KeyError("Could not infer density dimensions from metadata.")


def _write_trimmed_png(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGB")
    trim_white(img, pad=18).save(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density", required=True, type=Path)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--stem", default="fresh_cantilever_216k")
    parser.add_argument("--title", default="Fresh cantilever 216k")
    parser.add_argument("--iso", default=0.5, type=float)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    nx, ny, nz = _dims_from_meta(meta)
    arr = np.load(args.density)
    expected = nx * ny * nz
    if arr.size != expected:
        raise ValueError(f"Expected {expected} density values for {nx}x{ny}x{nz}, got {arr.size}.")

    rho = arr.reshape(nx, ny, nz)
    mesh = density_to_polydata(rho, level=args.iso)

    raw_png = args.out_dir / f"{args.stem}_raw.png"
    trimmed_png = args.out_dir / f"{args.stem}.png"
    pdf = args.out_dir / f"{args.stem}.pdf"
    manifest = args.out_dir / f"{args.stem}_render_meta.json"

    render_panel(mesh, nx, ny, nz, raw_png)
    _write_trimmed_png(raw_png, trimmed_png)
    make_figure_from_png(trimmed_png, pdf, title=args.title)

    summary = {
        "density": str(args.density),
        "meta": str(args.meta),
        "dims": [nx, ny, nz],
        "n_elem": int(expected),
        "rho_mean": float(rho.mean()),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
        "iso": float(args.iso),
        "mesh_points": int(mesh.n_points),
        "mesh_cells": int(mesh.n_cells),
        "outputs": {
            "raw_png": str(raw_png),
            "trimmed_png": str(trimmed_png),
            "pdf": str(pdf),
        },
    }
    manifest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
