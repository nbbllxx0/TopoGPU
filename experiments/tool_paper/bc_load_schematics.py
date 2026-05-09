from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tool_paper.fast_new_topology_probe import CASES  # noqa: E402


CASE_LABELS = {
    "tool_long_cantilever_vf16": "(a) Long cantilever, $V_f=0.16$",
    "tool_portal_bridge_vf18": "(b) Portal bridge, $V_f=0.18$",
    "tool_asymmetric_bracket_vf14": "(c) Asymmetric bracket, $V_f=0.14$",
}

FIGURE_META = {
    "tool_long_cantilever_vf16": {"dims": (96, 48, 48), "lengths": (2.4, 1.0, 1.0)},
    "tool_portal_bridge_vf18": {"dims": (112, 56, 40), "lengths": (112.0, 56.0, 40.0)},
    "tool_asymmetric_bracket_vf14": {"dims": (96, 64, 40), "lengths": (2.0, 1.25, 0.75)},
}


def _box_edges(lengths: tuple[float, float, float]) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    lx, ly, lz = lengths
    pts = np.array(
        [
            [0, 0, 0],
            [lx, 0, 0],
            [lx, ly, 0],
            [0, ly, 0],
            [0, 0, lz],
            [lx, 0, lz],
            [lx, ly, lz],
            [0, ly, lz],
        ],
        dtype=float,
    )
    idx = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(tuple(pts[i]), tuple(pts[j])) for i, j in idx]


def _lengths(case: str, cfg: dict) -> tuple[float, float, float]:
    if case in FIGURE_META:
        return tuple(FIGURE_META[case]["lengths"])
    if cfg["kind"] == "asym_bracket":
        return (2.0, 1.25, 0.75)
    if cfg["kind"] == "portal_bridge":
        nx, ny, nz = cfg["dims"]
        return (float(nx), float(ny), float(nz))
    return (2.4, 1.0, 1.0)


def _draw_fixed_face(ax, lengths: tuple[float, float, float], x: float, color: str, label: str) -> None:
    _, ly, lz = lengths
    yy = np.linspace(0.0, ly, 8)
    zz = np.linspace(0.0, lz, 8)
    for y in yy:
        ax.plot([x, x], [y, y], [0, lz], color=color, linewidth=1.1, alpha=0.65)
    for z in zz:
        ax.plot([x, x], [0, ly], [z, z], color=color, linewidth=1.1, alpha=0.65)
    ax.scatter([x], [0.5 * ly], [0.5 * lz], color=color, marker="s", s=58, depthshade=False)
    ax.text(x, 0.52 * ly, 1.08 * lz, label, color=color, fontsize=8.5, ha="center")


def _draw_arrow(ax, start, vec, color: str, label: str) -> None:
    start = np.array(start, dtype=float)
    vec = np.array(vec, dtype=float)
    ax.quiver(
        start[0],
        start[1],
        start[2],
        vec[0],
        vec[1],
        vec[2],
        color=color,
        linewidth=3.0,
        arrow_length_ratio=0.22,
        normalize=False,
    )
    end = start + vec
    ax.text(end[0], end[1], end[2], label, color=color, fontsize=10, fontweight="bold")


def _draw_axes(ax, lengths: tuple[float, float, float]) -> None:
    lx, ly, lz = lengths
    origin = np.array((-0.08 * lx, -0.08 * ly, -0.08 * lz))
    scale = 0.18 * min(lx, ly, lz)
    for vec, label in [((scale, 0, 0), "$x$"), ((0, scale, 0), "$y$"), ((0, 0, scale), "$z$")]:
        _draw_arrow(ax, origin, vec, "0.15", label)


def _dimension_text(cfg: dict, lengths: tuple[float, float, float]) -> str:
    dims = r"$%d\times%d\times%d$ elements" % tuple(cfg["dims"])
    lx, ly, lz = lengths
    return f"{dims}\nrelative domain {lx:g}:{ly:g}:{lz:g}"


def draw_case(ax, case: str, cfg: dict) -> dict:
    lengths = _lengths(case, cfg)
    plot_cfg = dict(cfg)
    if case in FIGURE_META:
        plot_cfg["dims"] = FIGURE_META[case]["dims"]
    lx, ly, lz = lengths
    for a, b in _box_edges(lengths):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="0.20", linewidth=1.25)
    ax.set_title(CASE_LABELS.get(case, case), fontsize=12, fontfamily="serif", pad=5)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect((lx, ly, lz))
    _draw_axes(ax, lengths)
    ax.text2D(0.02, 0.92, _dimension_text(plot_cfg, lengths), transform=ax.transAxes, fontsize=8.5, color="0.2")

    if cfg["kind"] == "portal_bridge":
        _draw_fixed_face(ax, lengths, 0.0, "#315f9f", r"$u_x=u_y=u_z=0$")
        _draw_fixed_face(ax, lengths, lx, "#6e6e6e", r"$u_x=0$ roller")
        load_start = (0.52 * lx, ly, 0.5 * lz)
        _draw_arrow(ax, (load_start[0], load_start[1] + 0.18 * ly, load_start[2]), (0, -0.22 * ly, 0), "#b3261e", "F")
        support_desc = "left fixed face; right roller-x face"
        load_desc = "distributed upper-span load patch near x=0.52L, direction -y"
        ax.text(0.52 * lx, 1.03 * ly, 0.40 * lz, "load patch", color="#b3261e", fontsize=8.5, ha="center")
    else:
        _draw_fixed_face(ax, lengths, 0.0, "#315f9f", r"$u_x=u_y=u_z=0$")
        load_rel = cfg.get("load_rel", (1.0, 0.5, 0.5))
        center = (lx, load_rel[1] * ly, load_rel[2] * lz)
        load_vec = np.array(cfg.get("load_vector", (0.0, 0.0, -1.0)), dtype=float)
        load_vec = load_vec / max(float(np.linalg.norm(load_vec)), 1e-12)
        scale = 0.28 * min(lx, ly, lz)
        _draw_arrow(ax, center - load_vec * scale, load_vec * scale, "#b3261e", "F")
        support_desc = "left fixed face"
        load_desc = f"right-face load patch at relative location {load_rel}, vector {tuple(cfg.get('load_vector', (0.0, 0.0, -1.0)))}"
        ax.scatter([center[0]], [center[1]], [center[2]], color="#b3261e", marker="o", s=62, depthshade=False)
        ax.text(center[0], center[1], center[2] + 0.08 * lz, "load patch", color="#b3261e", fontsize=8.5, ha="center")

    return {
        "case": case,
        "dims": "x".join(str(v) for v in plot_cfg["dims"]),
        "volfrac": cfg["volfrac"],
        "rmin": cfg["rmin"],
        "support": support_desc,
        "load": load_desc,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rerun_outputs/tool_paper_evidence_bundle/toolpaper_artifact_pack")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = args.case or [
        "tool_long_cantilever_vf16",
        "tool_portal_bridge_vf18",
        "tool_asymmetric_bracket_vf14",
    ]
    fig = plt.figure(figsize=(14.4, 4.8), facecolor="white")
    rows = []
    for idx, case in enumerate(cases, start=1):
        ax = fig.add_subplot(1, len(cases), idx, projection="3d")
        rows.append(draw_case(ax, case, CASES[case]))
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#315f9f", markersize=9, label="fixed support"),
            plt.Line2D([0], [0], color="#b3261e", linewidth=3.0, label="applied load"),
            plt.Line2D([0], [0], color="0.20", linewidth=1.25, label="domain boundary"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    fig.tight_layout(pad=0.25, rect=(0, 0.06, 1, 1))
    out_png = out_dir / "FIG_BC_LOAD_SCHEMATICS.png"
    fig.savefig(out_png, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with (out_dir / "TABLE_BC_LOADS.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "dims", "volfrac", "rmin", "support", "load"])
        writer.writeheader()
        writer.writerows(rows)
    print(out_png.relative_to(ROOT).as_posix(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
