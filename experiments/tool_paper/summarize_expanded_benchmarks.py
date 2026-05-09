from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUNS = {
    "pcg_tol1e-5_move0.10_rhomin1e-3": "rerun_outputs/tool_paper_residual_pcg_tol1e5",
    "pcg_tol1e-5_move0.05_rhomin1e-3": "rerun_outputs/tool_paper_expanded_damped_pcg_tol1e5",
    "pcg_tol1e-5_move0.03_rhomin1e-2": "rerun_outputs/tool_paper_expanded_conservative_pcg_tol1e5",
}

PRODUCTION_SELECTION = {
    "tool_long_cantilever_vf16": "pcg_tol1e-5_move0.10_rhomin1e-3",
    "tool_short_cantilever_vf25": "pcg_tol1e-5_move0.05_rhomin1e-3",
    "tool_side_load_cantilever_vf24": "pcg_tol1e-5_move0.05_rhomin1e-3",
    "tool_high_volume_bracket_vf28": "pcg_tol1e-5_move0.05_rhomin1e-3",
    "tool_portal_bridge_vf30": "pcg_tol1e-5_move0.03_rhomin1e-2",
}

CASE_FAMILIES = {
    "tool_long_cantilever_vf16": "long cantilever",
    "tool_short_cantilever_vf25": "short cantilever",
    "tool_side_load_cantilever_vf24": "side-load cantilever",
    "tool_high_volume_bracket_vf28": "asymmetric bracket, high volume",
    "tool_portal_bridge_vf30": "portal bridge, high volume",
    "tool_portal_bridge_vf18": "portal bridge, low volume",
    "tool_asymmetric_bracket_vf14": "asymmetric bracket, low volume",
    "tool_deep_cantilever_vf20": "deep cantilever",
    "tool_oblique_cantilever_vf22": "oblique cantilever",
    "tool_dual_load_cantilever_vf26": "dual-load cantilever",
}

PLOT_LABELS = {
    "tool_long_cantilever_vf16": "long cantilever",
    "tool_short_cantilever_vf25": "short cantilever",
    "tool_side_load_cantilever_vf24": "side-load cantilever",
    "tool_high_volume_bracket_vf28": "high-volume bracket",
    "tool_portal_bridge_vf30": "high-volume portal",
}

PANEL_LABELS = ("(a)", "(b)", "(c)", "(d)")
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_batch(run_rel: str) -> list[dict]:
    batch = _read_json(ROOT / run_rel / "batch_summary.json")
    settings = batch.get("settings", {})
    rows: list[dict] = []
    for summary in batch.get("summaries", []):
        summary["_run_rel"] = run_rel
        summary["_settings"] = settings
        rows.append(summary)
    return rows


def _case_row(protocol: str, summary: dict) -> dict:
    final = summary.get("final", {})
    backend = summary.get("backend", {})
    run_settings = summary.get("_settings", {})
    settings = {
        "move": backend.get("move_limit", run_settings.get("move")),
        "rho_min_protocol": backend.get("rho_min", run_settings.get("rho_min", 1.0e-3)),
        "tol": backend.get("cg_tol"),
        "cap": backend.get("cg_maxiter"),
    }
    residual = final.get("linear_relative_residual")
    cap_count = int(summary.get("linear_solve_cap_count", -1))
    status = (
        "cap_free"
        if cap_count == 0 and residual not in {"", None} and float(residual) <= float(settings["tol"])
        else "stress_or_diagnostic"
    )
    case = summary.get("case", "")
    return {
        "protocol": protocol,
        "case": case,
        "family": CASE_FAMILIES.get(case, case),
        "dims": "x".join(str(x) for x in summary.get("dims", [])),
        "n_elem": summary.get("n_elem", ""),
        "volfrac": summary.get("volfrac", ""),
        "move": settings["move"],
        "rho_min_protocol": settings["rho_min_protocol"],
        "tol": settings["tol"],
        "cap": settings["cap"],
        "design_iterations": summary.get("iters", ""),
        "cap_count": cap_count,
        "final_residual": residual,
        "final_outer_iters": final.get("outer_iters", ""),
        "final_compliance": final.get("compliance", ""),
        "final_grayness": final.get("grayness", ""),
        "rho_mean": final.get("rho_mean", ""),
        "rho_min_final": final.get("rho_min", ""),
        "rho_max_final": final.get("rho_max", ""),
        "first_iter_wall_s": summary.get("first_iter_wall_s", ""),
        "warm_iter_mean_wall_s": summary.get("warm_iter_mean_wall_s_excluding_first", ""),
        "warm_iter_max_wall_s": summary.get("warm_iter_max_wall_s_excluding_first", ""),
        "total_wall_s": summary.get("total_wall_s", ""),
        "gpu_mem_mb_final": final.get("gpu_mem_used_mb", ""),
        "gpu_mem_mb_total": final.get("gpu_mem_total_mb", ""),
        "history_csv": summary.get("history_csv", ""),
        "summary_json": str((ROOT / summary["_run_rel"] / case / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "status": status,
    }


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_history(rel: str) -> list[dict]:
    rows = []
    with (ROOT / rel).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _plot_histories(out_dir: Path, rows: list[dict]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.7,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, constrained_layout=True)
    handles = []
    labels = []
    for idx, row in enumerate(rows):
        history = _read_history(str(row["history_csv"]))
        x = [int(r["iteration"]) for r in history]
        label = PLOT_LABELS.get(row["case"], row["case"].replace("tool_", "").replace("_", " "))
        style = LINESTYLES[idx % len(LINESTYLES)]
        color = PALETTE[idx % len(PALETTE)]
        line = axes[0, 0].plot(x, [float(r["compliance"]) for r in history], color=color, linestyle=style, label=label)[0]
        axes[0, 1].semilogy(
            x,
            [max(float(r["linear_relative_residual"]), 1e-16) for r in history],
            color=color,
            linestyle=style,
            label=label,
        )
        axes[1, 0].plot(x, [float(r["grayness"]) for r in history], color=color, linestyle=style, label=label)
        axes[1, 1].plot(x, [float(r["outer_iters"]) for r in history], color=color, linestyle=style, label=label)
        handles.append(line)
        labels.append(label)
    titles = ("Compliance", "Unpreconditioned residual", "Grayness", "Krylov iterations")
    ylabels = ("compliance", "relative residual", "grayness", "outer iterations")
    for ax, panel, title, ylabel in zip(axes.ravel(), PANEL_LABELS, titles, ylabels):
        ax.set_title(f"{panel} {title}", loc="left")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Design iteration")
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[0, 1].axhline(1e-5, color="0.15", linestyle="--", linewidth=1.0)
    axes[0, 1].annotate(
        "residual gate",
        xy=(0.98, 1e-5),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.15",
    )
    axes[1, 1].axhline(800, color="0.15", linestyle="--", linewidth=1.0)
    axes[1, 1].annotate(
        "Krylov cap",
        xy=(0.98, 800),
        xycoords=("axes fraction", "data"),
        xytext=(-4, -12),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8,
        color="0.15",
    )
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(out_dir / "FIG_PRODUCTION_HISTORY_DIAGNOSTICS.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "FIG_PRODUCTION_HISTORY_DIAGNOSTICS.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    out_dir = ROOT / "rerun_outputs" / "tool_paper_expanded_benchmark_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    latest_by_case: dict[str, dict] = {}
    for protocol, run_rel in RUNS.items():
        for summary in _read_batch(run_rel):
            row = _case_row(protocol, summary)
            all_rows.append(row)
            latest_by_case[row["case"]] = row

    selected = []
    for case, protocol in PRODUCTION_SELECTION.items():
        match = next((row for row in all_rows if row["case"] == case and row["protocol"] == protocol), None)
        if match:
            selected.append(match)

    stress_rows = [
        row
        for row in all_rows
        if row["status"] != "cap_free" and row["case"] not in PRODUCTION_SELECTION
    ]
    fields = [
        "protocol",
        "case",
        "family",
        "dims",
        "n_elem",
        "volfrac",
        "move",
        "rho_min_protocol",
        "tol",
        "cap",
        "design_iterations",
        "cap_count",
        "final_residual",
        "final_outer_iters",
        "final_compliance",
        "final_grayness",
        "rho_mean",
        "rho_min_final",
        "rho_max_final",
        "first_iter_wall_s",
        "warm_iter_mean_wall_s",
        "warm_iter_max_wall_s",
        "total_wall_s",
        "gpu_mem_mb_final",
        "gpu_mem_mb_total",
        "history_csv",
        "summary_json",
        "status",
    ]
    _write_csv(out_dir / "TABLE_EXPANDED_PRODUCTION_BENCHMARKS.csv", selected, fields)
    _write_csv(out_dir / "TABLE_EXPANDED_STRESS_DIAGNOSTICS.csv", stress_rows, fields)
    _write_csv(out_dir / "TABLE_EXPANDED_ALL_RUNS.csv", all_rows, fields)
    _plot_histories(out_dir, selected)

    summary = {
        "production_rows": len(selected),
        "production_cap_free_rows": sum(1 for row in selected if row["status"] == "cap_free"),
        "stress_rows": len(stress_rows),
        "out": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "production_cases": [row["case"] for row in selected],
        "note": "Rows are protocol-specific. Conservative move/rho_min rows are admissible for residual/timing only unless grayness and visual convergence are separately accepted.",
    }
    (out_dir / "expanded_benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
