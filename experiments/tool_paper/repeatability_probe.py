from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tool_paper.initial_evidence import (  # noqa: E402
    _json_default,
    environment_report,
    run_uniform_probe,
)


DEFAULT_CASES = [
    "cantilever_gpu_medium",
    "bridge_gpu_medium@96x32x16",
    "bracket_gpu_medium@32x64x16",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "case",
        "repeat",
        "outer",
        "n_elem",
        "n_levels",
        "setup_s",
        "solve_s",
        "iters",
        "converged",
        "rel_residual",
        "compliance",
        "vram_used_gb",
        "residual_history_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["case"], []).append(row)

    summary = []
    for case, case_rows in grouped.items():
        setup = [float(row["setup_s"]) for row in case_rows]
        solve = [float(row["solve_s"]) for row in case_rows]
        iters = [int(row["iters"]) for row in case_rows]
        residuals = [float(row["rel_residual"]) for row in case_rows]
        summary.append(
            {
                "case": case,
                "repeats": len(case_rows),
                "n_elem": case_rows[0]["n_elem"],
                "n_levels": case_rows[0]["n_levels"],
                "outer": case_rows[0]["outer"],
                "all_converged": all(bool(row["converged"]) for row in case_rows),
                "setup_s_mean": statistics.fmean(setup),
                "setup_s_stdev": statistics.stdev(setup) if len(setup) > 1 else 0.0,
                "solve_s_mean": statistics.fmean(solve),
                "solve_s_stdev": statistics.stdev(solve) if len(solve) > 1 else 0.0,
                "iters_min": min(iters),
                "iters_max": max(iters),
                "rel_residual_max": max(residuals),
            }
        )
    return summary


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "case",
        "repeats",
        "n_elem",
        "n_levels",
        "outer",
        "all_converged",
        "setup_s_mean",
        "setup_s_stdev",
        "solve_s_mean",
        "solve_s_stdev",
        "iters_min",
        "iters_max",
        "rel_residual_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plot_repeatability(rows: list[dict], out_dir: Path) -> None:
    cases = list(dict.fromkeys(row["case"] for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=False)

    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        reps = [int(row["repeat"]) for row in case_rows]
        axes[0].plot(reps, [float(row["setup_s"]) for row in case_rows], marker="o", label=case)
        axes[1].plot(reps, [float(row["solve_s"]) for row in case_rows], marker="o", label=case)

    axes[0].set_title("Setup time by repeat")
    axes[0].set_xlabel("Repeat")
    axes[0].set_ylabel("Seconds")
    axes[0].grid(True, linestyle="--", alpha=0.35)

    axes[1].set_title("Solve time by repeat")
    axes[1].set_xlabel("Repeat")
    axes[1].set_ylabel("Seconds")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(fontsize=8)

    fig.suptitle("Warm repeatability probes")
    fig.tight_layout()
    fig.savefig(out_dir / "repeatability_timing.png", dpi=180)
    fig.savefig(out_dir / "repeatability_timing.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rerun_outputs/tool_paper_repeatability")
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--outer", choices=["pcg", "fgmres"], default="pcg")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-case", default="cantilever_3d")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    report = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": environment_report(),
        "warmup_case": args.warmup_case,
        "cases": cases,
        "repeats": args.repeats,
        "rows": [],
        "summary": [],
    }
    (out_dir / "environment.json").write_text(
        json.dumps(report["environment"], indent=2, default=_json_default),
        encoding="utf-8",
    )

    if args.warmup_case:
        print(f"[warmup] {args.warmup_case}", flush=True)
        warmup_dir = raw_dir / "warmup"
        warmup_dir.mkdir(parents=True, exist_ok=True)
        run_uniform_probe(args.warmup_case, warmup_dir, outer=args.outer)

    for repeat in range(1, args.repeats + 1):
        for case in cases:
            repeat_dir = raw_dir / f"repeat_{repeat:02d}"
            repeat_dir.mkdir(parents=True, exist_ok=True)
            print(f"[repeat {repeat}] {case}", flush=True)
            row = run_uniform_probe(case, repeat_dir, outer=args.outer)
            row["repeat"] = repeat
            report["rows"].append(row)
            _write_csv(out_dir / "repeatability_raw.csv", report["rows"])
            print(
                f"  setup={row['setup_s']:.3f}s solve={row['solve_s']:.3f}s "
                f"iters={row['iters']} rel_res={row['rel_residual']:.2e}",
                flush=True,
            )

    report["summary"] = _summarize(report["rows"])
    _write_summary_csv(out_dir / "repeatability_summary.csv", report["summary"])
    _plot_repeatability(report["rows"], out_dir)

    report["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "repeatability_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"[done] wrote {out_dir.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
