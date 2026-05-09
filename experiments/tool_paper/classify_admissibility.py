from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CAP_SUITES = [
    ("pcg_tol1e-6", "rerun_outputs/tool_paper_new_topology_solverv4_warm_large"),
    ("pcg_tol1e-5", "rerun_outputs/tool_paper_new_topology_solverv4_capdiag_tol1e5"),
    ("fgmres_tol1e-6", "rerun_outputs/tool_paper_new_topology_solverv4_capdiag_fgmres"),
]

DEFAULT_CASES = [
    "tool_long_cantilever_vf16",
    "tool_portal_bridge_vf18",
    "tool_asymmetric_bracket_vf14",
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def classify_case(suite_name: str, suite_dir: Path, case: str) -> dict | None:
    case_dir = suite_dir / case
    summary = _read_json(case_dir / "summary.json")
    history = _read_csv(case_dir / "history.csv")
    if not summary or not history:
        return None

    cg_maxiter = int(summary.get("backend", {}).get("cg_maxiter", summary.get("final", {}).get("outer_iters", 0)))
    capped_rows = [row for row in history if int(float(row["outer_iters"])) >= cg_maxiter]
    cap_count = len(capped_rows)
    first_cap_iter = int(capped_rows[0]["iteration"]) if capped_rows else ""
    final = history[-1]
    final_iter = int(float(final["outer_iters"]))
    final_compliance = _float(final, "compliance")
    final_grayness = _float(final, "grayness")
    start_compliance = _float(history[0], "compliance")
    compliance_drop_total = start_compliance - final_compliance
    if capped_rows:
        before_cap = history[max(0, int(capped_rows[0]["iteration"]) - 2)]
        compliance_at_cap_onset = _float(capped_rows[0], "compliance")
        compliance_before_cap = _float(before_cap, "compliance")
        compliance_drop_after_cap = compliance_at_cap_onset - final_compliance
        grayness_at_cap_onset = _float(capped_rows[0], "grayness")
    else:
        compliance_at_cap_onset = ""
        compliance_before_cap = ""
        compliance_drop_after_cap = 0.0
        grayness_at_cap_onset = ""

    cap_fraction = cap_count / max(len(history), 1)
    relative_late_change = (
        compliance_drop_after_cap / max(abs(final_compliance), 1.0e-300)
        if capped_rows
        else 0.0
    )
    if cap_count == 0:
        classification = "timing_admissible"
        manuscript_use = "timing_and_visual"
        reason = "No design iteration reached the configured linear-solve cap."
    elif suite_name == "pcg_tol1e-6":
        classification = "visual_stress_case"
        manuscript_use = "visual_only_with_cap_disclosure"
        reason = "Visual-generation protocol has cap hits; use only with explicit cap disclosure."
    else:
        classification = "conditioning_limited"
        manuscript_use = "limitation_or_diagnostic"
        reason = "Diagnostic protocol still hits the configured linear-solve cap."

    return {
        "suite": suite_name,
        "case": case,
        "dims": "x".join(str(v) for v in summary.get("dims", [])),
        "outer_solver": summary.get("backend", {}).get("gmg_outer_solver", summary.get("final", {}).get("outer_solver", "")),
        "cg_tol": summary.get("backend", {}).get("cg_tol", ""),
        "cg_maxiter": cg_maxiter,
        "design_iterations": len(history),
        "cap_count": cap_count,
        "cap_fraction": cap_fraction,
        "first_cap_iteration": first_cap_iter,
        "final_outer_iters": final_iter,
        "compliance_initial": start_compliance,
        "compliance_before_cap": compliance_before_cap,
        "compliance_at_cap_onset": compliance_at_cap_onset,
        "compliance_final": final_compliance,
        "compliance_drop_total": compliance_drop_total,
        "compliance_drop_after_cap_onset": compliance_drop_after_cap,
        "relative_late_compliance_change": relative_late_change,
        "grayness_at_cap_onset": grayness_at_cap_onset,
        "grayness_final": final_grayness,
        "residual_norm_recorded": False,
        "classification": classification,
        "manuscript_use": manuscript_use,
        "reason": reason,
        "summary_json": str((case_dir / "summary.json").relative_to(ROOT)).replace("\\", "/"),
        "history_csv": str((case_dir / "history.csv").relative_to(ROOT)).replace("\\", "/"),
    }


def run(out_dir: Path, cases: list[str]) -> dict:
    rows = []
    for suite_name, rel_suite in CAP_SUITES:
        suite_dir = ROOT / rel_suite
        for case in cases:
            row = classify_case(suite_name, suite_dir, case)
            if row is not None:
                rows.append(row)
    fields = [
        "suite",
        "case",
        "dims",
        "outer_solver",
        "cg_tol",
        "cg_maxiter",
        "design_iterations",
        "cap_count",
        "cap_fraction",
        "first_cap_iteration",
        "final_outer_iters",
        "compliance_initial",
        "compliance_before_cap",
        "compliance_at_cap_onset",
        "compliance_final",
        "compliance_drop_total",
        "compliance_drop_after_cap_onset",
        "relative_late_compliance_change",
        "grayness_at_cap_onset",
        "grayness_final",
        "residual_norm_recorded",
        "classification",
        "manuscript_use",
        "reason",
        "summary_json",
        "history_csv",
    ]
    _write_csv(out_dir / "TABLE_ADMISSIBILITY_CAP_STATUS.csv", rows, fields)
    summary = {
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(rows),
        "timing_admissible_rows": sum(1 for row in rows if row["classification"] == "timing_admissible"),
        "visual_stress_rows": sum(1 for row in rows if row["classification"] == "visual_stress_case"),
        "conditioning_limited_rows": sum(1 for row in rows if row["classification"] == "conditioning_limited"),
        "residual_norm_recorded": False,
        "note": "Histories record Krylov iteration counts but not final residual norms; cap status is an admissibility proxy, not a residual substitute.",
    }
    (out_dir / "admissibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify result admissibility from solver cap histories.")
    parser.add_argument("--out", default="rerun_outputs/tool_paper_admissibility")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run(out_dir, args.case or DEFAULT_CASES)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
