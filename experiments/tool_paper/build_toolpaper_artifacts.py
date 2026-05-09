from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CASES = [
    "tool_long_cantilever_vf16",
    "tool_portal_bridge_vf18",
    "tool_asymmetric_bracket_vf14",
]

CAP_DIAGNOSTIC_SUITES = [
    ("pcg_tol1e-6", "rerun_outputs/tool_paper_new_topology_solverv4_warm_large"),
    ("pcg_tol1e-5", "rerun_outputs/tool_paper_new_topology_solverv4_capdiag_tol1e5"),
    ("fgmres_tol1e-6", "rerun_outputs/tool_paper_new_topology_solverv4_capdiag_fgmres"),
]

VERIFICATION_FILES = [
    ("TABLE_OPERATOR_VERIFICATION.csv", "operator_verification_csv"),
    ("TABLE_SENSITIVITY_VERIFICATION.csv", "sensitivity_verification_csv"),
    ("TABLE_FILTER_VERIFICATION.csv", "filter_verification_csv"),
    ("verification_summary.json", "verification_summary_json"),
]

ADMISSIBILITY_FILES = [
    ("TABLE_ADMISSIBILITY_CAP_STATUS.csv", "admissibility_cap_status_csv"),
    ("admissibility_summary.json", "admissibility_summary_json"),
]

BASELINE_FILES = [
    ("TABLE_BASELINE_ABLATION.csv", "baseline_ablation_csv"),
    ("baseline_ablation_summary.json", "baseline_ablation_summary_json"),
]

EXPANDED_BENCHMARK_FILES = [
    ("TABLE_EXPANDED_PRODUCTION_BENCHMARKS.csv", "expanded_production_benchmarks_csv"),
    ("TABLE_EXPANDED_STRESS_DIAGNOSTICS.csv", "expanded_stress_diagnostics_csv"),
    ("TABLE_EXPANDED_ALL_RUNS.csv", "expanded_all_runs_csv"),
    ("expanded_benchmark_summary.json", "expanded_benchmark_summary_json"),
    ("FIG_PRODUCTION_HISTORY_DIAGNOSTICS.png", "production_history_diagnostics_png"),
    ("FIG_PRODUCTION_HISTORY_DIAGNOSTICS.pdf", "production_history_diagnostics_pdf"),
]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _case_summary(suite: Path, case: str) -> dict:
    summary = _read_json(suite / case / "summary.json")
    if summary:
        summary["_summary_path"] = str((suite / case / "summary.json").relative_to(ROOT)).replace("\\", "/")
    return summary


def _colored_meta(colored_dir: Path, case: str) -> dict:
    path = colored_dir / f"{case}_colored_meta.json"
    meta = _read_json(path)
    if meta:
        meta["_meta_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
    return meta


def build_tables(out_dir: Path, suite: Path, colored_dir: Path, cases: list[str]) -> dict:
    matrix = _read_json(ROOT / "experiments" / "tool_paper" / "sample_matrix.json")
    target_by_case = {item["case"]: item for item in matrix.get("new_toolpaper_topology_targets", [])}

    sample_rows = []
    backend_rows = []
    convergence_rows = []
    timing_rows = []
    cap_diag_rows = []
    manifest_rows = []
    for case in cases:
        summary = _case_summary(suite, case)
        colored = _colored_meta(colored_dir, case)
        target = target_by_case.get(case, {})
        dims = "x".join(str(v) for v in summary.get("dims", []))
        final = summary.get("final", {})
        backend = summary.get("backend", {})
        sample_rows.append(
            {
                "case": case,
                "bvp": target.get("bvp", ""),
                "volume_fraction": summary.get("volfrac", ""),
                "dims": dims,
                "n_elem": summary.get("n_elem", ""),
                "status": "present" if summary else "missing",
                "main_story_use": "new_toolpaper_primary_candidate",
                "summary_json": summary.get("_summary_path", ""),
            }
        )
        backend_rows.append(
            {
                "case": case,
                "solver": backend.get("solver", ""),
                "matrix_free": backend.get("enable_matrix_free", ""),
                "fused_cuda": backend.get("enable_fused_cuda", ""),
                "matfree_gmg": backend.get("enable_matfree_gmg", ""),
                "gmg_levels": backend.get("matfree_gmg_levels", ""),
                "fine_smoother": backend.get("gmg_fine_smoother", ""),
                "smoother_type": backend.get("gmg_smoother_type", ""),
                "cg_tol": backend.get("cg_tol", ""),
                "cg_maxiter": backend.get("cg_maxiter", ""),
            }
        )
        convergence_rows.append(
            {
                "case": case,
                "iterations": summary.get("iters", ""),
                "final_compliance": final.get("compliance", ""),
                "final_grayness": final.get("grayness", ""),
                "rho_mean": final.get("rho_mean", ""),
                "rho_min": final.get("rho_min", ""),
                "rho_max": final.get("rho_max", ""),
                "final_outer_iters": final.get("outer_iters", ""),
                "linear_solve_cap_count": summary.get("linear_solve_cap_count", ""),
                "mesh_points": colored.get("mesh_points", summary.get("render_meta", {}).get("mesh_points", "")),
                "mesh_cells": colored.get("mesh_cells", summary.get("render_meta", {}).get("mesh_cells", "")),
            }
        )
        timing_rows.append(
            {
                "case": case,
                "n_elem": summary.get("n_elem", ""),
                "total_wall_s": summary.get("total_wall_s", ""),
                "first_iter_wall_s": summary.get("first_iter_wall_s", ""),
                "warm_iter_mean_wall_s_excluding_first": summary.get("warm_iter_mean_wall_s_excluding_first", ""),
                "warm_iter_max_wall_s_excluding_first": summary.get("warm_iter_max_wall_s_excluding_first", ""),
                "peak_or_final_gpu_mem_mb": final.get("gpu_mem_used_mb", ""),
                "gpu_total_mb": final.get("gpu_mem_total_mb", ""),
                "gpu_util_final_pct": final.get("gpu_util_pct", ""),
            }
        )
        for key in [
            "rho_final_npy",
            "disp_elem_npy",
            "history_csv",
            "render_png",
            "projection_png",
        ]:
            rel = summary.get(key)
            if rel:
                p = ROOT / rel
                manifest_rows.append(
                    {
                        "case": case,
                        "artifact": rel,
                        "kind": key,
                        "exists": p.exists(),
                        "size_bytes": p.stat().st_size if p.exists() else "",
                        "sha256": _sha256(p) if p.exists() and p.is_file() else "",
                    }
                )
        for colored_key in ["colored_render_png", "colored_clean_png"]:
            rel = colored.get(colored_key)
            if rel:
                p = ROOT / rel
                manifest_rows.append(
                    {
                        "case": case,
                        "artifact": rel,
                        "kind": colored_key,
                        "exists": p.exists(),
                        "size_bytes": p.stat().st_size if p.exists() else "",
                        "sha256": _sha256(p) if p.exists() and p.is_file() else "",
                    }
                )

    for artifact, kind in [
        (out_dir / "FIG_BC_LOAD_SCHEMATICS.png", "bc_load_schematic_png"),
        (out_dir / "TABLE_BC_LOADS.csv", "bc_load_table_csv"),
        (out_dir / "EQUATIONS_AND_ALGORITHMS.md", "equations_algorithms_md"),
    ]:
        if artifact.exists():
            manifest_rows.append(
                {
                    "case": "all",
                    "artifact": str(artifact.relative_to(ROOT)).replace("\\", "/"),
                    "kind": kind,
                    "exists": True,
                    "size_bytes": artifact.stat().st_size,
                    "sha256": _sha256(artifact) if artifact.is_file() else "",
                }
            )

    colored_summary = _read_json(colored_dir / "colored_render_summary.json")
    for key, kind in [
        ("colored_panel_png", "colored_panel_png"),
        ("colored_panel_shared_colorbar_png", "colored_panel_shared_colorbar_png"),
    ]:
        rel = colored_summary.get(key)
        if rel:
            p = ROOT / rel
            manifest_rows.append(
                {
                    "case": "all",
                    "artifact": rel,
                    "kind": kind,
                    "exists": p.exists(),
                    "size_bytes": p.stat().st_size if p.exists() else "",
                    "sha256": _sha256(p) if p.exists() and p.is_file() else "",
                }
            )

    for label, suite_rel in CAP_DIAGNOSTIC_SUITES:
        diag_suite = ROOT / suite_rel
        for case in cases:
            summary = _read_json(diag_suite / case / "summary.json")
            if not summary:
                continue
            final = summary.get("final", {})
            backend = summary.get("backend", {})
            cap_diag_rows.append(
                {
                    "suite": label,
                    "case": case,
                    "dims": "x".join(str(v) for v in summary.get("dims", [])),
                    "outer_solver": backend.get("gmg_outer_solver") or final.get("outer_solver", ""),
                    "cg_tol": backend.get("cg_tol", ""),
                    "cg_maxiter": backend.get("cg_maxiter", ""),
                    "linear_solve_cap_count": summary.get("linear_solve_cap_count", ""),
                    "warm_iter_mean_wall_s_excluding_first": summary.get("warm_iter_mean_wall_s_excluding_first", ""),
                    "final_outer_iters": final.get("outer_iters", ""),
                    "final_compliance": final.get("compliance", ""),
                    "final_grayness": final.get("grayness", ""),
                    "summary_json": str((diag_suite / case / "summary.json").relative_to(ROOT)).replace("\\", "/"),
                }
            )

    _write_csv(
        out_dir / "TABLE_SAMPLE_MATRIX.csv",
        sample_rows,
        ["case", "bvp", "volume_fraction", "dims", "n_elem", "status", "main_story_use", "summary_json"],
    )
    _write_csv(
        out_dir / "TABLE_BACKEND_CONFIG.csv",
        backend_rows,
        [
            "case",
            "solver",
            "matrix_free",
            "fused_cuda",
            "matfree_gmg",
            "gmg_levels",
            "fine_smoother",
            "smoother_type",
            "cg_tol",
            "cg_maxiter",
        ],
    )
    _write_csv(
        out_dir / "TABLE_CONVERGENCE_VISUAL.csv",
        convergence_rows,
        [
            "case",
            "iterations",
            "final_compliance",
            "final_grayness",
            "rho_mean",
            "rho_min",
            "rho_max",
            "final_outer_iters",
            "linear_solve_cap_count",
            "mesh_points",
            "mesh_cells",
        ],
    )
    _write_csv(
        out_dir / "TABLE_TIMING_MEMORY.csv",
        timing_rows,
        [
            "case",
            "n_elem",
            "total_wall_s",
            "first_iter_wall_s",
            "warm_iter_mean_wall_s_excluding_first",
            "warm_iter_max_wall_s_excluding_first",
            "peak_or_final_gpu_mem_mb",
            "gpu_total_mb",
            "gpu_util_final_pct",
        ],
    )
    _write_csv(
        out_dir / "TABLE_CAP_DIAGNOSTICS.csv",
        cap_diag_rows,
        [
            "suite",
            "case",
            "dims",
            "outer_solver",
            "cg_tol",
            "cg_maxiter",
            "linear_solve_cap_count",
            "warm_iter_mean_wall_s_excluding_first",
            "final_outer_iters",
            "final_compliance",
            "final_grayness",
            "summary_json",
        ],
    )
    cap_diag_path = out_dir / "TABLE_CAP_DIAGNOSTICS.csv"
    manifest_rows.append(
        {
            "case": "all",
            "artifact": str(cap_diag_path.relative_to(ROOT)).replace("\\", "/"),
            "kind": "cap_diagnostics_csv",
            "exists": cap_diag_path.exists(),
            "size_bytes": cap_diag_path.stat().st_size if cap_diag_path.exists() else "",
            "sha256": _sha256(cap_diag_path) if cap_diag_path.exists() and cap_diag_path.is_file() else "",
        }
    )
    verification_dir = ROOT / "rerun_outputs" / "tool_paper_verification"
    verification_rows = []
    for filename, kind in VERIFICATION_FILES:
        src = verification_dir / filename
        dst = out_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        verification_rows.append(
            {
                "file": filename,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "included": dst.exists(),
                "kind": kind,
            }
        )
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "kind": kind,
                "exists": dst.exists(),
                "size_bytes": dst.stat().st_size if dst.exists() else "",
                "sha256": _sha256(dst) if dst.exists() and dst.is_file() else "",
            }
        )
    admissibility_dir = ROOT / "rerun_outputs" / "tool_paper_admissibility"
    admissibility_rows = []
    for filename, kind in ADMISSIBILITY_FILES:
        src = admissibility_dir / filename
        dst = out_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        admissibility_rows.append(
            {
                "file": filename,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "included": dst.exists(),
                "kind": kind,
            }
        )
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "kind": kind,
                "exists": dst.exists(),
                "size_bytes": dst.stat().st_size if dst.exists() else "",
                "sha256": _sha256(dst) if dst.exists() and dst.is_file() else "",
            }
        )
    baseline_dir = ROOT / "rerun_outputs" / "tool_paper_baselines"
    baseline_rows = []
    for filename, kind in BASELINE_FILES:
        src = baseline_dir / filename
        dst = out_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        baseline_rows.append(
            {
                "file": filename,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "included": dst.exists(),
                "kind": kind,
            }
        )
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(dst.relative_to(ROOT)).replace("\\", "/"),
                "kind": kind,
                "exists": dst.exists(),
                "size_bytes": dst.stat().st_size if dst.exists() else "",
                "sha256": _sha256(dst) if dst.exists() and dst.is_file() else "",
            }
        )
    residual_rows = []
    residual_summary_path = ROOT / "rerun_outputs" / "tool_paper_residual_smoke" / "batch_summary.json"
    residual_summary = _read_json(residual_summary_path)
    residual_items = []
    if residual_summary.get("warmup"):
        residual_items.append(("warmup", residual_summary["warmup"]))
    for item in residual_summary.get("summaries", []):
        residual_items.append(("requested", item))
    for role, item in residual_items:
        final = item.get("final", {})
        residual_rows.append(
            {
                "role": role,
                "case": item.get("case", ""),
                "dims": "x".join(str(v) for v in item.get("dims", [])),
                "n_elem": item.get("n_elem", ""),
                "iters": item.get("iters", ""),
                "cg_tol": item.get("backend", {}).get("cg_tol", ""),
                "cg_maxiter": item.get("backend", {}).get("cg_maxiter", ""),
                "final_outer_iters": final.get("outer_iters", ""),
                "final_linear_relative_residual": final.get("linear_relative_residual", ""),
                "linear_residual_available": final.get("linear_residual_available", ""),
                "linear_solve_cap_count": item.get("linear_solve_cap_count", ""),
                "history_csv": item.get("history_csv", ""),
                "summary_json": str((ROOT / item.get("history_csv", "")).parent.joinpath("summary.json").relative_to(ROOT)).replace("\\", "/")
                if item.get("history_csv")
                else "",
            }
        )
    if residual_rows:
        residual_table_path = out_dir / "TABLE_RESIDUAL_INSTRUMENTATION_SMOKE.csv"
        _write_csv(
            residual_table_path,
            residual_rows,
            [
                "role",
                "case",
                "dims",
                "n_elem",
                "iters",
                "cg_tol",
                "cg_maxiter",
                "final_outer_iters",
                "final_linear_relative_residual",
                "linear_residual_available",
                "linear_solve_cap_count",
                "history_csv",
                "summary_json",
            ],
        )
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(residual_table_path.relative_to(ROOT)).replace("\\", "/"),
                "kind": "residual_instrumentation_smoke_csv",
                "exists": residual_table_path.exists(),
                "size_bytes": residual_table_path.stat().st_size if residual_table_path.exists() else "",
                "sha256": _sha256(residual_table_path) if residual_table_path.exists() and residual_table_path.is_file() else "",
            }
        )
        if residual_summary_path.exists():
            copied_summary = out_dir / "residual_smoke_batch_summary.json"
            shutil.copy2(residual_summary_path, copied_summary)
            manifest_rows.append(
                {
                    "case": "all",
                    "artifact": str(copied_summary.relative_to(ROOT)).replace("\\", "/"),
                    "kind": "residual_smoke_summary_json",
                    "exists": copied_summary.exists(),
                    "size_bytes": copied_summary.stat().st_size if copied_summary.exists() else "",
                    "sha256": _sha256(copied_summary) if copied_summary.exists() and copied_summary.is_file() else "",
                }
            )
    production_residual_rows = []
    production_residual_summary_path = ROOT / "rerun_outputs" / "tool_paper_residual_pcg_tol1e5" / "batch_summary.json"
    production_residual_summary = _read_json(production_residual_summary_path)
    for item in production_residual_summary.get("summaries", []):
        final = item.get("final", {})
        iters = float(item.get("iters", 0) or 0)
        cap_count = float(item.get("linear_solve_cap_count", 0) or 0)
        production_residual_rows.append(
            {
                "case": item.get("case", ""),
                "dims": "x".join(str(v) for v in item.get("dims", [])),
                "n_elem": item.get("n_elem", ""),
                "iters": item.get("iters", ""),
                "cg_tol": item.get("backend", {}).get("cg_tol", ""),
                "cg_maxiter": item.get("backend", {}).get("cg_maxiter", ""),
                "cap_count": item.get("linear_solve_cap_count", ""),
                "cap_fraction": cap_count / iters if iters else "",
                "final_outer_iters": final.get("outer_iters", ""),
                "final_linear_relative_residual": final.get("linear_relative_residual", ""),
                "linear_residual_available": final.get("linear_residual_available", ""),
                "final_compliance": final.get("compliance", ""),
                "final_grayness": final.get("grayness", ""),
                "warm_iter_mean_wall_s_excluding_first": item.get("warm_iter_mean_wall_s_excluding_first", ""),
                "history_csv": item.get("history_csv", ""),
                "summary_json": str((ROOT / item.get("history_csv", "")).parent.joinpath("summary.json").relative_to(ROOT)).replace("\\", "/")
                if item.get("history_csv")
                else "",
            }
        )
    if production_residual_rows:
        production_residual_table_path = out_dir / "TABLE_PRODUCTION_RESIDUAL_DIAGNOSTICS.csv"
        _write_csv(
            production_residual_table_path,
            production_residual_rows,
            [
                "case",
                "dims",
                "n_elem",
                "iters",
                "cg_tol",
                "cg_maxiter",
                "cap_count",
                "cap_fraction",
                "final_outer_iters",
                "final_linear_relative_residual",
                "linear_residual_available",
                "final_compliance",
                "final_grayness",
                "warm_iter_mean_wall_s_excluding_first",
                "history_csv",
                "summary_json",
            ],
        )
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(production_residual_table_path.relative_to(ROOT)).replace("\\", "/"),
                "kind": "production_residual_diagnostics_csv",
                "exists": production_residual_table_path.exists(),
                "size_bytes": production_residual_table_path.stat().st_size if production_residual_table_path.exists() else "",
                "sha256": _sha256(production_residual_table_path)
                if production_residual_table_path.exists() and production_residual_table_path.is_file()
                else "",
            }
        )
        copied_summary = out_dir / "production_residual_pcg_tol1e5_batch_summary.json"
        shutil.copy2(production_residual_summary_path, copied_summary)
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(copied_summary.relative_to(ROOT)).replace("\\", "/"),
                "kind": "production_residual_summary_json",
                "exists": copied_summary.exists(),
                "size_bytes": copied_summary.stat().st_size if copied_summary.exists() else "",
                "sha256": _sha256(copied_summary) if copied_summary.exists() and copied_summary.is_file() else "",
            }
        )
    fgmres_residual_rows = []
    fgmres_residual_summary_path = ROOT / "rerun_outputs" / "tool_paper_fgmres_residual_diag" / "batch_summary.json"
    fgmres_residual_summary = _read_json(fgmres_residual_summary_path)
    for item in fgmres_residual_summary.get("summaries", []):
        final = item.get("final", {})
        iters = float(item.get("iters", 0) or 0)
        cap_count = float(item.get("linear_solve_cap_count", 0) or 0)
        fgmres_residual_rows.append(
            {
                "case": item.get("case", ""),
                "dims": "x".join(str(v) for v in item.get("dims", [])),
                "n_elem": item.get("n_elem", ""),
                "iters": item.get("iters", ""),
                "tol": item.get("backend", {}).get("cg_tol", ""),
                "max_krylov": item.get("backend", {}).get("cg_maxiter", ""),
                "cap_count": item.get("linear_solve_cap_count", ""),
                "cap_fraction": cap_count / iters if iters else "",
                "final_outer_iters": final.get("outer_iters", ""),
                "final_linear_relative_residual": final.get("linear_relative_residual", ""),
                "final_compliance": final.get("compliance", ""),
                "final_grayness": final.get("grayness", ""),
                "warm_iter_mean_wall_s_excluding_first": item.get("warm_iter_mean_wall_s_excluding_first", ""),
                "history_csv": item.get("history_csv", ""),
                "summary_json": str((ROOT / item.get("history_csv", "")).parent.joinpath("summary.json").relative_to(ROOT)).replace("\\", "/")
                if item.get("history_csv")
                else "",
            }
        )
    if fgmres_residual_rows:
        fgmres_residual_table_path = out_dir / "TABLE_FGMRES_RESIDUAL_DIAGNOSTICS.csv"
        _write_csv(
            fgmres_residual_table_path,
            fgmres_residual_rows,
            [
                "case",
                "dims",
                "n_elem",
                "iters",
                "tol",
                "max_krylov",
                "cap_count",
                "cap_fraction",
                "final_outer_iters",
                "final_linear_relative_residual",
                "final_compliance",
                "final_grayness",
                "warm_iter_mean_wall_s_excluding_first",
                "history_csv",
                "summary_json",
            ],
        )
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(fgmres_residual_table_path.relative_to(ROOT)).replace("\\", "/"),
                "kind": "fgmres_residual_diagnostics_csv",
                "exists": fgmres_residual_table_path.exists(),
                "size_bytes": fgmres_residual_table_path.stat().st_size if fgmres_residual_table_path.exists() else "",
                "sha256": _sha256(fgmres_residual_table_path)
                if fgmres_residual_table_path.exists() and fgmres_residual_table_path.is_file()
                else "",
            }
        )
        copied_summary = out_dir / "fgmres_residual_diag_batch_summary.json"
        shutil.copy2(fgmres_residual_summary_path, copied_summary)
        manifest_rows.append(
            {
                "case": "all",
                "artifact": str(copied_summary.relative_to(ROOT)).replace("\\", "/"),
                "kind": "fgmres_residual_summary_json",
                "exists": copied_summary.exists(),
                "size_bytes": copied_summary.stat().st_size if copied_summary.exists() else "",
                "sha256": _sha256(copied_summary) if copied_summary.exists() and copied_summary.is_file() else "",
            }
        )
    expanded_rows = []
    expanded_dir = ROOT / "rerun_outputs" / "tool_paper_expanded_benchmark_summary"
    for filename, kind in EXPANDED_BENCHMARK_FILES:
        src = expanded_dir / filename
        dst = out_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        expanded_rows.append(
            {
                "file": filename,
                "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                "included": dst.exists(),
                "kind": kind,
            }
        )
        if dst.exists():
            manifest_rows.append(
                {
                    "case": "all",
                    "artifact": str(dst.relative_to(ROOT)).replace("\\", "/"),
                    "kind": kind,
                    "exists": True,
                    "size_bytes": dst.stat().st_size,
                    "sha256": _sha256(dst) if dst.is_file() else "",
                }
            )
    _write_csv(
        out_dir / "ARTIFACT_MANIFEST.csv",
        manifest_rows,
        ["case", "artifact", "kind", "exists", "size_bytes", "sha256"],
    )
    return {
        "sample_rows": sample_rows,
        "backend_rows": backend_rows,
        "convergence_rows": convergence_rows,
        "timing_rows": timing_rows,
        "cap_diag_rows": cap_diag_rows,
        "verification_rows": verification_rows,
        "admissibility_rows": admissibility_rows,
        "baseline_rows": baseline_rows,
        "residual_rows": residual_rows,
        "production_residual_rows": production_residual_rows,
        "fgmres_residual_rows": fgmres_residual_rows,
        "expanded_rows": expanded_rows,
        "manifest_rows": manifest_rows,
    }


def write_equations_algorithms(out_dir: Path) -> None:
    text = r"""# Tool-Paper Equations and Algorithms

This file records equations and algorithms used by the evidence package.

## Core Equations

1. Compliance objective:
   \[
   \min_{\rho} \; c(\rho) = \mathbf{f}^{T}\mathbf{u}(\rho)
   \]

2. Volume constraint and bounds:
   \[
   \frac{1}{N_e}\sum_{e=1}^{N_e}\rho_e \le V_f,\qquad \rho_{\min}\le\rho_e\le 1
   \]

3. SIMP interpolation:
   \[
   E_e(\rho_e)=E_{\min}+\rho_e^p(E_0-E_{\min})
   \]

4. Discrete equilibrium:
   \[
   \mathbf{K}(\rho)\mathbf{u}=\mathbf{f}
   \]

5. Element sensitivity:
   \[
   \frac{\partial c}{\partial \rho_e}
   = -p(E_0-E_{\min})\rho_e^{p-1}\mathbf{u}_e^T\mathbf{K}_e^0\mathbf{u}_e
   \]

6. OC-style update used in the evidence runner:
   \[
   \rho_e^{k+1}=\mathrm{clip}\left(
   \rho_e^k\sqrt{-\frac{\partial c/\partial\rho_e}{\lambda}},
   \rho_e^k-m,\rho_e^k+m
   \right)
   \]

7. Matrix-free operator action:
   \[
   \mathbf{y}_i = \sum_{e\in\mathcal{E}(i)} \mathbf{K}_e(\rho_e)\mathbf{u}_e
   \]

8. Preconditioned Krylov solve:
   \[
   \mathbf{M}^{-1}\mathbf{K}(\rho)\mathbf{u}=\mathbf{M}^{-1}\mathbf{f}
   \]

9. Evidence metrics:
   \[
   g(\rho)=4\,\mathrm{mean}(\rho(1-\rho)),\qquad
   t_{\mathrm{warm}}=\mathrm{mean}_{k>1}(t_k)
   \]

## Algorithm 1: Evidence-Gated GPU SIMP Workflow

1. Define a new problem setting: mesh, supports, loads, target volume fraction, and filter radius.
2. Build element DOF connectivity and free-DOF maps.
3. Initialize a uniform density field at the target volume fraction.
4. For each design iteration, solve equilibrium on GPU using the accepted SolverV4 backend.
5. Compute compliance and sensitivities.
6. Apply the bounded OC update.
7. Record compliance, grayness, linear iterations, wall time, and GPU memory/utilization.
8. Export final density, displacement scalar, history table, render metadata, and hashes.

## Algorithm 2: SolverV4 Matrix-Free GMG Equilibrium Solve

1. Construct the fused matrix-free element operator for the current density field.
2. Build or refresh the Galerkin matrix-free GMG hierarchy using the configured grid dimensions.
3. Select the outer Krylov solver and GMG smoother configuration.
4. Use the previous displacement as a warm start when available.
5. Run the preconditioned Krylov solve to the configured tolerance or iteration cap.
6. Save the final free displacement vector for the next SIMP iteration and for colored rendering.

## Algorithm 3: Evidence Bundle Generation

1. Read each case summary, history, density array, displacement scalar, and render metadata.
2. Generate sample, backend, convergence/visual, timing/memory, and manifest tables.
3. Hash every tracked artifact.
4. Keep Phase 3 registry outputs in verification-only sections.
5. Promote only new SolverV4 cases to the main figure/table candidate set.
"""
    (out_dir / "EQUATIONS_AND_ALGORITHMS.md").write_text(text, encoding="utf-8")


def write_readme(out_dir: Path, suite: Path, colored_dir: Path, tables: dict) -> None:
    missing = [row for row in tables["sample_rows"] if row.get("status") != "present"]
    cap_rows = [
        row
        for row in tables["convergence_rows"]
        if str(row.get("linear_solve_cap_count", "")) not in {"", "0"}
    ]
    text = [
        "# Tool-Paper Evidence Package",
        "",
        "This package records run-level, numerical-verification, and provenance evidence for the GPU topology-optimization toolkit manuscript.",
        "",
        "## Source Runs",
        f"- Warm/larger SolverV4 suite: `{suite.relative_to(ROOT).as_posix()}`",
        f"- Colored render suite: `{colored_dir.relative_to(ROOT).as_posix()}`",
        "",
        "## Tables",
        "- `TABLE_SAMPLE_MATRIX.csv`",
        "- `TABLE_BACKEND_CONFIG.csv`",
        "- `TABLE_CONVERGENCE_VISUAL.csv`",
        "- `TABLE_TIMING_MEMORY.csv`",
        "- `TABLE_CAP_DIAGNOSTICS.csv`",
        "- `TABLE_OPERATOR_VERIFICATION.csv`",
        "- `TABLE_SENSITIVITY_VERIFICATION.csv`",
        "- `TABLE_FILTER_VERIFICATION.csv`",
        "- `verification_summary.json`",
        "- `TABLE_ADMISSIBILITY_CAP_STATUS.csv`",
        "- `admissibility_summary.json`",
        "- `TABLE_BASELINE_ABLATION.csv`",
        "- `baseline_ablation_summary.json`",
        "- `TABLE_RESIDUAL_INSTRUMENTATION_SMOKE.csv`",
        "- `residual_smoke_batch_summary.json`",
        "- `TABLE_PRODUCTION_RESIDUAL_DIAGNOSTICS.csv`",
        "- `production_residual_pcg_tol1e5_batch_summary.json`",
        "- `TABLE_FGMRES_RESIDUAL_DIAGNOSTICS.csv`",
        "- `fgmres_residual_diag_batch_summary.json`",
        "- `TABLE_EXPANDED_PRODUCTION_BENCHMARKS.csv`",
        "- `TABLE_EXPANDED_STRESS_DIAGNOSTICS.csv`",
        "- `TABLE_EXPANDED_ALL_RUNS.csv`",
        "- `expanded_benchmark_summary.json`",
        "- `FIG_PRODUCTION_HISTORY_DIAGNOSTICS.png/.pdf`",
        "- `TABLE_BC_LOADS.csv`",
        "- `ARTIFACT_MANIFEST.csv`",
        "",
        "## Methods Scaffold",
        "- `EQUATIONS_AND_ALGORITHMS.md`",
        "",
        "## Main-Story Boundary",
        "Phase 3 registry topologies remain verification-only. Main figures and primary tables should use the new SolverV4 cases unless a later run supersedes them.",
        "",
        "## Remaining Draft Blockers",
    ]
    if missing:
        text.append("- Missing one or more requested new SolverV4 case summaries.")
    if cap_rows:
        text.append("- At least one run still hit the configured linear-iteration cap; treat performance claims as diagnostic until cap counts are zero or explicitly justified.")
    if not missing and not cap_rows:
        text.append("- No missing case summaries or solver-cap blockers detected in the current table build.")
    text.append("- Visually inspect and, if needed, refine BC/load schematics before promoting them to manuscript figures.")
    text.append("- Visually inspect final colored renders before promoting them to the manuscript figure set.")
    (out_dir / "README_EVIDENCE_PACKAGE.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="rerun_outputs/tool_paper_new_topology_solverv4_warm_large")
    parser.add_argument("--colored", default="rerun_outputs/tool_paper_new_topology_solverv4_warm_large_colored")
    parser.add_argument("--out", default="rerun_outputs/tool_paper_evidence_bundle/toolpaper_artifact_pack")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    suite = ROOT / args.suite
    colored_dir = ROOT / args.colored
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = args.case or DEFAULT_CASES
    write_equations_algorithms(out_dir)
    tables = build_tables(out_dir, suite, colored_dir, cases)
    write_readme(out_dir, suite, colored_dir, tables)
    summary = {
        "out": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "suite": str(suite.relative_to(ROOT)).replace("\\", "/"),
        "colored": str(colored_dir.relative_to(ROOT)).replace("\\", "/"),
        "cases": cases,
        "table_counts": {key: len(value) for key, value in tables.items()},
        "files": [
            "README_EVIDENCE_PACKAGE.md",
            "TABLE_SAMPLE_MATRIX.csv",
            "TABLE_BACKEND_CONFIG.csv",
            "TABLE_CONVERGENCE_VISUAL.csv",
            "TABLE_TIMING_MEMORY.csv",
            "TABLE_CAP_DIAGNOSTICS.csv",
            "TABLE_OPERATOR_VERIFICATION.csv",
            "TABLE_SENSITIVITY_VERIFICATION.csv",
            "TABLE_FILTER_VERIFICATION.csv",
            "verification_summary.json",
            "TABLE_ADMISSIBILITY_CAP_STATUS.csv",
            "admissibility_summary.json",
            "TABLE_BASELINE_ABLATION.csv",
            "baseline_ablation_summary.json",
            "TABLE_RESIDUAL_INSTRUMENTATION_SMOKE.csv",
            "residual_smoke_batch_summary.json",
            "TABLE_PRODUCTION_RESIDUAL_DIAGNOSTICS.csv",
            "production_residual_pcg_tol1e5_batch_summary.json",
            "TABLE_FGMRES_RESIDUAL_DIAGNOSTICS.csv",
            "fgmres_residual_diag_batch_summary.json",
            "TABLE_EXPANDED_PRODUCTION_BENCHMARKS.csv",
            "TABLE_EXPANDED_STRESS_DIAGNOSTICS.csv",
            "TABLE_EXPANDED_ALL_RUNS.csv",
            "expanded_benchmark_summary.json",
            "FIG_PRODUCTION_HISTORY_DIAGNOSTICS.png",
            "FIG_PRODUCTION_HISTORY_DIAGNOSTICS.pdf",
            "TABLE_BC_LOADS.csv",
            "FIG_BC_LOAD_SCHEMATICS.png",
            "ARTIFACT_MANIFEST.csv",
            "EQUATIONS_AND_ALGORITHMS.md",
        ],
    }
    (out_dir / "artifact_pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
