from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOLVER_SOURCES = [
    "rerun_outputs/tool_paper_initial/uniform_probe_summary.csv",
    "rerun_outputs/tool_paper_non_cantilever/uniform_probe_summary.csv",
    "rerun_outputs/tool_paper_non_cantilever_fgmres/uniform_probe_summary.csv",
    "rerun_outputs/tool_paper_even_non_cantilever/uniform_probe_summary.csv",
    "rerun_outputs/tool_paper_scaling_cantilever/uniform_probe_summary.csv",
]

DEFAULT_REPEATABILITY_SOURCE = (
    "rerun_outputs/tool_paper_repeatability/repeatability_summary.csv"
)

DEFAULT_END_TO_END_SOURCE = "rerun_outputs/tool_paper_end_to_end_simp_60iter/summary.json"
DEFAULT_GPU_SIMP_64K_SOURCE = "rerun_outputs/tool_paper_gpu_simp_visual_64k_fixed/summary.json"
DEFAULT_FRESH_216K_SOURCE = "rerun_outputs/tool_paper_fresh_216k/cantilever_216k_meta.json"
DEFAULT_FRESH_216K_RENDER_SOURCE = (
    "rerun_outputs/tool_paper_fresh_216k_standard_render/"
    "fresh_cantilever_216k_render_meta.json"
)
DEFAULT_FRESH_MULTI_BC_SOURCES = [
    (
        "bridge_216k",
        "rerun_outputs/tool_paper_fresh_multi_bc/bridge_216k_meta.json",
        "rerun_outputs/tool_paper_fresh_multi_bc_standard_render/fresh_bridge_216k_render_meta.json",
    ),
    (
        "torsion_500k",
        "rerun_outputs/tool_paper_fresh_multi_bc/torsion_500k_meta.json",
        "rerun_outputs/tool_paper_fresh_multi_bc_standard_render/fresh_torsion_500k_render_meta.json",
    ),
    (
        "doubleclamp_216k",
        "rerun_outputs/tool_paper_fresh_multi_bc/doubleclamp_216k_meta.json",
        "rerun_outputs/tool_paper_fresh_multi_bc_standard_render/fresh_doubleclamp_216k_render_meta.json",
    ),
]

DEFAULT_NEW_TOPOLOGY_SOURCES = [
    (
        "tool_long_cantilever_vf16",
        "rerun_outputs/tool_paper_new_topology_solverv4_warm_large/tool_long_cantilever_vf16/summary.json",
    ),
    (
        "tool_portal_bridge_vf18",
        "rerun_outputs/tool_paper_new_topology_solverv4_warm_large/tool_portal_bridge_vf18/summary.json",
    ),
    (
        "tool_asymmetric_bracket_vf14",
        "rerun_outputs/tool_paper_new_topology_solverv4_warm_large/tool_asymmetric_bracket_vf14/summary.json",
    ),
]

DEFAULT_FIGURES = [
    {
        "artifact": "rerun_outputs/tool_paper_topology_3d/F11_main_context.png",
        "type": "true_3d_topology_render",
        "status": "generated_and_visually_checked",
        "use": "primary evidence that retained density fields render as true 3D topology panels",
    },
    {
        "artifact": "rerun_outputs/tool_paper_topology_3d/F11_gallery.pdf",
        "type": "true_3d_topology_render_gallery",
        "status": "generated",
        "use": "panel gallery for inspection and later publication polish",
    },
    {
        "artifact": "rerun_outputs/tool_paper_scaling_cantilever/residual_histories.png",
        "type": "solver_residual_plot",
        "status": "generated_and_visually_checked",
        "use": "internal evidence for cantilever scaling convergence",
    },
    {
        "artifact": "rerun_outputs/tool_paper_repeatability/repeatability_timing.png",
        "type": "repeatability_timing_plot",
        "status": "generated_and_visually_checked",
        "use": "internal evidence for warm solve-time stability",
    },
    {
        "artifact": "rerun_outputs/tool_paper_topology_figs/topology_density_snapshot_gallery.png",
        "type": "density_projection_gallery",
        "status": "generated_and_visually_checked",
        "use": "quick density-array provenance check, not a substitute for true 3D render",
    },
    {
        "artifact": "rerun_outputs/tool_paper_end_to_end_simp_60iter/fresh_cantilever_3d_render.png",
        "type": "fresh_end_to_end_3d_render",
        "status": "generated_and_visually_checked",
        "use": "fresh bounded small 3D SIMP run rendered through marching cubes; provenance-only, not publication visual quality",
    },
    {
        "artifact": "rerun_outputs/tool_paper_end_to_end_simp_60iter/compliance_history.png",
        "type": "fresh_end_to_end_compliance_history",
        "status": "generated",
        "use": "fresh bounded small 3D SIMP 60-iteration compliance decrease check",
    },
    {
        "artifact": "rerun_outputs/tool_paper_end_to_end_simp_60iter/density_projection_gallery.png",
        "type": "fresh_end_to_end_density_projection",
        "status": "generated_and_visually_checked",
        "use": "fresh bounded small 3D SIMP 60-iteration density projection check",
    },
    {
        "artifact": "rerun_outputs/tool_paper_gpu_simp_visual_64k_fixed/gpu_simp_render.png",
        "type": "fresh_gpu_simp_64k_render",
        "status": "generated_and_visually_checked_diagnostic_only",
        "use": "fresh 64k GPU SolverV4 SIMP visual diagnostic; not final manuscript visual because late solves hit PCG cap and no filter/projection polish is applied",
    },
    {
        "artifact": "rerun_outputs/tool_paper_fresh_216k/F12_cantilever_216k_profile.png",
        "type": "phase3_registry_simp_216k_profile",
        "status": "generated_and_visually_checked_verification_only",
        "use": "Phase 3 registry 216k production SIMP density visual from the staged toolkit path; verification-only because topology setting is not new for this tool paper",
    },
    {
        "artifact": "rerun_outputs/tool_paper_fresh_216k_standard_render/fresh_cantilever_216k.png",
        "type": "phase3_registry_simp_216k_standard_render",
        "status": "generated_and_visually_checked_verification_only",
        "use": "Phase 3 registry 216k production SIMP density rendered with the paper-style marching-cubes/PyVista path; renderer/provenance verification only",
    },
    {
        "artifact": "rerun_outputs/tool_paper_fresh_multi_bc_standard_render/fresh_bridge_216k.png",
        "type": "phase3_registry_simp_bridge_216k_standard_render",
        "status": "generated_and_visually_checked_verification_only",
        "use": "Phase 3 registry bridge/support-transfer production SIMP case; verification-only, not a main new topology figure",
    },
    {
        "artifact": "rerun_outputs/tool_paper_fresh_multi_bc_standard_render/fresh_torsion_500k.png",
        "type": "phase3_registry_simp_torsion_500k_standard_render",
        "status": "generated_and_visually_checked_verification_only_visually_weak",
        "use": "Phase 3 registry torsion production SIMP case; verification-only and visually less illustrative",
    },
    {
        "artifact": "rerun_outputs/tool_paper_fresh_multi_bc_standard_render/fresh_doubleclamp_216k.png",
        "type": "phase3_registry_simp_doubleclamp_216k_standard_render",
        "status": "generated_and_visually_checked_verification_only",
        "use": "Phase 3 registry double-clamped production SIMP case; verification-only, not a main new topology figure",
    },
    {
        "artifact": "rerun_outputs/tool_paper_new_topology_solverv4_suite/tool_long_cantilever_vf16/render.png",
        "type": "new_toolpaper_solverv4_render",
        "status": "generated_visual_candidate_needs_publication_polish",
        "use": "New tool-paper topology candidate run with SolverV4 fused CUDA + matrix-free GMG",
    },
    {
        "artifact": "rerun_outputs/tool_paper_new_topology_solverv4_suite/tool_portal_bridge_vf18/render.png",
        "type": "new_toolpaper_solverv4_render",
        "status": "generated_visual_candidate_needs_publication_polish",
        "use": "New tool-paper topology candidate run with SolverV4 fused CUDA + matrix-free GMG",
    },
    {
        "artifact": "rerun_outputs/tool_paper_new_topology_solverv4_suite/tool_asymmetric_bracket_vf14/render.png",
        "type": "new_toolpaper_solverv4_render",
        "status": "generated_visual_candidate_needs_publication_polish",
        "use": "New tool-paper topology candidate run with SolverV4 fused CUDA + matrix-free GMG",
    },
    {
        "artifact": "rerun_outputs/tool_paper_new_topology_solverv4_suite/new_toolpaper_solverv4_render_panel.png",
        "type": "new_toolpaper_solverv4_render_panel",
        "status": "generated_visual_review_panel",
        "use": "Three-case review panel for current SolverV4 new-topology visual quality; not a submission figure",
    },
    {
        "artifact": "rerun_outputs/tool_paper_new_topology_solverv4_warm_large_colored/new_toolpaper_solverv4_colored_panel.png",
        "type": "new_toolpaper_solverv4_displacement_colored_panel",
        "status": "generated_visual_review_panel",
        "use": "Larger new SolverV4 cases colored by normalized element displacement magnitude; review figure, not final camera-polished manuscript art",
    },
    {
        "artifact": "rerun_outputs/tool_paper_new_topology_solverv4_warm_large_colored/new_toolpaper_solverv4_colored_panel_shared_colorbar.png",
        "type": "new_toolpaper_solverv4_displacement_colored_panel_shared_colorbar",
        "status": "generated_visual_review_panel",
        "use": "Shared-colorbar version of the larger new SolverV4 displacement-colored topology panel",
    },
    {
        "artifact": "rerun_outputs/tool_paper_evidence_bundle/toolpaper_artifact_pack/README_EVIDENCE_PACKAGE.md",
        "type": "toolpaper_artifact_pack_readme",
        "status": "generated",
        "use": "Evidence-package index for tables, algorithms, equations, and remaining blockers",
    },
    {
        "artifact": "rerun_outputs/tool_paper_evidence_bundle/toolpaper_artifact_pack/ARTIFACT_MANIFEST.csv",
        "type": "toolpaper_artifact_manifest",
        "status": "generated",
        "use": "Hash manifest for the new SolverV4 evidence artifacts",
    },
    {
        "artifact": "rerun_outputs/tool_paper_evidence_bundle/toolpaper_artifact_pack/FIG_BC_LOAD_SCHEMATICS.png",
        "type": "bc_load_schematic_panel",
        "status": "generated_visual_review_panel",
        "use": "Boundary-condition and loading schematic panel for the three new tool-paper cases",
    },
    {
        "artifact": "rerun_outputs/tool_paper_evidence_bundle/toolpaper_artifact_pack/TABLE_BC_LOADS.csv",
        "type": "bc_load_table",
        "status": "generated",
        "use": "Structured BC/load settings for the three new tool-paper cases",
    },
]


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_case(case: str) -> str:
    return case.strip()


def _case_entries(matrix: dict) -> list[dict]:
    entries = []
    for section in [
        "solver_smoke",
        "main_solver_validation",
        "scaling_probe",
        "stress_or_diagnostic",
        "do_not_use_as_primary_validation",
    ]:
        for item in matrix.get(section, []):
            entry = dict(item)
            entry["section"] = section
            entries.append(entry)
    return entries


def _solver_index(sources: list[str]) -> dict[tuple[str, str], dict]:
    rows_by_case: dict[tuple[str, str], dict] = {}
    for source in sources:
        for row in _read_csv(ROOT / source):
            row = dict(row)
            row["source_csv"] = source
            case = _norm_case(row.get("case", ""))
            outer = row.get("outer") or "pcg"
            rows_by_case[(case, outer)] = row
    return rows_by_case


def _best_solver_row(entry: dict, rows_by_case: dict[tuple[str, str], dict]) -> dict:
    case = _norm_case(entry["case"])
    wanted_outer = "fgmres" if entry.get("status") == "failed_fgmres" else "pcg"
    if (case, wanted_outer) in rows_by_case:
        return rows_by_case[(case, wanted_outer)]
    if (case, "pcg") in rows_by_case:
        return rows_by_case[(case, "pcg")]
    candidates = [row for (candidate, _), row in rows_by_case.items() if candidate == case]
    return candidates[0] if candidates else {}


def build_solver_table(matrix: dict, solver_sources: list[str], repeatability_source: str) -> list[dict]:
    rows_by_case = _solver_index(solver_sources)
    repeat_rows = {
        _norm_case(row["case"]): row
        for row in _read_csv(ROOT / repeatability_source)
        if row.get("case")
    }

    table = []
    for entry in _case_entries(matrix):
        row = _best_solver_row(entry, rows_by_case)
        repeat = repeat_rows.get(_norm_case(entry["case"]), {})
        table.append(
            {
                "section": entry["section"],
                "case": entry["case"],
                "role": entry.get("role", ""),
                "status": entry.get("status", ""),
                "n_elem": row.get("n_elem", ""),
                "nelx": row.get("nelx", ""),
                "nely": row.get("nely", ""),
                "nelz": row.get("nelz", ""),
                "n_levels": row.get("n_levels", ""),
                "outer": row.get("outer") or "pcg",
                "iters": row.get("iters", ""),
                "converged": row.get("converged", ""),
                "rel_residual": row.get("rel_residual", ""),
                "solve_s": row.get("solve_s", ""),
                "setup_s": row.get("setup_s", ""),
                "repeat_solve_s_mean": repeat.get("solve_s_mean", ""),
                "repeat_solve_s_stdev": repeat.get("solve_s_stdev", ""),
                "repeat_all_converged": repeat.get("all_converged", ""),
                "source_csv": row.get("source_csv", ""),
                "note": entry.get("note") or entry.get("reason", ""),
            }
        )
    return table


def build_figure_manifest(figures: list[dict]) -> list[dict]:
    rows = []
    for item in figures:
        path = ROOT / item["artifact"]
        rows.append(
            {
                "artifact": item["artifact"],
                "type": item["type"],
                "status": item["status"],
                "use": item["use"],
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else "",
                "sha256": _sha256(path) if path.exists() else "",
            }
        )
    return rows


def _md_table(rows: list[dict], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _load_json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _max_history_value(path: Path, field: str):
    if not path.exists():
        return None
    vals = []
    for row in _read_csv(path):
        raw = row.get(field)
        if raw in (None, ""):
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return max(vals) if vals else None


def _load_fresh_multi_bc() -> list[dict]:
    rows = []
    for key, meta_source, render_source in DEFAULT_FRESH_MULTI_BC_SOURCES:
        meta = _load_json_or_none(ROOT / meta_source)
        render = _load_json_or_none(ROOT / render_source)
        if not meta:
            continue
        row = dict(meta)
        row["key"] = key
        row["meta_source"] = meta_source
        row["render_source"] = render_source
        if render:
            row["render_density_mean"] = render.get("rho_mean")
            row["render_mesh_points"] = render.get("mesh_points")
            row["render_mesh_cells"] = render.get("mesh_cells")
            row["render_png"] = render.get("outputs", {}).get("trimmed_png")
        rows.append(row)
    return rows


def _load_new_topology_suite() -> list[dict]:
    rows = []
    for key, source in DEFAULT_NEW_TOPOLOGY_SOURCES:
        summary = _load_json_or_none(ROOT / source)
        if not summary:
            continue
        final = summary.get("final", {})
        backend = summary.get("backend", {})
        rows.append(
            {
                "case": key,
                "n_elem": summary.get("n_elem"),
                "volfrac": summary.get("volfrac"),
                "iters": summary.get("iters"),
                "final_compliance": final.get("compliance"),
                "final_grayness": final.get("grayness"),
                "final_outer_iters": final.get("outer_iters"),
                "linear_solve_cap_count": summary.get("linear_solve_cap_count"),
                "warm_iter_mean_s_excl_first": summary.get("warm_iter_mean_wall_s_excluding_first"),
                "first_iter_s": summary.get("first_iter_wall_s"),
                "render_mesh_cells": summary.get("render_meta", {}).get("mesh_cells"),
                "max_gpu_mem_mb": _max_history_value(ROOT / summary.get("history_csv", ""), "gpu_mem_used_mb"),
                "backend": backend.get("solver"),
                "fused_cuda": backend.get("enable_fused_cuda"),
                "matfree_gmg": backend.get("enable_matfree_gmg"),
                "render_png": summary.get("render_png"),
                "status": "larger_colored_visual_candidate_cap_diagnostic_needed"
                if summary.get("linear_solve_cap_count")
                else "larger_colored_visual_candidate",
            }
        )
    return rows


def write_markdown(
    out_dir: Path,
    solver_rows: list[dict],
    figure_rows: list[dict],
    end_to_end: dict | None,
    gpu_simp_64k: dict | None,
    fresh_216k: dict | None,
    fresh_216k_render: dict | None,
    fresh_multi_bc: list[dict],
    new_topology_suite: list[dict],
) -> None:
    pass_rows = [row for row in solver_rows if row["status"].startswith("passed")]
    failed_rows = [row for row in solver_rows if row["status"] == "failed"]
    repeat_rows = [row for row in solver_rows if row["repeat_solve_s_mean"]]

    md = [
        "# Consolidated Evidence Bundle",
        "",
        "## Verdict",
        "Evidence supports continuing setup and new-case evidence building, but not final performance claims or final manuscript figures. The main story should use the new SolverV4 tool-paper samples, not Phase 3 registry cases. Solver convergence is verified on the accepted validation matrix, and repeatability is stable for selected medium cases. Setup timing remains warmup-sensitive.",
        "",
        "## Passed Solver Cases",
        _md_table(
            pass_rows,
            [
                "section",
                "case",
                "role",
                "n_elem",
                "n_levels",
                "iters",
                "rel_residual",
                "solve_s",
            ],
        ),
        "",
        "## Repeatability Cases",
        _md_table(
            repeat_rows,
            [
                "case",
                "repeat_solve_s_mean",
                "repeat_solve_s_stdev",
                "repeat_all_converged",
                "iters",
                "rel_residual",
            ],
        ),
        "",
        "## Failed Or Excluded Cases",
        _md_table(
            failed_rows,
            ["case", "role", "n_elem", "n_levels", "outer", "rel_residual", "note"],
        ),
        "",
        "## Figure Manifest",
        _md_table(
            figure_rows,
            ["artifact", "type", "status", "exists", "size_bytes", "sha256"],
        ),
        "",
        "## Fresh End-to-End SIMP Probe",
    ]
    if end_to_end:
        md.extend(
            [
                f"- Case: `{end_to_end.get('case')}`",
                f"- Mesh: {end_to_end.get('nelx')}x{end_to_end.get('nely')}x{end_to_end.get('nelz')} ({end_to_end.get('n_elem')} elements)",
                f"- Iterations: {end_to_end.get('n_iter_reported')} bounded iterations",
                f"- Compliance: {end_to_end.get('final_compliance')}",
                f"- Volume fraction: {end_to_end.get('final_volume_fraction')}",
                f"- Grayness: {end_to_end.get('final_grayness')}",
                f"- Render: `{end_to_end.get('render_png')}`",
                f"- Note: {end_to_end.get('note')}",
            ]
        )
    else:
        md.append("- Not available.")
    md.extend(["", "## Fresh 64k GPU SIMP Visual Probe"])
    if gpu_simp_64k:
        md.extend(
            [
                f"- Case: `{gpu_simp_64k.get('case_label')}`",
                f"- Mesh: {gpu_simp_64k.get('nelx')}x{gpu_simp_64k.get('nely')}x{gpu_simp_64k.get('nelz')} ({gpu_simp_64k.get('n_elem')} elements)",
                f"- Iterations: {gpu_simp_64k.get('iters')}",
                f"- Final compliance: {gpu_simp_64k.get('final_compliance_solve')}",
                f"- Final volume fraction: {gpu_simp_64k.get('final_rho_mean')}",
                f"- Final grayness: {gpu_simp_64k.get('final_grayness')}",
                f"- Render mesh: {gpu_simp_64k.get('render_meta', {}).get('mesh_points')} points, {gpu_simp_64k.get('render_meta', {}).get('mesh_cells')} cells",
                f"- Render: `{gpu_simp_64k.get('render_png')}`",
                "- Note: Diagnostic visual only. It proves a meaningful fresh GPU SIMP run at 64k elements, but late iterations hit the 1000-iteration solver cap and the loop has no density-filter/projection polish.",
            ]
        )
    else:
        md.append("- Not available.")

    md.extend(["", "## Internal Lineage Controls Excluded From Main Story"])
    md.append(
        "- Phase 3 registry density/rerun artifacts are retained only as internal verification controls. They should not be used in the abstract, contribution list, main figures, or primary tables."
    )
    md.extend(["", "### Phase 3 Registry 216k Production SIMP Density"])
    if fresh_216k:
        md.extend(
            [
                f"- Case: `{fresh_216k.get('prob_key')}`",
                f"- Mesh: {fresh_216k.get('nelx')}x{fresh_216k.get('nely')}x{fresh_216k.get('nelz')} ({fresh_216k.get('n_elem')} elements)",
                f"- SIMP iterations: {fresh_216k.get('max_iter')}",
                f"- Best iteration: {fresh_216k.get('best_iteration')}",
                f"- Final compliance: {fresh_216k.get('final_compliance')}",
                f"- Best compliance: {fresh_216k.get('best_compliance')}",
                f"- Final grayness: {fresh_216k.get('final_grayness')}",
                f"- Best valid: {fresh_216k.get('best_is_valid')}",
            ]
        )
        if fresh_216k_render:
            md.extend(
                [
                    f"- Standard render density mean: {fresh_216k_render.get('rho_mean')}",
                    f"- Standard render mesh: {fresh_216k_render.get('mesh_points')} points, {fresh_216k_render.get('mesh_cells')} cells",
                    f"- Standard render: `{fresh_216k_render.get('outputs', {}).get('trimmed_png')}`",
                ]
            )
        md.append(
            "- Note: This is verification/provenance evidence only. It demonstrates local execution and rendering, but the topology setting belongs to the Phase 3 registry and should not be used as the main new tool-paper visual."
        )
    else:
        md.append("- Not available.")

    md.extend(["", "### Phase 3 Registry Multi-BC Production SIMP Densities"])
    if fresh_multi_bc:
        md.append(
            _md_table(
                fresh_multi_bc,
                [
                    "prob_key",
                    "bvp",
                    "n_elem",
                    "volfrac",
                    "best_iteration",
                    "best_compliance",
                    "final_grayness",
                    "render_mesh_cells",
                    "render_png",
                ],
            )
        )
        md.append(
            "- Note: These are verification/provenance controls. They prove local multi-BC execution and rendering, but they are not the new tool-paper figure suite."
        )
    else:
        md.append("- Not available.")
    md.extend(["", "## New Tool-Paper SolverV4 Topology Suite"])
    if new_topology_suite:
        md.append(
            _md_table(
                new_topology_suite,
                [
                    "case",
                    "n_elem",
                    "volfrac",
                    "iters",
                    "final_compliance",
                    "final_grayness",
                    "final_outer_iters",
                    "linear_solve_cap_count",
                    "warm_iter_mean_s_excl_first",
                    "max_gpu_mem_mb",
                    "fused_cuda",
                    "matfree_gmg",
                    "render_png",
                    "status",
                ],
            )
        )
        md.append(
            "- Note: These larger new topology candidates establish the main-story suite on the Phase 4-style SolverV4 fused CUDA + matrix-free GMG path. Displacement-colored renders and artifact tables are now available, but linear solver cap counts still block final performance claims for the affected rows."
        )
    else:
        md.append("- Not available.")
    md.extend(
        [
            "",
        "## Draft Gate",
        "- Allowed next: draft outline and methods/results skeleton using only the evidence above.",
        "- Not yet allowed: final performance-table claims, because setup timing needs a controlled cold/warm protocol.",
            "- Fresh small SIMP provenance is available, but its 3D render is provenance-only because the grid is too coarse for publication visual quality.",
            "- Phase 3 registry production SIMP densities are internal controls only, not main manuscript content.",
            "- Required before final figures: publication-grade colored/camera-polished renders from accepted new tool-paper cases.",
        ]
    )
    (out_dir / "CONSOLIDATED_EVIDENCE.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-matrix", default="experiments/tool_paper/sample_matrix.json")
    parser.add_argument("--out", default="rerun_outputs/tool_paper_evidence_bundle")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = json.loads((ROOT / args.sample_matrix).read_text(encoding="utf-8"))
    solver_rows = build_solver_table(
        matrix,
        DEFAULT_SOLVER_SOURCES,
        DEFAULT_REPEATABILITY_SOURCE,
    )
    figure_rows = build_figure_manifest(DEFAULT_FIGURES)
    end_to_end = _load_json_or_none(ROOT / DEFAULT_END_TO_END_SOURCE)
    gpu_simp_64k = _load_json_or_none(ROOT / DEFAULT_GPU_SIMP_64K_SOURCE)
    fresh_216k = _load_json_or_none(ROOT / DEFAULT_FRESH_216K_SOURCE)
    fresh_216k_render = _load_json_or_none(ROOT / DEFAULT_FRESH_216K_RENDER_SOURCE)
    fresh_multi_bc = _load_fresh_multi_bc()
    new_topology_suite = _load_new_topology_suite()

    solver_fields = [
        "section",
        "case",
        "role",
        "status",
        "n_elem",
        "nelx",
        "nely",
        "nelz",
        "n_levels",
        "outer",
        "iters",
        "converged",
        "rel_residual",
        "solve_s",
        "setup_s",
        "repeat_solve_s_mean",
        "repeat_solve_s_stdev",
        "repeat_all_converged",
        "source_csv",
        "note",
    ]
    figure_fields = [
        "artifact",
        "type",
        "status",
        "use",
        "exists",
        "size_bytes",
        "sha256",
    ]
    _write_csv(out_dir / "evidence_table.csv", solver_rows, solver_fields)
    _write_csv(out_dir / "figure_manifest.csv", figure_rows, figure_fields)
    (out_dir / "evidence_table.json").write_text(
        json.dumps(solver_rows, indent=2), encoding="utf-8"
    )
    (out_dir / "figure_manifest.json").write_text(
        json.dumps(figure_rows, indent=2), encoding="utf-8"
    )
    if end_to_end:
        (out_dir / "end_to_end_simp_summary.json").write_text(
            json.dumps(end_to_end, indent=2), encoding="utf-8"
        )
    if gpu_simp_64k:
        (out_dir / "gpu_simp_visual_64k_summary.json").write_text(
            json.dumps(gpu_simp_64k, indent=2), encoding="utf-8"
        )
    if fresh_216k:
        (out_dir / "fresh_216k_simp_summary.json").write_text(
            json.dumps(fresh_216k, indent=2), encoding="utf-8"
        )
    if fresh_216k_render:
        (out_dir / "fresh_216k_render_summary.json").write_text(
            json.dumps(fresh_216k_render, indent=2), encoding="utf-8"
        )
    if fresh_multi_bc:
        (out_dir / "fresh_multi_bc_simp_summary.json").write_text(
            json.dumps(fresh_multi_bc, indent=2), encoding="utf-8"
        )
    if new_topology_suite:
        (out_dir / "new_toolpaper_solverv4_topology_summary.json").write_text(
            json.dumps(new_topology_suite, indent=2), encoding="utf-8"
        )
    write_markdown(
        out_dir,
        solver_rows,
        figure_rows,
        end_to_end,
        gpu_simp_64k,
        fresh_216k,
        fresh_216k_render,
        fresh_multi_bc,
        new_topology_suite,
    )

    missing = [row["artifact"] for row in figure_rows if not row["exists"]]
    if missing:
        print("[warn] missing figure artifacts:")
        for item in missing:
            print(f"  - {item}")
    print(f"[done] wrote {out_dir.relative_to(ROOT)}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
