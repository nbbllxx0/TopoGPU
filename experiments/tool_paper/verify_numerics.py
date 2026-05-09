from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.tool_paper.fast_new_topology_probe import build_problem  # noqa: E402
from gpu_fem.pub_simp_solver import (  # noqa: E402
    KE_UNIT_3D,
    _build_density_filter,
    _build_sparse_indices,
    _edof_table_3d,
)
from gpu_fem.solver_v2 import MatrixFreeKff  # noqa: E402


DEFAULT_CASE_DIMS = {
    "tool_long_cantilever_vf16": "6x4x4",
    "tool_portal_bridge_vf18": "8x4x4",
    "tool_asymmetric_bracket_vf14": "6x4x4",
}

E0 = 1.0
EMIN = 1.0e-9
PENAL = 3.0
OPERATOR_REL_TOL = 1.0e-10
SENSITIVITY_REL_TOL = 5.0e-4
FILTER_REL_TOL = 1.0e-12


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _parse_dims(dims: str) -> tuple[int, int, int]:
    values = tuple(int(x) for x in dims.lower().split("x"))
    if len(values) != 3:
        raise ValueError(f"dims must have form nelx x nely x nelz; got {dims!r}")
    return values


def _density(n_elem: int, volfrac: float) -> np.ndarray:
    idx = np.arange(n_elem, dtype=np.float64)
    pattern = 0.5 + 0.5 * np.sin(0.73 * idx + 0.19 * np.cos(0.11 * idx))
    rho = volfrac + 0.12 * (pattern - pattern.mean())
    return np.clip(rho, 0.08, 0.92).astype(np.float64)


def _assemble_full_k(
    edof: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    ndof: int,
    rho: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    e_elem = EMIN + (E0 - EMIN) * rho**PENAL
    values = (e_elem[:, None] * KE_UNIT_3D.ravel()[None, :]).ravel()
    k_full = sp.csc_matrix((values, (row_idx, col_idx)), shape=(ndof, ndof))
    k_full.sum_duplicates()
    return k_full, e_elem


def _compliance_and_sensitivity(
    edof: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    free: np.ndarray,
    force: np.ndarray,
    ndof: int,
    rho: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    k_full, _ = _assemble_full_k(edof, row_idx, col_idx, ndof, rho)
    u = np.zeros(ndof, dtype=np.float64)
    u[free] = spla.spsolve(k_full[free][:, free], force[free])
    compliance = float(force @ u)
    u_elem = u[edof]
    ku_elem = u_elem @ KE_UNIT_3D
    ce = np.einsum("ei,ei->e", ku_elem, u_elem)
    dc = -PENAL * (E0 - EMIN) * rho ** (PENAL - 1.0) * ce
    return compliance, dc, u, ce


def _operator_rows(case_name: str, dims: str, seed: int) -> list[dict]:
    import cupy as cp

    built = build_problem(case_name, dims)
    spec = built["spec"]
    bc = built["bc"]
    nelx, nely, nelz = _parse_dims(dims)
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    free = bc.free_dofs.astype(np.int32)
    fixed = bc.fixed_dofs.astype(np.int32)
    rho = _density(edof.shape[0], float(spec.volfrac))
    k_full, e_elem = _assemble_full_k(edof, row_idx, col_idx, bc.ndof, rho)
    kff = k_full[free][:, free].tocsr()

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(len(free))
    y = rng.standard_normal(len(free))
    mf_op = MatrixFreeKff(
        edof_gpu=cp.asarray(edof, dtype=cp.int32),
        KE_unit_gpu=cp.asarray(KE_UNIT_3D, dtype=cp.float64),
        free_gpu=cp.asarray(free, dtype=cp.int32),
        n_free=len(free),
        ndof=bc.ndof,
    )
    e_gpu = cp.asarray(e_elem, dtype=cp.float64)
    mf_x = cp.asnumpy(mf_op.matvec(cp.asarray(x, dtype=cp.float64), e_gpu))
    mf_y = cp.asnumpy(mf_op.matvec(cp.asarray(y, dtype=cp.float64), e_gpu))
    ref_x = kff @ x
    ref_y = kff @ y
    diag_ref = kff.diagonal()
    diag_mf = cp.asnumpy(mf_op.extract_diagonal(e_gpu))

    rel_l2 = np.linalg.norm(ref_x - mf_x) / max(np.linalg.norm(ref_x), 1.0e-300)
    rel_inf = np.linalg.norm(ref_x - mf_x, ord=np.inf) / max(np.linalg.norm(ref_x, ord=np.inf), 1.0e-300)
    energy_ref = float(x @ ref_x)
    energy_mf = float(x @ mf_x)
    energy_rel = abs(energy_ref - energy_mf) / max(abs(energy_ref), 1.0e-300)
    sym_ref = abs(float(x @ ref_y) - float(y @ ref_x)) / max(abs(float(x @ ref_y)), 1.0e-300)
    sym_mf = abs(float(x @ mf_y) - float(y @ mf_x)) / max(abs(float(x @ mf_y)), 1.0e-300)
    diag_rel = np.linalg.norm(diag_ref - diag_mf) / max(np.linalg.norm(diag_ref), 1.0e-300)
    fixed_overlap = int(np.intersect1d(free, fixed).size)
    pass_flag = (
        rel_l2 <= OPERATOR_REL_TOL
        and rel_inf <= OPERATOR_REL_TOL
        and energy_rel <= OPERATOR_REL_TOL
        and diag_rel <= OPERATOR_REL_TOL
        and fixed_overlap == 0
    )
    cp.get_default_memory_pool().free_all_blocks()

    return [
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "matvec_rel_l2",
            "value": rel_l2,
            "threshold": OPERATOR_REL_TOL,
            "pass": rel_l2 <= OPERATOR_REL_TOL,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "matvec_rel_inf",
            "value": rel_inf,
            "threshold": OPERATOR_REL_TOL,
            "pass": rel_inf <= OPERATOR_REL_TOL,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "energy_rel",
            "value": energy_rel,
            "threshold": OPERATOR_REL_TOL,
            "pass": energy_rel <= OPERATOR_REL_TOL,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "symmetry_ref_rel",
            "value": sym_ref,
            "threshold": OPERATOR_REL_TOL,
            "pass": sym_ref <= OPERATOR_REL_TOL,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "symmetry_matfree_rel",
            "value": sym_mf,
            "threshold": OPERATOR_REL_TOL,
            "pass": sym_mf <= OPERATOR_REL_TOL,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "diagonal_rel_l2",
            "value": diag_rel,
            "threshold": OPERATOR_REL_TOL,
            "pass": diag_rel <= OPERATOR_REL_TOL,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "fixed_free_overlap",
            "value": fixed_overlap,
            "threshold": 0,
            "pass": fixed_overlap == 0,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": edof.shape[0],
            "ndof": bc.ndof,
            "n_free": len(free),
            "n_fixed": len(fixed),
            "metric": "operator_case_pass",
            "value": int(pass_flag),
            "threshold": 1,
            "pass": pass_flag,
        },
    ]


def _sensitivity_rows(case_name: str, dims: str, seed: int) -> list[dict]:
    built = build_problem(case_name, dims)
    spec = built["spec"]
    bc = built["bc"]
    nelx, nely, nelz = _parse_dims(dims)
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    free = bc.free_dofs.astype(np.int32)
    rho = _density(edof.shape[0], float(spec.volfrac))
    compliance, dc, _, _ = _compliance_and_sensitivity(edof, row_idx, col_idx, free, built["F"], bc.ndof, rho)

    rng = np.random.default_rng(seed)
    strong = list(np.argsort(np.abs(dc))[-3:])
    sampled = list(rng.choice(edof.shape[0], size=min(2, edof.shape[0]), replace=False))
    elements = list(dict.fromkeys(strong + sampled))
    rows = []
    for elem in elements:
        step = min(1.0e-6, 0.25 * min(rho[elem] - 0.02, 0.98 - rho[elem]))
        if step <= 0.0:
            step = 1.0e-7
        rho_plus = rho.copy()
        rho_minus = rho.copy()
        rho_plus[elem] += step
        rho_minus[elem] -= step
        comp_plus, _, _, _ = _compliance_and_sensitivity(
            edof, row_idx, col_idx, free, built["F"], bc.ndof, rho_plus
        )
        comp_minus, _, _, _ = _compliance_and_sensitivity(
            edof, row_idx, col_idx, free, built["F"], bc.ndof, rho_minus
        )
        fd = (comp_plus - comp_minus) / (2.0 * step)
        abs_err = abs(fd - dc[elem])
        rel_err = abs_err / max(abs(dc[elem]), abs(fd), 1.0e-300)
        rows.append(
            {
                "case": case_name,
                "dims": dims,
                "n_elem": edof.shape[0],
                "element": int(elem),
                "rho": float(rho[elem]),
                "compliance": compliance,
                "analytic_dc": float(dc[elem]),
                "finite_difference_dc": float(fd),
                "abs_error": abs_err,
                "rel_error": rel_err,
                "threshold": SENSITIVITY_REL_TOL,
                "pass": rel_err <= SENSITIVITY_REL_TOL,
            }
        )
    return rows


def _filter_rows(case_name: str, dims: str) -> list[dict]:
    built = build_problem(case_name, dims)
    spec = built["spec"]
    nelx, nely, nelz = _parse_dims(dims)
    h = _build_density_filter(nelx, nely, float(spec.rmin), nelz)
    ones = np.ones(h.shape[1])
    row_sums = np.asarray(h.sum(axis=1)).ravel()
    uniform_error = np.linalg.norm(h @ ones - ones, ord=np.inf)
    row_sum_error = np.linalg.norm(row_sums - ones, ord=np.inf)
    return [
        {
            "case": case_name,
            "dims": dims,
            "n_elem": h.shape[0],
            "rmin": float(spec.rmin),
            "metric": "row_sum_inf_error",
            "value": row_sum_error,
            "threshold": FILTER_REL_TOL,
            "pass": row_sum_error <= FILTER_REL_TOL,
        },
        {
            "case": case_name,
            "dims": dims,
            "n_elem": h.shape[0],
            "rmin": float(spec.rmin),
            "metric": "uniform_density_inf_error",
            "value": uniform_error,
            "threshold": FILTER_REL_TOL,
            "pass": uniform_error <= FILTER_REL_TOL,
        },
    ]


def run(out_dir: Path, case_dims: dict[str, str]) -> dict:
    operator_rows = []
    sensitivity_rows = []
    filter_rows = []
    for idx, (case_name, dims) in enumerate(case_dims.items()):
        operator_rows.extend(_operator_rows(case_name, dims, seed=1000 + idx))
        sensitivity_rows.extend(_sensitivity_rows(case_name, dims, seed=2000 + idx))
        filter_rows.extend(_filter_rows(case_name, dims))

    _write_csv(
        out_dir / "TABLE_OPERATOR_VERIFICATION.csv",
        operator_rows,
        [
            "case",
            "dims",
            "n_elem",
            "ndof",
            "n_free",
            "n_fixed",
            "metric",
            "value",
            "threshold",
            "pass",
        ],
    )
    _write_csv(
        out_dir / "TABLE_SENSITIVITY_VERIFICATION.csv",
        sensitivity_rows,
        [
            "case",
            "dims",
            "n_elem",
            "element",
            "rho",
            "compliance",
            "analytic_dc",
            "finite_difference_dc",
            "abs_error",
            "rel_error",
            "threshold",
            "pass",
        ],
    )
    _write_csv(
        out_dir / "TABLE_FILTER_VERIFICATION.csv",
        filter_rows,
        ["case", "dims", "n_elem", "rmin", "metric", "value", "threshold", "pass"],
    )

    all_pass = all(bool(row["pass"]) for row in operator_rows + sensitivity_rows + filter_rows)
    summary = {
        "out_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
        "case_dims": case_dims,
        "operator_rows": len(operator_rows),
        "sensitivity_rows": len(sensitivity_rows),
        "filter_rows": len(filter_rows),
        "operator_max_value_by_metric": {},
        "sensitivity_max_rel_error": max(float(row["rel_error"]) for row in sensitivity_rows),
        "filter_max_value": max(float(row["value"]) for row in filter_rows),
        "all_pass": all_pass,
        "thresholds": {
            "operator_rel": OPERATOR_REL_TOL,
            "sensitivity_rel": SENSITIVITY_REL_TOL,
            "filter_rel": FILTER_REL_TOL,
        },
    }
    for row in operator_rows:
        metric = row["metric"]
        summary["operator_max_value_by_metric"][metric] = max(
            float(row["value"]),
            float(summary["operator_max_value_by_metric"].get(metric, 0.0)),
        )
    (out_dir / "verification_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _case_dims_from_args(items: list[str]) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_CASE_DIMS)
    case_dims: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--case-dim entries must have form case=dims")
        case, dims = item.split("=", 1)
        _parse_dims(dims)
        case_dims[case] = dims
    return case_dims


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate numerical verification evidence for the tool paper.")
    parser.add_argument("--out", default="rerun_outputs/tool_paper_verification")
    parser.add_argument(
        "--case-dim",
        action="append",
        default=[],
        help="Case and small verification mesh, e.g. tool_long_cantilever_vf16=6x4x4.",
    )
    args = parser.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run(out_dir, _case_dims_from_args(args.case_dim))
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
