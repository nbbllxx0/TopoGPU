from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _prefer_pytorch_env() -> None:
    root = Path(__file__).resolve().parents[2]
    runtime_tmp = root / ".runtime_tmp"
    cupy_cache = root / ".cupy_cache"
    runtime_tmp.mkdir(exist_ok=True)
    cupy_cache.mkdir(exist_ok=True)
    os.environ["TMP"] = str(runtime_tmp)
    os.environ["TEMP"] = str(runtime_tmp)
    os.environ["CUPY_CACHE_DIR"] = str(cupy_cache)
    os.environ["CUPY_TEMPDIR"] = str(runtime_tmp)
    tempfile.tempdir = str(runtime_tmp)



_prefer_pytorch_env()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import json

from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d
from gpu_fem.solver_v2 import MatrixFreeKff


def main() -> None:
    import cupy as cp

    out_dir = ROOT / "experiments" / "paper4"
    nelx, nely, nelz = 12, 6, 6
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    fixed = []
    for iy in range(nely + 1):
        for iz in range(nelz + 1):
            nid = iy * (nelz + 1) + iz
            fixed.extend([3 * nid, 3 * nid + 1, 3 * nid + 2])
    fixed = np.unique(np.asarray(fixed, dtype=np.int32))
    free = np.setdiff1d(np.arange(ndof, dtype=np.int32), fixed)

    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    mf_op = MatrixFreeKff(
        edof_gpu=cp.asarray(edof),
        KE_unit_gpu=cp.asarray(KE_UNIT_3D),
        free_gpu=cp.asarray(free),
        n_free=len(free),
        ndof=ndof,
    )
    rho = np.full(nelx * nely * nelz, 0.5)
    E_e = cp.asarray(1e-9 + (1.0 - 1e-9) * rho ** 3.0)
    fine_diag = mf_op.extract_diagonal(E_e)

    gmg = GalerkinMatFreeGMG(
        mf_op=mf_op,
        free=free,
        free_gpu=cp.asarray(free),
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        KE_UNIT=KE_UNIT_3D,
        n_levels=3,
        fine_smoother="fp32",
        fine_smoother_degree=2,
    )
    gmg.setup(E_e, fine_diag=fine_diag)

    n = len(free)
    K_dense = np.zeros((n, n), dtype=np.float64)
    MK_dense = np.zeros((n, n), dtype=np.float64)
    I = cp.eye(n, dtype=cp.float64)
    for j in range(n):
        e_j = I[:, j]
        K_dense[:, j] = cp.asnumpy(gmg.apply_fine_operator(e_j))
        MK_dense[:, j] = cp.asnumpy(gmg.apply_preconditioned_operator(e_j))

    ev_K = np.linalg.eigvalsh(K_dense)
    raw_kappa = ev_K.max() / ev_K.min()

    sym_err = np.linalg.norm(MK_dense - MK_dense.T, "fro") / max(np.linalg.norm(MK_dense, "fro"), 1e-300)
    ev_MK = np.linalg.eigvals(MK_dense)
    real_ev = np.real(ev_MK[np.abs(np.imag(ev_MK)) < 1e-8])
    real_ev = real_ev[real_ev > 1e-12]
    eff_kappa = float(real_ev.max() / real_ev.min()) if real_ev.size else float("nan")

    eps_bf16 = 2.0 ** -8
    row = {
        "nelx": nelx,
        "nely": nely,
        "nelz": nelz,
        "n_elem": nelx * nely * nelz,
        "n_free": n,
        "lambda_max": float(ev_K.max()),
        "lambda_min": float(ev_K.min()),
        "kappa_raw": float(raw_kappa),
        "eps_bf16_kappa_raw": float(eps_bf16 * raw_kappa),
        "symmetry_error_precond": float(sym_err),
        "kappa_eff": float(eff_kappa),
        "eps_bf16_kappa_eff": float(eps_bf16 * eff_kappa),
        "probe_finite": bool(gmg.probe_quality(cp.ones(n, dtype=cp.float64))["finite"]),
        "probe_pd": bool(gmg.probe_quality(cp.ones(n, dtype=cp.float64))["pd"]),
        "probe_z_over_jacobi": float(gmg.probe_quality(cp.ones(n, dtype=cp.float64))["z_over_jacobi"]),
    }

    print("=" * 72)
    print("Paper-4 preconditioned spectral probe")
    print("=" * 72)
    print(f"n_free              : {n}")
    print(f"lambda_max(K)       : {ev_K.max():.6e}")
    print(f"lambda_min(K)       : {ev_K.min():.6e}")
    print(f"kappa(K)            : {raw_kappa:.6e}")
    print(f"eps_bf16 * kappa(K) : {eps_bf16 * raw_kappa:.6e}")
    print(f"symmetry error(MA)  : {sym_err:.6e}")
    print(f"kappa(MA)           : {eff_kappa:.6e}")
    print(f"eps_bf16 * kappa(MA): {eps_bf16 * eff_kappa:.6e}")
    print(f"quality probe       : {gmg.probe_quality(cp.ones(n, dtype=cp.float64))}")

    out_json = out_dir / "estimate_preconditioned_kappa.json"
    out_json.write_text(json.dumps([row], indent=2))
    out_csv = out_dir / "estimate_preconditioned_kappa.csv"
    header = list(row.keys())
    values = [row[k] for k in header]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        f.write(",".join(str(v) for v in values) + "\n")
    print(f"\nSaved {out_csv.relative_to(ROOT)}")
    print(f"Saved {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
