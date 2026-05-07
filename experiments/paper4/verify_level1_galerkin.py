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
import scipy.sparse as sp
import csv
import json

from gpu_fem.multigrid_v4 import _build_level1_galerkin_struct, assemble_level1_galerkin
from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d
from gpu_fem.solver_v2 import _build_scalar_prolongation, _coarse_free_dofs_injection


def assemble_fine_K(nelx: int, nely: int, nelz: int, free: np.ndarray, rho: np.ndarray) -> sp.csr_matrix:
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    n_free = len(free)
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    free_mask = np.zeros(ndof, dtype=bool)
    free_mask[free] = True
    keep = free_mask[row_idx] & free_mask[col_idx]
    keep_idx = np.nonzero(keep)[0]
    free_local = np.full(ndof, -1, dtype=np.int32)
    free_local[free] = np.arange(n_free, dtype=np.int32)
    rows_local = free_local[row_idx[keep_idx]]
    cols_local = free_local[col_idx[keep_idx]]
    elem_of = (keep_idx // 576).astype(np.int32)
    KE_of = KE_UNIT_3D.ravel()[keep_idx % 576]
    E_e = 1e-9 + (1.0 - 1e-9) * rho[elem_of] ** 3.0
    K = sp.csr_matrix((E_e * KE_of, (rows_local, cols_local)), shape=(n_free, n_free))
    K.sum_duplicates()
    return K


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

    nelx_c, nely_c, nelz_c = nelx // 2, nely // 2, nelz // 2
    free_c = _coarse_free_dofs_injection(nelx, nely, nelz, nelx_c, nely_c, nelz_c, free)
    P_sc = _build_scalar_prolongation(nelx, nely, nelz, nelx_c, nely_c, nelz_c)
    P_vec = sp.kron(P_sc, sp.eye(3, format="csr", dtype=np.float64), format="csr")
    P_free = P_vec[free, :][:, free_c].tocsr()

    level1 = _build_level1_galerkin_struct(
        nelx_c,
        nely_c,
        nelz_c,
        nelx,
        nely,
        nelz,
        free_c,
        KE_UNIT_3D,
    )

    cases = {
        "uniform-0.5": np.full(nelx * nely * nelz, 0.5),
        "uniform-1.0": np.full(nelx * nely * nelz, 1.0),
        "random": np.random.default_rng(42).random(nelx * nely * nelz),
    }

    print("=" * 72)
    print("Level-1 Galerkin exactness check")
    print("=" * 72)
    rows = []
    for label, rho in cases.items():
        K_f = assemble_fine_K(nelx, nely, nelz, free, rho)
        K_c_ref = (P_free.T @ K_f @ P_free).toarray()
        E_e = 1e-9 + (1.0 - 1e-9) * rho ** 3.0
        K_c_gpu, _ = assemble_level1_galerkin(level1, cp.asarray(E_e))
        K_c = cp.asnumpy(K_c_gpu.toarray())
        rel_err = np.linalg.norm(K_c - K_c_ref, "fro") / max(np.linalg.norm(K_c_ref, "fro"), 1e-300)
        row = {
            "case": label,
            "nelx": nelx,
            "nely": nely,
            "nelz": nelz,
            "n_elem": nelx * nely * nelz,
            "n_free": len(free),
            "n_free_c": len(free_c),
            "rel_err_fro": float(rel_err),
        }
        rows.append(row)
        print(f"{label:12s}  rel_err={rel_err:.3e}")

    out_csv = out_dir / "verify_level1_galerkin.csv"
    out_json = out_dir / "verify_level1_galerkin.json"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out_csv.relative_to(ROOT)}")
    print(f"Saved {out_json.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
