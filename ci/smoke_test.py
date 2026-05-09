"""
Smoke test -- 2-minute sanity check that the release stack is wired up.

Runs a single FP64 V-cycle preconditioned CG on a 64k-element cantilever
probe at uniform density rho = 0.5, p = 3, and verifies the relative residual
drops below 1e-10 in at most 300 iterations. If this passes, your CuPy +
CUDA + gpu_fem stack is functional and you can proceed to the full validation
battery (validate_phase1.py) and benchmark suite (run_experiments_e1_e10.py).

Usage:
    python ci/smoke_test.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

# Environment bootstrap -- keep CuPy's persistent JIT cache inside the release
# tree while leaving transient compiler files in the platform temp directory.
# Forcing NVRTC temporaries into the repository is brittle on Windows because
# antivirus/indexing hooks can briefly lock just-created compiler files.
ROOT = Path(__file__).resolve().parents[1]
cache_dir = ROOT / ".cupy_cache"
cache_dir.mkdir(exist_ok=True)
runtime_tmp = ROOT / ".runtime_tmp"
runtime_tmp.mkdir(exist_ok=True)
os.environ.setdefault("CUPY_CACHE_DIR", str(cache_dir))
os.environ.setdefault("CUPY_CACHE_IN_MEMORY", "1")
os.environ.setdefault("TEMP", str(runtime_tmp))
os.environ.setdefault("TMP", str(runtime_tmp))
os.environ.setdefault("TMPDIR", str(runtime_tmp))
tempfile.tempdir = str(runtime_tmp)


class _WorkspaceTemporaryDirectory:
    """CuPy/NVRTC temp dirs need normal workspace ACLs in this sandbox."""

    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False):
        base = Path(dir) if dir is not None else runtime_tmp
        name = f"{prefix or 'tmp'}{uuid.uuid4().hex}{suffix or ''}"
        self.name = str(base / name)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        Path(self.name).mkdir(parents=True, exist_ok=False)

    def __enter__(self):
        return self.name

    def __exit__(self, exc_type, exc, tb):
        self.cleanup()
        return False

    def cleanup(self):
        shutil.rmtree(self.name, ignore_errors=True)


tempfile.TemporaryDirectory = _WorkspaceTemporaryDirectory

for cuda_root in (
    os.environ.get("CUDA_PATH"),
    os.environ.get("CUDA_HOME"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
):
    if not cuda_root:
        continue
    cuda_bin = Path(cuda_root) / "bin"
    if cuda_bin.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(cuda_bin))

sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def main() -> int:
    t_start = time.perf_counter()

    try:
        import cupy as cp
    except ImportError as e:
        print("FAIL: CuPy is not installed. See environment.yml.", file=sys.stderr)
        print(f"  underlying error: {e}", file=sys.stderr)
        return 2

    try:
        dev = cp.cuda.runtime.getDeviceProperties(0)
        name = dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"]
        print(f"[smoke] GPU detected: {name}")
    except Exception as e:
        print(f"FAIL: CuPy cannot access GPU 0. {e}", file=sys.stderr)
        return 3

    from gpu_fem.bc_generator import generate_bc
    from gpu_fem.presets import get_preset
    from gpu_fem.pub_simp_solver import KE_UNIT_3D, _edof_table_3d
    from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG
    from gpu_fem.solver_v2 import MatrixFreeKff, _cupy_pcg

    print("[smoke] building 64k cantilever probe ...")
    spec = get_preset("cantilever_gpu_medium")
    bc = generate_bc(spec)
    free = bc.free_dofs.astype(np.int32)
    edof_gpu = cp.asarray(
        _edof_table_3d(spec.nelx, spec.nely, spec.nelz).astype(np.int32)
    )
    F_free_gpu = cp.asarray(bc.F[free].astype(np.float64))
    free_gpu = cp.asarray(free)

    n_elem = spec.nelx * spec.nely * spec.nelz
    rho = cp.full(n_elem, 0.5, dtype=cp.float64)
    E_e = 1e-9 + (1.0 - 1e-9) * rho ** 3.0

    mf_op = MatrixFreeKff(
        edof_gpu=edof_gpu,
        KE_unit_gpu=cp.asarray(KE_UNIT_3D),
        free_gpu=free_gpu,
        n_free=len(free),
        ndof=bc.ndof,
    )
    gmg = GalerkinMatFreeGMG(
        mf_op=mf_op,
        free=free,
        free_gpu=free_gpu,
        nelx=spec.nelx,
        nely=spec.nely,
        nelz=spec.nelz,
        KE_UNIT=KE_UNIT_3D,
        n_levels=4,
        fine_smoother="fp64",
        smoother_type="chebyshev",
    )
    gmg.setup(E_e)

    print("[smoke] running FP64 V-cycle PCG ...")

    def A_op(v):
        return mf_op.matvec(v, E_e)

    x, iters, conv = _cupy_pcg(A_op, F_free_gpu, gmg.apply, tol=1e-10, maxiter=300)
    cp.cuda.Stream.null.synchronize()
    rel_res = float(cp.linalg.norm(F_free_gpu - A_op(x))) / float(
        cp.linalg.norm(F_free_gpu)
    )
    wall = time.perf_counter() - t_start

    print(f"[smoke] iters      = {iters}")
    print(f"[smoke] converged  = {conv}")
    print(f"[smoke] rel_resid  = {rel_res:.2e}")
    print(f"[smoke] wall time  = {wall:.1f} s")

    passed = bool(conv) and rel_res < 1e-10 and iters <= 300
    if passed:
        print("\nPASS --- stack is healthy. Proceed to validate_phase1.py.")
        return 0
    print("\nFAIL --- V-cycle PCG did not meet the 1e-10 gate.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
