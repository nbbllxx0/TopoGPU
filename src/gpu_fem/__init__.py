"""gpu_fem release subset.

Heavy solver modules import SciPy and CuPy. Keep them lazy so lightweight
metadata, case-schema, and TopoGPU package imports still work in environments
that have not installed the full GPU stack yet.
"""

from __future__ import annotations


__all__ = [
    "SolverV2",
    "SolverV4",
    "HexGridGMG",
    "MatrixFreeKff",
    "GalerkinMatFreeGMG",
    "GPUFEMSolver",
    "detect_gpu_backend",
]


def __getattr__(name: str):
    if name in {"SolverV2", "HexGridGMG", "MatrixFreeKff"}:
        from .solver_v2 import HexGridGMG, MatrixFreeKff, SolverV2

        return {"SolverV2": SolverV2, "HexGridGMG": HexGridGMG, "MatrixFreeKff": MatrixFreeKff}[name]
    if name == "SolverV4":
        from .solver_v4 import SolverV4

        return SolverV4
    if name == "GalerkinMatFreeGMG":
        from .multigrid_v4 import GalerkinMatFreeGMG

        return GalerkinMatFreeGMG
    if name in {"GPUFEMSolver", "detect_gpu_backend"}:
        from .fem_gpu import GPUFEMSolver, detect_gpu_backend

        return {"GPUFEMSolver": GPUFEMSolver, "detect_gpu_backend": detect_gpu_backend}[name]
    raise AttributeError(name)
