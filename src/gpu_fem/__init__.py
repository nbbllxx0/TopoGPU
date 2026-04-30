"""
gpu_fem --- minimal paper-4 release subset.

This release ships only the modules required to reproduce Phase 1 (M1--M8
validation) and Phase 2 (E1--E10 experiments) of:

    Yang, Wang, Wang (2026). "Mixed-Precision Matrix-Free Geometric Multigrid
    for 3D SIMP Topology Optimization on a Single GPU."

The surrogate-routing and adaptive-SIMP modules that live alongside this
package in the full research workspace (surrogate_gpu, simp_gpu, workflow,
auto_simp, local_agents, agents, pub_baseline_controller) are NOT included
here because they are not exercised by any paper-4 experiment.
"""

from .solver_v2 import SolverV2, HexGridGMG, MatrixFreeKff
from .solver_v4 import SolverV4
from .multigrid_v4 import GalerkinMatFreeGMG
from .fem_gpu import GPUFEMSolver, detect_gpu_backend

__all__ = [
    "SolverV2",
    "SolverV4",
    "HexGridGMG",
    "MatrixFreeKff",
    "GalerkinMatFreeGMG",
    "GPUFEMSolver",
    "detect_gpu_backend",
]
