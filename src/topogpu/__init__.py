"""Public TopoGPU package facade."""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "TopologyProblem",
    "CaseGallery",
    "DensityFilter",
    "SIMPModel",
    "OCOptimizer",
    "MatrixFreeOperator",
    "GMGPreconditioner",
    "SIMPSolver",
    "OptimizationResult",
    "EvidenceBundle",
    "cantilever_3d",
    "side_load_cantilever",
    "tool_case",
    "solve",
]

_EXPORTS = {
    "EvidenceBundle": ("topogpu.evidence", "EvidenceBundle"),
    "DensityFilter": ("topogpu.filter", "DensityFilter"),
    "CaseGallery": ("topogpu.gallery", "CaseGallery"),
    "cantilever_3d": ("topogpu.gallery", "cantilever_3d"),
    "side_load_cantilever": ("topogpu.gallery", "side_load_cantilever"),
    "tool_case": ("topogpu.gallery", "tool_case"),
    "TopologyProblem": ("topogpu.problem", "TopologyProblem"),
    "GMGPreconditioner": ("topogpu.solver", "GMGPreconditioner"),
    "MatrixFreeOperator": ("topogpu.solver", "MatrixFreeOperator"),
    "OCOptimizer": ("topogpu.solver", "OCOptimizer"),
    "OptimizationResult": ("topogpu.solver", "OptimizationResult"),
    "SIMPModel": ("topogpu.solver", "SIMPModel"),
    "SIMPSolver": ("topogpu.solver", "SIMPSolver"),
    "solve": ("topogpu.solver", "solve"),
}


def __getattr__(name: str):
    if name == "gallery":
        return importlib.import_module("topogpu.gallery")
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(name)
