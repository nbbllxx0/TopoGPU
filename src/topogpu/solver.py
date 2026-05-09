"""Public solver facade for TopoGPU."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import EvidenceBundle
from .problem import TopologyProblem


def __getattr__(name: str):
    if name == "MatrixFreeOperator":
        from gpu_fem.solver_v2 import MatrixFreeKff

        return MatrixFreeKff
    if name == "GMGPreconditioner":
        from gpu_fem.multigrid_v4 import GalerkinMatFreeGMG

        return GalerkinMatFreeGMG
    raise AttributeError(name)


@dataclass(slots=True)
class SIMPModel:
    penal: float = 3.0
    rho_min: float = 1.0e-3


@dataclass(slots=True)
class OCOptimizer:
    move: float = 0.10


@dataclass(slots=True)
class OptimizationResult:
    problem: TopologyProblem
    rho_final: np.ndarray
    history: list[dict[str, Any]]
    summary: dict[str, Any]

    def save(self, path: str | Path) -> EvidenceBundle:
        bundle = EvidenceBundle(Path(path))
        bundle.write_history(self.history)
        bundle.write_density(self.rho_final)
        bundle.write_summary(self.summary)
        bundle.write_manifest()
        return bundle

    def plot_history(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.5))
        if self.history and "compliance" in self.history[0]:
            ax.plot([r["iteration"] for r in self.history], [r["compliance"] for r in self.history])
            ax.set_ylabel("Compliance")
        ax.set_xlabel("Iteration")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def plot_topology(self):
        import matplotlib.pyplot as plt

        nelx, nely, nelz = self.problem.nel
        volume = self.rho_final.reshape(nelx, nely, nelz)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, image, title in zip(
            axes,
            [volume.max(axis=2).T, volume.max(axis=1).T, volume.max(axis=0).T],
            ["xy", "xz", "yz"],
        ):
            ax.imshow(image, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig


def _oc_update(rho: np.ndarray, dc: np.ndarray, volfrac: float, move: float, rho_min: float) -> np.ndarray:
    dc_safe = np.minimum(dc, -1.0e-12)
    lam_lo, lam_hi = 0.0, 1.0

    def candidate(lam: float) -> np.ndarray:
        return np.clip(
            rho * np.sqrt(np.maximum(-dc_safe / max(lam, 1.0e-40), 0.0)),
            np.maximum(rho - move, rho_min),
            np.minimum(rho + move, 1.0),
        )

    while candidate(lam_hi).mean() > volfrac and lam_hi < 1.0e40:
        lam_lo = lam_hi
        lam_hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lam_lo + lam_hi)
        rho_new = candidate(mid)
        if rho_new.mean() > volfrac:
            lam_lo = mid
        else:
            lam_hi = mid
        if abs(float(rho_new.mean()) - volfrac) < 1.0e-8:
            break
    return rho_new


def _grayness(rho: np.ndarray) -> float:
    return float(4.0 * np.mean(rho * (1.0 - rho)))


class SIMPSolver:
    """Run a bounded SIMP loop through the existing CPU or SolverV4 backend."""

    def __init__(
        self,
        backend: str = "cpu",
        linear_solver: str = "auto",
        optimizer: str = "oc",
        tol: float = 1.0e-5,
        max_krylov: int = 800,
        max_iter: int = 12,
        move: float = 0.10,
        rho_min: float = 1.0e-3,
    ) -> None:
        if optimizer != "oc":
            raise ValueError("Only optimizer='oc' is implemented in v0.1.0.")
        self.backend = backend
        self.linear_solver = linear_solver
        self.tol = tol
        self.max_krylov = max_krylov
        self.max_iter = max_iter
        self.move = move
        self.rho_min = rho_min

    def solve(self, problem: TopologyProblem) -> OptimizationResult:
        problem.validate()
        if self.backend in {"cpu", "scipy"}:
            return self._solve_cpu(problem)
        if self.backend in {"cuda", "cupy"}:
            return self._solve_solverv4(problem)
        raise ValueError(f"Unknown backend {self.backend!r}.")

    def _solve_cpu(self, problem: TopologyProblem) -> OptimizationResult:
        from gpu_fem.pub_simp_solver import SIMPParams, run_simp

        spec = problem.spec
        params = SIMPParams(
            nelx=spec.nelx,
            nely=spec.nely,
            nelz=spec.nelz,
            volfrac=spec.volfrac,
            rmin=problem.filter_radius or spec.rmin or 1.5,
            max_iter=self.max_iter,
            min_iter=min(2, self.max_iter),
            move=self.move,
            tol=1.0e-3,
        )
        result = run_simp(params, verbose=False)
        history = [
            {"iteration": i + 1, "compliance": float(c)}
            for i, c in enumerate(result["compliance_history"])
        ]
        summary = {
            "problem": problem.to_dict(),
            "backend": "cpu",
            "linear_solver": "scipy_or_pyamg",
            "n_iter": int(result["n_iter"]),
            "final_compliance": float(result["final_compliance"]),
            "final_grayness": float(result["final_grayness"]),
        }
        return OptimizationResult(problem, result["rho_final"], history, summary)

    def _solve_solverv4(self, problem: TopologyProblem) -> OptimizationResult:
        from gpu_fem.bc_generator import generate_bc
        from gpu_fem.pub_simp_solver import KE_UNIT_3D, _build_sparse_indices, _edof_table_3d
        from gpu_fem.solver_v4 import SolverV4

        spec = problem.spec
        if not problem.is_3d:
            raise ValueError("SolverV4 backend currently requires a 3D problem.")
        bc = generate_bc(spec)
        force = problem.force_override if problem.force_override is not None else bc.F
        dims = problem.nel
        edof = _edof_table_3d(*dims)
        row_idx, col_idx = _build_sparse_indices(edof)
        solver = SolverV4(
            edof=edof,
            row_idx=row_idx,
            col_idx=col_idx,
            KE_UNIT=KE_UNIT_3D,
            free=bc.free_dofs.astype(np.int32),
            F=force,
            ndof=bc.ndof,
            backend="cupy",
            enable_warm_start=True,
            enable_matrix_free=True,
            enable_fused_cuda=True,
            enable_matfree_gmg=True,
            matfree_gmg_levels=4,
            gmg_fine_smoother="fp32",
            gmg_smoother_type="chebyshev",
            grid_dims=dims,
            cg_tol=self.tol,
            cg_maxiter=self.max_krylov,
            enable_profiling=True,
        )
        rho = np.full(edof.shape[0], spec.volfrac, dtype=np.float64)
        history: list[dict[str, Any]] = []
        t_total = time.perf_counter()
        for iteration in range(1, self.max_iter + 1):
            t0 = time.perf_counter()
            compliance, dc = solver.solve(rho, penal=3.0)
            wall_s = time.perf_counter() - t0
            rho = _oc_update(rho, dc, spec.volfrac, self.move, self.rho_min)
            history.append(
                {
                    "iteration": iteration,
                    "compliance": float(compliance),
                    "outer_iters": int(getattr(solver, "last_cg_iters", -1)),
                    "rho_mean": float(rho.mean()),
                    "rho_min": float(rho.min()),
                    "rho_max": float(rho.max()),
                    "grayness": _grayness(rho),
                    "wall_s": wall_s,
                    "outer_solver": getattr(solver, "last_outer_solver", ""),
                }
            )
        summary = {
            "problem": problem.to_dict(),
            "backend": "SolverV4",
            "linear_solver": "pcg_gmg" if self.linear_solver == "auto" else self.linear_solver,
            "n_iter": self.max_iter,
            "total_wall_s": time.perf_counter() - t_total,
            "final": history[-1] if history else {},
        }
        return OptimizationResult(problem, rho, history, summary)


def solve(problem: TopologyProblem, **kwargs: Any) -> OptimizationResult:
    return SIMPSolver(**kwargs).solve(problem)
