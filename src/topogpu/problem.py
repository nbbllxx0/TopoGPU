"""Problem definitions for the public TopoGPU interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from gpu_fem.problem_spec import ProblemSpec


@dataclass(slots=True)
class TopologyProblem:
    """Structured-grid SIMP topology-optimization problem.

    Parameters are intentionally narrow in the first public release: TopoGPU
    exposes the existing structured hexahedral workflow before general CAD or
    unstructured-mesh support.
    """

    name: str
    spec: ProblemSpec
    filter_radius: float | None = None
    force_override: np.ndarray | None = None
    role: str = "user_case"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nel(self) -> tuple[int, int, int]:
        return (self.spec.nelx, self.spec.nely, self.spec.nelz)

    @property
    def volfrac(self) -> float:
        return float(self.spec.volfrac)

    @property
    def is_3d(self) -> bool:
        return self.spec.is_3d

    def validate(self) -> None:
        errors = self.spec.validate()
        if errors:
            joined = "; ".join(errors)
            raise ValueError(f"Invalid TopoGPU problem {self.name!r}: {joined}")

    def to_dict(self) -> dict[str, Any]:
        data = self.spec.to_dict()
        data.update(
            {
                "name": self.name,
                "filter_radius": self.filter_radius,
                "role": self.role,
                "metadata": dict(self.metadata),
                "has_force_override": self.force_override is not None,
            }
        )
        return data

