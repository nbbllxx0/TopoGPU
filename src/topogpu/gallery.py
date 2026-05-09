"""Ready-to-run TopoGPU case gallery."""

from __future__ import annotations

from gpu_fem.problem_spec import EdgeSupport, PointLoad, ProblemSpec

from .problem import TopologyProblem


def cantilever_3d(
    nel: tuple[int, int, int] = (24, 12, 6),
    volfrac: float = 0.30,
    filter_radius: float = 1.5,
    support: str = "xmin",
    load: str = "tip_patch_z",
) -> TopologyProblem:
    """Create a simple 3D cantilever problem.

    The ``support`` and ``load`` arguments are reserved for the public schema;
    this first release supports the standard left-face clamp and a right-tip
    downward load used by the legacy examples.
    """

    if support != "xmin":
        raise ValueError("Only support='xmin' is currently implemented.")
    if load not in {"tip_patch_z", "tip_point_y"}:
        raise ValueError("Supported loads are 'tip_patch_z' and 'tip_point_y'.")

    nelx, nely, nelz = nel
    lz = 0.5
    force = {"tip_patch_z": {"fz": -1.0}, "tip_point_y": {"fy": -1.0}}[load]
    spec = ProblemSpec(
        Lx=2.0,
        Ly=1.0,
        Lz=lz,
        nelx=nelx,
        nely=nely,
        nelz=nelz,
        volfrac=volfrac,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, **force)],
        rmin=filter_radius,
    )
    return TopologyProblem(
        name="cantilever_3d",
        spec=spec,
        filter_radius=filter_radius,
        role="example",
        metadata={"support": support, "load": load},
    )


def side_load_cantilever(
    nel: tuple[int, int, int] = (56, 36, 28),
    volfrac: float = 0.24,
    filter_radius: float = 2.3,
) -> TopologyProblem:
    return cantilever_3d(
        nel=nel,
        volfrac=volfrac,
        filter_radius=filter_radius,
        load="tip_point_y",
    )


def tool_case(name: str, dims: str | None = None) -> TopologyProblem:
    """Load one of the manuscript tool-paper cases.

    This bridges the public API to the already verified case builder. The
    builder remains in ``experiments.tool_paper`` for this transition release.
    """

    from experiments.tool_paper.fast_new_topology_probe import build_problem

    built = build_problem(name, dims)
    spec = built["spec"]
    cfg = built["cfg"]
    return TopologyProblem(
        name=name,
        spec=spec,
        filter_radius=float(cfg.get("rmin", spec.rmin or 1.5)),
        force_override=built.get("F"),
        role=str(cfg.get("role", "tool_paper_case")),
        metadata={"source": "experiments.tool_paper.fast_new_topology_probe"},
    )


class CaseGallery:
    """Namespace for gallery constructors."""

    cantilever_3d = staticmethod(cantilever_3d)
    side_load_cantilever = staticmethod(side_load_cantilever)
    tool_case = staticmethod(tool_case)

