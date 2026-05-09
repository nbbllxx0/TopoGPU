---
title: "TopoGPU: GPU-Accelerated 3D SIMP Topology Optimization in Python"
tags:
  - Python
  - CUDA
  - topology optimization
  - SIMP
  - finite elements
  - geometric multigrid
authors:
  - name: Shaoliang Yang
    affiliation: 1
  - name: Jun Wang
    affiliation: 1
  - name: Yunsheng Wang
    affiliation: 1
affiliations:
  - name: Santa Clara University
    index: 1
date: 9 May 2026
bibliography: paper.bib
---

# Summary

TopoGPU is an open-source Python/CUDA package for reproducible structured-grid
three-dimensional SIMP topology optimization. It provides case constructors,
YAML case suites, density filtering, bounded optimality-criteria updates,
matrix-free GPU state solves, verification commands, rendering metadata, and
manifest-tracked evidence bundles. The package targets researchers, graduate
students, and computational mechanics groups who need runnable GPU
topology-optimization examples with explicit boundary conditions, residual
histories, cap diagnostics, and archived artifacts.

The initial release exposes a public `topogpu` API over the existing
matrix-free `gpu_fem` implementation core. A typical use is:

```python
import topogpu as tg

problem = tg.gallery.cantilever_3d(nel=(24, 12, 6), volfrac=0.30)
result = tg.SIMPSolver(backend="cuda", linear_solver="pcg_gmg",
                       tol=1e-5, max_krylov=800).solve(problem)
result.save("runs/cantilever_3d")
```

Each saved result writes `history.csv`, `summary.json`, `rho_final.npy`,
`render_metadata.json`, and `ARTIFACT_MANIFEST.csv`, allowing figures and tables
to be traced back to run-level evidence.

# Statement of Need

Topology optimization has a strong tradition of compact and inspectable
research software. The 99-line MATLAB code [@sigmund2001], the 88-line code
[@andreassen2011], and Top3D [@liu2014] made density-based methods easier to
study, modify, and cite. Larger-scale three-dimensional topology optimization,
however, increasingly depends on matrix-free operators, Krylov solvers,
multigrid preconditioners, GPU kernels, and careful benchmark protocols
[@aage2015; @aage2017; @yang2026fused; @yang2026gmg]. These details are often
hard to reconstruct from a final rendered topology or a single timing table.

TopoGPU targets users who need a compact GPU baseline between educational
MATLAB/CPU examples and full HPC frameworks. Its public scope is deliberately
narrow: structured-grid 3D compliance minimization with SIMP densities, density
filtering, bounded OC updates, and SolverV4-backed linear state solves. Within
that scope, TopoGPU emphasizes installable code, example cases, numerical
verification, benchmark-role classification, and artifact manifests.

The package is justified as a separate research artifact because its contribution
is the workflow boundary: case definitions, package-level API, CLI commands,
verification tests, role-gated evidence, and manifest-tracked outputs around the
existing matrix-free solver backend.

# Functionality

TopoGPU currently provides:

- public case constructors under `topogpu.gallery`;
- `TopologyProblem` objects for mesh, volume, support/load, and metadata;
- `DensityFilter` utilities for small-case checks and package tests;
- `SIMPSolver` with CPU smoke-test and CUDA/SolverV4 execution paths;
- `OptimizationResult.save()` for evidence-bundle outputs;
- command-line entry points for citation metadata, case listing, lightweight
  runs, rendering metadata, benchmark-suite declarations, and numerical
  verification;
- YAML case-suite declarations for examples, production candidates, and stress
  diagnostics;
- documentation for installation, quick start, reproducibility, and limitations.

The current implementation distinguishes evidence roles:

| Role | Meaning |
| --- | --- |
| Verification | numerical consistency checks used before benchmark claims |
| Production timing | residual-clean, cap-free rows satisfying volume and manifest gates |
| Visual sample | topology figure evidence with disclosed solver status |
| Stress diagnostic | intentionally reported conditioning or cap-hit behavior |
| Invalid | missing, inconsistent, or incomplete artifact evidence |

This separation lets users distinguish verified runs, reference benchmark rows,
visual samples, and diagnostic failures.

# Verification and Reproducibility

The package exposes the following minimum verification commands:

```bash
pip install -e .
python -c "import topogpu; print(topogpu.__version__)"
python examples/cantilever_3d.py --small
topogpu verify --small
topogpu run cases/cantilever_3d.yaml --small --backend cpu
topogpu render runs/cantilever_3d
topogpu benchmark cases/production_suite.yaml
pytest
```

The numerical verification script checks matrix-free operator action, energy,
symmetry, diagonal extraction, sensitivity finite differences, and density
filter behavior on small reference cases. Production benchmark rows are
admissible only when final residual, Krylov cap status, volume error, history
completeness, timing fields, memory fields, and manifest hashes are present.

# Relationship to the Technical Paper

This software paper is the citation anchor for the package. The longer
technical manuscript reports the SolverV4-backed workflow, production and
stress case separation, residual histories, timing decomposition, render
protocols, and limitations in more detail. The package paper is intentionally
short: it documents why the software exists, how to install it, how to run it,
what it verifies, and how to cite it.

# Availability

Repository URL: `https://github.com/nbbllxx0/TopoGPU`

Release: `v0.1.0`.

Archived DOI: `10.5281/zenodo.20100693`.

License: BSD-3-Clause.

Documentation: `https://github.com/nbbllxx0/TopoGPU/tree/main/docs`.

Issue tracker and pull requests: `https://github.com/nbbllxx0/TopoGPU/issues`.

# References
