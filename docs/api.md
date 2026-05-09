# API

Primary public objects:

- `TopologyProblem`
- `CaseGallery`
- `DensityFilter`
- `SIMPSolver`
- `OptimizationResult`
- `EvidenceBundle`

The first release keeps the public API narrow and structured-grid focused. The
existing `gpu_fem` SolverV4 backend remains available as implementation
infrastructure, but new users should start with `topogpu.gallery` and
`topogpu.SIMPSolver`.

