# Quick Start

```python
import topogpu as tg

problem = tg.gallery.cantilever_3d(nel=(24, 12, 6), volfrac=0.30)
result = tg.SIMPSolver(backend="cpu", max_iter=3).solve(problem)
result.save("runs/cantilever_3d")
```

For the SolverV4 GPU path:

```python
result = tg.SIMPSolver(
    backend="cuda",
    linear_solver="pcg_gmg",
    tol=1e-5,
    max_krylov=800,
    max_iter=12,
).solve(problem)
```

