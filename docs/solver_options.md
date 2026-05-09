# Solver Options

`SIMPSolver` exposes a small stable interface:

```python
solver = tg.SIMPSolver(
    backend="cpu",
    linear_solver="auto",
    optimizer="oc",
    tol=1e-5,
    max_krylov=800,
    max_iter=12,
    move=0.10,
    rho_min=1e-3,
)
```

Backends:

- `cpu`: small smoke tests through the bundled CPU path.
- `cuda` or `cupy`: SolverV4 matrix-free GPU path.

The default optimizer is bounded OC. Other optimizers are not implemented in v0.1.0.

For CUDA runs, use `tol` and `max_krylov` to set residual and cap behavior. Runs with cap hits or residual failures should be treated as stress diagnostics unless the evidence gates pass.
