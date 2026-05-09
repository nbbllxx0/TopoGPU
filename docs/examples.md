# Examples

Run the lightweight CPU smoke example:

```bash
python examples/cantilever_3d.py --small
```

Run a CUDA SolverV4 example after installing the full GPU environment:

```bash
python examples/cantilever_3d.py --backend cuda --out runs/cantilever_cuda
```

Run numerical verification:

```bash
python examples/verify_numerics.py
```

