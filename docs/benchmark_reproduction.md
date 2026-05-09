# Benchmark Reproduction

The reference suite declarations live in `cases/`.

```bash
topogpu benchmark cases/production_suite.yaml
topogpu benchmark cases/stress_suite.yaml --out runs/stress_suite
```

The benchmark command records the declared suite membership. Full production timing rows should be regenerated only after verification succeeds:

```bash
topogpu verify --small
python examples/cantilever_3d.py --small --backend cpu
```

Warm timing values in the software paper exclude first-iteration setup and JIT-visible costs. Hardware, driver, CUDA, CuPy, and GPU memory determine whether a user run matches the reference timings.
