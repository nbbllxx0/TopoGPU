# CUDA and CuPy Troubleshooting

Check the installed package:

```bash
python -c "import topogpu; print(topogpu.__version__)"
python -c "import cupy; print(cupy.__version__); print(cupy.cuda.runtime.runtimeGetVersion())"
```

Common issues:

- CuPy build does not match the installed CUDA runtime.
- NVIDIA driver is too old for the CUDA runtime.
- `CUDA_PATH` points to an older toolkit.
- GPU memory is insufficient for the selected mesh.
- First CUDA run includes JIT/cache overhead and should not be used as warm timing.

Use CPU smoke tests first:

```bash
python examples/cantilever_3d.py --small --backend cpu
```

Then test CUDA on a small case:

```bash
python examples/cantilever_3d.py --small --backend cuda
```
