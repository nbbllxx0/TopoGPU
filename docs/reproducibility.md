# Reproducibility

Primary commands:

```bash
topogpu verify
topogpu run --case cantilever_3d --backend cpu --iters 3 --out runs/cantilever_3d
python experiments/tool_paper/build_toolpaper_artifacts.py
```

For GPU verification on Windows CUDA 13 workstations, set:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
```

The artifact convention is:

- `history.csv`
- `summary.json`
- `rho_final.npy`
- `ARTIFACT_MANIFEST.csv`

Production timing rows must pass residual, cap-hit, volume-error, and manifest
completeness gates before they are used as benchmark evidence.
