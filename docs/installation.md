# Installation

```bash
git clone https://github.com/nbbllxx0/TopoGPU.git
cd topogpu
conda env create -f environment.yml
conda activate topogpu
pip install -e .
python -c "import topogpu; print(topogpu.__version__)"
```

The CUDA path requires a working NVIDIA driver and a CuPy build matching the
local CUDA runtime. The release environment pins `cupy-cuda13x`; on Windows
RTX 50-series workstations, set CUDA 13 before running GPU verification:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
```

CPU smoke examples can run without CuPy.
