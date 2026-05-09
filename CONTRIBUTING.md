# Contributing to TopoGPU

TopoGPU is a research software package for structured-grid 3D SIMP topology optimization. Contributions should keep the public API, tests, and evidence outputs reproducible.

## Development Setup

```bash
conda env create -f environment.yml
conda activate topogpu
pip install -e .
pytest
topogpu verify --small
```

CUDA examples require a compatible NVIDIA driver, CUDA runtime, and CuPy build. CPU smoke tests should run without a GPU.

## Issue Reports

Please use GitHub Issues for bug reports, installation failures, numerical-verification failures, and feature requests. Include:

- operating system, Python version, CUDA version, CuPy version, and GPU model;
- command run and complete error message;
- case YAML or gallery case name;
- `summary.json`, `history.csv`, and `ARTIFACT_MANIFEST.csv` when reporting run behavior.

## Pull Requests

Pull requests should:

- keep changes narrowly scoped;
- add or update tests for behavior changes;
- update documentation for changed commands, outputs, or public APIs;
- preserve manifest-tracked artifact conventions;
- avoid changing benchmark/admissibility thresholds without a documented reason.

Run `pytest` before opening a pull request. Run `topogpu verify --small` when changing solver, filter, case, or evidence logic.
