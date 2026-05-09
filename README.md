# TopoGPU

GPU-accelerated 3D SIMP topology optimization in Python, with structured case
definitions, matrix-free SolverV4 state solves, bounded OC updates,
verification checks, and manifest-tracked evidence bundles.

This repository is being prepared as the package-first release for the
toolkit/software paper:

> **TopoGPU: GPU-Accelerated 3D SIMP Topology Optimization in Python**

The package currently wraps the verified `gpu_fem` SolverV4 research core while
the paper-facing scripts are being consolidated into a stable public API.

## Package Quick Start

```bash
git clone https://github.com/nbbllxx0/TopoGPU.git
cd topogpu
conda env create -f environment.yml
conda activate topogpu
pip install -e .
python -c "import topogpu; print(topogpu.__version__)"
python examples/cantilever_3d.py --small
topogpu verify
```

On RTX 50-series / CUDA 13 workstations, make sure the CUDA 13 toolkit is the
active toolkit before running GPU verification:

```powershell
$env:CUDA_PATH='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0'
$env:CUDA_HOME=$env:CUDA_PATH
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
```

Minimal Python API:

```python
import topogpu as tg

problem = tg.gallery.cantilever_3d(nel=(24, 12, 6), volfrac=0.30)
result = tg.SIMPSolver(backend="cpu", max_iter=3).solve(problem)
result.save("runs/cantilever_3d")
```

SolverV4 GPU path:

```python
result = tg.SIMPSolver(
    backend="cuda",
    linear_solver="pcg_gmg",
    tol=1e-5,
    max_krylov=800,
    max_iter=12,
).solve(problem)
```

## Public Release Checklist

- `pyproject.toml` for editable/package installation
- `src/topogpu/` public API facade
- `src/gpu_fem/` existing SolverV4 implementation core
- `examples/` runnable package examples
- `cases/` YAML case-suite declarations
- `tests/` import/filter/API smoke tests
- `docs/` installation, quickstart, reproducibility, and limitations pages
- `paper/` short software-paper draft and bibliography
- `CITATION.cff` with repository and Zenodo DOI metadata

## Legacy Solver Paper Context

The solver core below remains the technical basis for TopoGPU's SolverV4
backend and the longer technical/performance manuscript.

# A Failure-Aware Matrix-Free Galerkin Multigrid Preconditioner for Single-GPU 3D SIMP Elasticity Systems

Code-only public release accompanying:

> Yang, S., Wang, J., and Wang, Y. (2026).  
> *A Failure-Aware Matrix-Free Galerkin Multigrid Preconditioner for Single-GPU
> 3D SIMP Elasticity Systems.* arXiv:2604.26441  
> https://arxiv.org/abs/2604.26441

This repository is the code-only public release accompanying the arXiv
preprint and journal submission. It contains source code, experiment drivers, figure-generation
scripts, environment pins, citation metadata, and documentation. It
intentionally does not ship the manuscript source/PDF, submission manifests,
runtime snapshots, generated result files, raw logs, retained density arrays,
root data directories, or generated manuscript figures.

## Scope

The level-0 fused gather-GEMM-scatter operator is inherited infrastructure from
the companion paper:

> Yang, S., Wang, J., and Wang, Y. (2026).  
> *Matrix-Free 3D SIMP Topology Optimization with Fused Gather-GEMM-Scatter
> Kernels.* arXiv:2604.18020

The code packaged here implements the matrix-free full-Galerkin GMG hierarchy,
a Chebyshev-Jacobi smoother option, the mixed-precision level policy, the
GMG-integrated solver path, and the drivers used to evaluate solver behavior.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/gpu_fem/` | GMG hierarchy, baseline solver path, SIMP driver integration, fused level-0 kernel |
| `experiments/paper4/validate_phase1.py` | Pre-benchmark validation checks M1-M8 |
| `experiments/paper4/estimate_preconditioned_kappa.py` | Small symmetry / `kappa_eff` diagnostic used only to interpret solver-policy wording |
| `experiments/paper4/run_experiments_e1_e10.py` | Main benchmark suite E1-E10 |
| `experiments/paper4/README.md` | Experiment-driver notes |
| `figures/plot_figures.py` | Regenerates quantitative figures from locally produced CSV outputs |
| `figures/make_3d_renders.py` | Regenerates qualitative gallery panels from locally supplied density arrays |
| `docs/ARTIFACT_MAP.md` | Code-entrypoint and output map |
| `ci/smoke_test.py` | Short sanity check for the release stack |
| `environment.yml` | Reference conda environment |

## Hardware and Software Requirements

- NVIDIA GPU with compute capability 8.0 or newer for the BF16 WMMA path.
- A 24 GiB card is needed for the largest E7 entry at 1M elements.
- Smaller experiments fit on lower-VRAM cards, but the largest scaling result
  does not.
- Python and package versions are pinned in `environment.yml`.

The TopoGPU release environment pins `cupy-cuda13x` for current CUDA 13
workstations. If your GPU does not expose BF16 WMMA, the FP32/FP64 paths still
remain useful for partial experiments, but BF16-specific rows and tensor-core
proxy measurements are not directly comparable.

## Legacy Solver-Core Installation

```bash
git clone https://github.com/nbbllxx0/TopoGPU.git
cd topogpu
conda env create -f environment.yml
conda activate topogpu
pip install -e .
```

The older experiment and figure scripts still resolve paths relative to the
repository root, but the package-first workflow should use the editable install
and the `topogpu` command.

## Quick Sanity Check

```bash
python ci/smoke_test.py
```

This runs one FP64 V-cycle-preconditioned solve on a 64k cantilever probe and
checks that the relative residual drops below `1e-10` within 300 iterations.
If this fails, fix the local CUDA / CuPy stack before running the full suite.

## Step 1: Pre-Benchmark Validation Checks

```bash
python experiments/paper4/validate_phase1.py --out results_phase1.json
```

These checks gate the hierarchy before performance claims are collected.

| ID | Check |
| --- | --- |
| M1 | FP64 V-cycle against a direct solve |
| M2 | Outer-iteration bound on uniform-density probes |
| M3 | Matrix-free vs assembled-Galerkin compliance agreement |
| M4 | Chebyshev-Jacobi smoother convergence |
| M5 | Selected SIMP sanity probes over representative penalties |
| M6 | `kappa_eff <= 256` on the nominal probe |
| M7 | BF16 drop-in compliance error |
| M8 | Three-level FP32 hierarchy on representative benchmarks |

The output file `results_phase1.json` records the measured quantities and
pass/fail status for each gate.

Optional small diagnostic:

```bash
python experiments/paper4/estimate_preconditioned_kappa.py
```

This probe is not a headline benchmark. It is retained only to interpret the
PCG-vs-FGMRES policy used in different parts of the suite.

## Step 2: Main Benchmark Suite

Run the whole benchmark suite into an explicit local output directory:

```bash
python experiments/paper4/run_experiments_e1_e10.py --experiments all --out rerun_outputs/paper4
```

Run a subset:

```bash
python experiments/paper4/run_experiments_e1_e10.py --experiments E1 E2 E5B E6H E7 --out rerun_outputs/paper4
```

Timing controls:

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `PAPER4_N_WARMUP` | `2` | Warm-up trials discarded before timing |
| `PAPER4_N_TRIALS` | `10` | Timed trials used for reported mean/std |

Main outputs written under the selected output directory:

| Experiment | Primary outputs |
| --- | --- |
| E1 | `e1_vcycle_iters.csv` |
| E2 | `e2_per_solve_wall_time.csv`, `e2_per_solve_wall_time_trials.csv`, `e2_residual_histories.csv` |
| E3 | `e3_simp_speedup.csv`, `e3_simp_trajectory.csv` |
| E4 | `e4_tc_throughput.csv`, `e4_roofline.csv` |
| E5 | `e5_kappa_eff.csv`, `e5_bf16_validation.csv`, `e5_bf16_validation_residual_histories.csv` |
| E6 | `e6a_precision_ablation.csv`, `e6a_precision_ablation_trials.csv`, `e6b_depth_sweep.csv`, `e6b_depth_sweep_trials.csv`, `e6c_vcycle_vs_wcycle.csv`, `e6c_vcycle_vs_wcycle_trials.csv`, `e6d_smoother_type.csv`, `e6d_smoother_type_trials.csv`, `e6_sensitivity_surface.csv`, `e6_sensitivity_surface_trials.csv`, `e6_high_contrast_smoother.csv`, `e6_high_contrast_smoother_residual_histories.csv` |
| E7 | `e7_large_scale.csv`, `e7_large_scale_trials.csv` |
| E8 | `e8_external_baseline.csv`, `e8_external_baseline_trials.csv` |
| E9 | `e9_energy.csv`, `e9_energy_trials.csv` |
| E10 | `e10_robustness.csv`, `e10_basin.csv` |

## Step 3: Figure Generation From Local Outputs

Point the quantitative figure script at the output directory produced by a
local rerun:

```bash
cd figures
python plot_figures.py --results-dir ../rerun_outputs/paper4 --figs-dir ../rerun_outputs/paper4_figs
```

This writes the manuscript quantitative figure set, including both the combined
and split FP32/BF16 E6 sensitivity-surface figures.

Qualitative gallery generation requires locally supplied density arrays:

```bash
cd figures
python make_3d_renders.py --renders-dir ../rerun_outputs/topology_renders --figs-dir ../rerun_outputs/paper4_figs
```

This writes the qualitative gallery, the four-panel main-context image, and the
individual qualitative panels used for provenance checks.

The released repository does not include retained density arrays or generated
figure directories.

## Important Interpretation Notes

- E2 wall-time ratios are measured against a Jacobi-PCG reference path that
  reaches the 200-iteration cap without convergence in all timed trials at
  64k, 216k, and 512k elements. Interpret these values as capped-baseline
  wall-time ratios, not as speedups over successful Jacobi solves.
- E3 is an auxiliary fixed 30-step SIMP schedule. It should be read as a
  same-schedule execution study, not as a matched-endpoint continuation
  comparison.
- E3 records both `compliance_err_k0_pct` and `compliance_err_final_pct`.
- Some generated CSVs retain compact code identifiers for benchmark cases and
  trajectory sources; `docs/ARTIFACT_MAP.md` maps those identifiers to the
  reader-facing labels used in the paper.
- E1, E2, E6(a-d), and E8 retain the paper's empirical PCG pairings for the
  headline FP32/FP64 comparison paths; the small
  `estimate_preconditioned_kappa.py` probe records an approximate symmetry
  defect, so those rows should be read as empirical solver pairings rather than
  as an exact SPD certificate.
- E6 sensitivity-surface runs use explicit FGMRES paths for both FP32 and BF16
  rows.
- E7 large-scale runs use FGMRES with restart 50.
- E10 uses FGMRES with restart 50; the headline robustness table uses
  `maxiter = 500`, while the broader basin diagnostic uses `maxiter = 300`.
- In E7, `vram_delta_mb` is the setup-time hierarchy-allocation delta measured
  around `gmg.setup(E_e)`. It is not a peak setup-plus-solve memory trace.
- E8 reference timings use a cold-start GMG hierarchy-initialization
  measurement that reconstructs the FP32 GMG object and performs the first
  initialization call on each trial from precomputed free-set and
  element-connectivity tables.
- E4 benchmarks the 216k cantilever case (`120x60x30`) with a uniform
  `E_e = 1` field and an all-ones free-DOF input vector.
- E4 roofline guide rails use RTX 4090 vendor-spec theoretical ceilings; they
  are not measured saturation ceilings.
- E9 records the 216k cantilever solve (`120x60x30`, uniform `rho = 0.5`,
  `penal = 3.0`) and computes energy from timestamped NVML samples integrated
  over the measured solve interval. The sampler runs every `50 ms`; no explicit
  idle-power baseline is subtracted.
- The qualitative gallery is visual context only. It is not part of the
  E1-E10 quantitative result bundle.

## Troubleshooting

If `ci/smoke_test.py` fails:

- verify that CuPy can see the intended GPU
- make sure the CUDA toolkit and driver are compatible with the pinned
  environment
- clear stale local caches only after confirming you are not hitting a simple
  installation mismatch

If a 1M E7 run runs out of memory:

- reduce `PAPER4_N_TRIALS`
- run smaller subsets first
- do not interpret reduced-trial timing noise as a regression in solver logic

If figure generation fails:

- confirm that `--results-dir`, `--renders-dir`, and `--figs-dir` point to
  local outputs that actually exist
- regenerate benchmark CSVs before running `plot_figures.py`
- provide locally generated or locally retained density arrays before
  running `make_3d_renders.py`

## Citation and License

Citation metadata are recorded in `CITATION.cff`. The paper is available at
https://arxiv.org/abs/2604.26441. The archived TopoGPU software DOI is
https://doi.org/10.5281/zenodo.20100693. The release license is BSD
3-Clause; see `LICENSE`.
