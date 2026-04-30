# Mixed-Precision GMG Code Release Map

**Title:** A Matrix-Free Galerkin Multigrid Solver and Failure-Mode Screen for Single-GPU 3D SIMP Linear Systems

**Release scope:** code-only public GitHub release accompanying arXiv:2604.26441
(https://arxiv.org/abs/2604.26441). The repository contains implementation
code, experiment drivers, figure-generation scripts, documentation, citation
metadata, license text, and the environment specification. It intentionally does
not include manuscript source/PDF files, submission manifests, runtime
snapshots, generated result files, raw logs, retained density arrays, root data
directories, or generated manuscript figures.

Fresh reruns should be written to an explicit output directory such as
`rerun_outputs/paper4`. Figure-generation scripts should be pointed at those
local outputs.

---

## 1. Core Implementation

| Paper topic | Code-release file |
|---|---|
| Mixed-precision GMG hierarchy, cycle type, and `kappa_eff` probe | `src/gpu_fem/multigrid_v4.py` |
| Matrix-free SIMP driver and GMG integration | `src/gpu_fem/solver_v4.py` |
| Jacobi-PCG and `MatrixFreeKff` baseline paths used by comparisons | `src/gpu_fem/solver_v2.py` |
| BF16 WMMA fused gather-GEMM-scatter kernel | `src/gpu_fem/cuda_fused_matvec.py` |
| Reproducible Python package pins | `environment.yml` |

## 2. Pre-Benchmark Validation Checks

Run:

```bash
python experiments/paper4/validate_phase1.py --out results_phase1.json
```

| ID | Gate | Pass criterion |
|---|---|---|
| M1 | FP64 V-cycle vs direct solve on 64k cantilever | residual < 1e-10 |
| M2 | Uniform-density 64k/216k/512k probe | FGMRES iterations <= 30 |
| M3 | Matrix-free vs assembled-Galerkin compliance | relative difference < 0.1% |
| M4 | Chebyshev vs Jacobi smoother | converges with iterations <= 50 |
| M5 | Selected SIMP sanity probes for `p in {1.5, 3.0, 4.5}` and `E_min = 1e-9` | converges on the validation probes |
| M6 | Nominal 64k, `p = 3`, `rho = 0.5` probe | `kappa_eff <= 256` |
| M7 | BF16 drop-in compliance check | compliance error <= 0.5%; convergence recorded separately |
| M8 | Three-level FP32 hierarchy on four benchmarks | compliance error <= 0.5% |

## 3. Main Benchmark Suite

Run:

```bash
python experiments/paper4/run_experiments_e1_e10.py --experiments all --out rerun_outputs/paper4
```

| Experiment | Generated figure/table inputs | Script stage | Local output file |
|---|---|---|---|
| E1 outer iteration count vs mesh size | `F1_vcycle_iters.pdf` | `e1_vcycle_iteration_count` | `e1_vcycle_iters.csv` |
| E2 per-linear-solve wall time and capped-baseline ratios | `F2_solve_scaling.pdf`, `F3_solve_speedup.pdf` | `e2_per_solve_wall_time` | `e2_per_solve_wall_time.csv`, `e2_per_solve_wall_time_trials.csv` |
| E2 residual histories | `F12_residual_histories.pdf` | `e2_per_solve_wall_time` | `e2_residual_histories.csv` |
| E3 auxiliary 30-step OC schedule timing | `F4_simp_speedup.pdf` | `e3_simp_speedup` | `e3_simp_speedup.csv` |
| E3 SIMP trajectory | `F13_simp_trajectory.pdf` | `e3_simp_speedup` | `e3_simp_trajectory.csv` |
| E4 tensor-core throughput | `F5_tc_throughput.pdf` | `e4_tc_throughput` | `e4_tc_throughput.csv` |
| E4 roofline placement | `F14_roofline.pdf` | `e4_tc_throughput` | `e4_roofline.csv` |
| E5 `kappa_eff` spectral-proxy map | `F6_kappa_eff.pdf` | `e5_kappa_eff` | `e5_kappa_eff.csv` |
| E5 direct BF16 validation | paper table input | `e5_bf16_direct_validation` (`E5B`) | `e5_bf16_validation.csv`, `e5_bf16_validation_residual_histories.csv` |
| E6 ablations | `F7_ablations.pdf` | `e6_ablations` | `e6a_precision_ablation.csv`, `e6a_precision_ablation_trials.csv`, `e6b_depth_sweep.csv`, `e6b_depth_sweep_trials.csv`, `e6c_vcycle_vs_wcycle.csv`, `e6c_vcycle_vs_wcycle_trials.csv`, `e6d_smoother_type.csv`, `e6d_smoother_type_trials.csv` |
| E6 joint sensitivity sweep | `F15_sensitivity_surface.pdf` | `e6_ablations` | `e6_sensitivity_surface.csv`, `e6_sensitivity_surface_trials.csv` |
| E6 high-contrast smoother screen | paper table input | `e6_high_contrast_smoother_ablation` (`E6H`) | `e6_high_contrast_smoother.csv`, `e6_high_contrast_smoother_residual_histories.csv` |
| E7 large-scale single solves | `F8_large_scale.pdf` | `e7_large_scale` | `e7_large_scale.csv`, `e7_large_scale_trials.csv` |
| E8 external baseline | `F9_external_baseline.pdf` | `e8_external_baseline` | `e8_external_baseline.csv`, `e8_external_baseline_trials.csv` |
| E9 energy efficiency | table input | `e9_energy` | `e9_energy.csv`, `e9_energy_trials.csv` |
| E10 robustness table | `F10_robustness.pdf` | `e10_robustness_edges` | `e10_robustness.csv` |
| E10 single-seed basin screen | `F16_robustness_basin.pdf` | `e10_robustness_edges` | `e10_basin.csv` |

## 4. Figure Generation

Quantitative figures can be regenerated from locally produced CSVs:

```bash
cd figures
python plot_figures.py --results-dir ../rerun_outputs/paper4 --figs-dir ../rerun_outputs/paper4_figs
```

Qualitative gallery generation requires locally supplied density arrays:

```bash
cd figures
python make_3d_renders.py --renders-dir ../rerun_outputs/topology_renders --figs-dir ../rerun_outputs/paper4_figs
```

The code-only release does not ship retained density arrays or generated figure
directories.

## 5. Provenance Notes

- Top-level `experiments/paper4/*.csv` and `*.json` files are local rerun
  outputs and are ignored by git.
- `rerun_outputs/` is the recommended location for local benchmark outputs.
- Generated figures should be treated as local build products, not as tracked
  public-release contents.
- Manuscript result files and retained qualitative density fields remain
  outside this code-only public-release tree.

### Generated CSV label map

These identifiers may appear in generated CSV/log artifacts and scripts. The
manuscript translates them to reader-facing labels before they appear in text,
tables, or figures.

| Reader-facing label | Generated identifier | Where it may appear |
|---|---|---|
| 216k cantilever benchmark | `cantilever_gpu_large`, `cantilever_216k` | E3/E4/E9 CSVs and trajectories |
| Small torsion auxiliary schedule case | `torsion_small` | E3 schedule and trajectory CSVs |
| Small MBB-style auxiliary schedule case | `mbb_small` | E3 schedule and trajectory CSVs |
| Jacobi-PCG reference trajectory | `paper3_jacobi` | E3 trajectory CSVs |
| FP32-GMG trajectory | `paper4_gmg_fp32` | E3 trajectory CSVs |
