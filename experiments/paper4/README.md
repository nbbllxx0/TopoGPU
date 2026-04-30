# Paper 4 Experiments

This directory contains the public benchmark drivers for the mixed-precision
GMG paper. The public release is code-only and does not ship manuscript
reference CSVs.

Main entrypoints:

- `validate_phase1.py`: pre-benchmark validation gates M1--M8. Writes
  `results_phase1.json`.
- `run_experiments_e1_e10.py`: benchmark-suite driver for the E1--E10 CSV
  bundle. By default, fresh reruns should be written to a local output directory
  such as `rerun_outputs/paper4` or to the ignored top-level files in this
  directory.

Supporting diagnostics:

- `verify_level1_galerkin.py`: exactness check for the level-1 elementwise
  Galerkin assembly against an explicit triple product on a small cantilever.
- `estimate_preconditioned_kappa.py`: small-problem `kappa_eff` and symmetry
  diagnostic used only to interpret the PCG/FGMRES policy; it is not a
  primary quantitative result.

Exploratory scripts retained for context, not as current benchmark entrypoints:

- `benchmark_linear_solves.py`
- `benchmark_simp_paper4.py`

Output note:

- Fresh local outputs written as top-level `experiments/paper4/*.csv` or
  `experiments/paper4/*.json` are ignored by git so they do not overwrite the
  code release accidentally.
