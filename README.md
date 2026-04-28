# A Matrix-Free Galerkin Multigrid Solver and Failure-Mode Screen for Single-GPU 3D SIMP Linear Systems

Companion repository for:

> Yang, S., Wang, J., and Wang, Y. (2026).
> *A Matrix-Free Galerkin Multigrid Solver and Failure-Mode Screen for Single-GPU 3D SIMP Linear Systems.*
> Preprint in preparation.

## Status

**Public repository address reserved.** The code payload is staged for upload
when the arXiv version of the paper is online. The release will be code-only and
will include the implementation, experiment drivers, figure-generation scripts,
environment specification, citation metadata, license text, and reproduction
documentation.

The code release will not include the manuscript source/PDF, submission
manifests, author-side reference CSVs, logs, retained density arrays, root data
directories, or generated manuscript figures. The manuscript source package is
the source for submitted paper files and qualitative gallery density fields.

## Scope

This paper builds directly on the companion paper:

> Yang, S., Wang, J., and Wang, Y. (2026).
> *Matrix-Free 3D SIMP Topology Optimization with Fused Gather-GEMM-Scatter
> Kernels.* arXiv:2604.18020.
> Companion repository:
> https://github.com/nbbllxx0/Fused-Gather-GEMM-Scatter-Kernels

The Level-0 fused gather-GEMM-scatter matvec kernel introduced in that paper is
used here as fixed infrastructure. The present work contributes the
matrix-free Galerkin multigrid hierarchy, Chebyshev-Jacobi smoothing options,
mixed-precision hierarchy policies, the solver integration used in the paper,
and the failure-mode screens reported in the benchmark suite.

## Planned Contents

- `src/gpu_fem/`: matrix-free Galerkin GMG hierarchy and SIMP solver code
- `experiments/paper4/`: validation gates M1-M8 and benchmark suite E1-E10
- `figures/`: scripts for regenerating paper figures from local rerun outputs
- `docs/ARTIFACT_MAP.md`: mapping from paper results to code drivers and output files
- `environment.yml`: conda environment specification
- `CITATION.cff` and `LICENSE`: citation and licensing metadata

## License

Code will be released under the BSD 3-Clause license at the time of arXiv
publication, matching the license of the companion repository.
