# Software Paper Submission Checklist

## Required Before Submission

- [ ] Confirm `https://github.com/nbbllxx0/TopoGPU` is public and browsable.
- [ ] Confirm BSD-3-Clause license is visible in the public repository.
- [ ] Create and test a fresh `topogpu` environment from `environment.yml`.
- [ ] Run `pip install -e .`.
- [ ] Run `python -c "import topogpu; print(topogpu.__version__)"`.
- [ ] Run `python examples/cantilever_3d.py --small`.
- [ ] Run `topogpu verify`.
- [ ] Run `pytest`.
- [ ] Publish `v0.1.0` GitHub release.
- [ ] Archive release with Zenodo.
- [ ] Insert DOI into `CITATION.cff`, `README.md`, docs, and `paper.md`.

## Recommended Before Submission

- [ ] Add screenshots or figures generated from package examples.
- [ ] Add issue templates and contribution notes.
- [ ] Add API documentation generated from `src/topogpu`.
- [ ] Add one CI workflow for import and lightweight tests.
- [ ] Confirm all production/stress case labels match the technical paper.

## Scope Guard

This paper should remain short and package-first. Detailed residual histories,
cap diagnostics, timing decomposition, raw artifact schemas, and stress-case
analysis belong in the longer technical manuscript or supplement.
