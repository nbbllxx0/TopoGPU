# TopoGPU v0.1.0 Release Checklist

This repository is prepared for a package-first TopoGPU release, but the
release is not complete until the external publication steps below are done.

## Local Package Gate

- [x] Package name: `topogpu`
- [x] Version: `0.1.0`
- [x] `pyproject.toml`
- [x] `environment.yml`
- [x] `README.md`
- [x] `LICENSE`
- [x] `CITATION.cff`
- [x] Public API under `src/topogpu/`
- [x] Examples under `examples/`
- [x] Case declarations under `cases/`
- [x] Tests under `tests/`
- [x] Documentation under `docs/`
- [x] Software paper draft under `paper/`
- [x] Evidence bundle and SHA256 manifest helpers
- [x] CLI entry point: `topogpu`

## Verification Gate

- [x] Lightweight import/API/CLI tests pass in the ambient environment
- [x] Full SolverV4 verification passes in the pinned `topogpu` environment
- [x] Small CUDA example passes in the pinned `topogpu` environment
- [ ] Release artifact manifest is regenerated from the final release tree

## GitHub and Archive Gate

- [ ] Create or confirm public repository: `https://github.com/nbbllxx0/topogpu`
- [ ] Commit the intended release files on `main`
- [ ] Tag `v0.1.0`
- [ ] Push `main` and `v0.1.0`
- [ ] Publish the GitHub release
- [x] Archive the release with Zenodo
- [x] Insert the Zenodo DOI into `README.md`, `CITATION.cff`, docs, and papers

## Paper Gate

- [x] Short software-paper draft folder exists
- [x] Full software-publication manuscript folder exists
- [x] Longer technical manuscript and supplement exist
- [x] Replace release placeholders with final repository/tag/DOI metadata
