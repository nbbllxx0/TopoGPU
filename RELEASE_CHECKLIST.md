# TopoGPU v0.1.0 Release Checklist

This repository has a public package-first TopoGPU release. The v0.1.0 tag is
archived on Zenodo; later documentation and manuscript hardening changes are on
`main` and should be released as v0.1.1 if a new archived snapshot is needed.

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
- [x] Release artifact manifest is regenerated from the final release tree

## GitHub and Archive Gate

- [x] Create or confirm public repository: `https://github.com/nbbllxx0/TopoGPU`
- [x] Commit the intended release files on `main`
- [x] Tag `v0.1.0`
- [x] Push `main` and `v0.1.0`
- [x] Publish the GitHub release
- [x] Archive the release with Zenodo
- [x] Insert the Zenodo DOI into `README.md`, `CITATION.cff`, docs, and papers

## Paper Gate

- [x] Short software-paper draft folder exists
- [x] Full software-publication manuscript folder exists
- [x] Longer technical manuscript and supplement exist
- [x] Replace release placeholders with final repository/tag/DOI metadata
- [ ] Add JOSS-style AI usage disclosure to the software paper when the user is ready to address it
- [x] Remove pending-release and internal manuscript-use wording
