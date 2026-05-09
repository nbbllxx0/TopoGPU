# Evidence Bundles

`OptimizationResult.save(path)` writes a manifest-tracked run directory:

```text
runs/case_id/
  history.csv
  summary.json
  rho_final.npy
  render_metadata.json
  ARTIFACT_MANIFEST.csv
```

Runs that include displacement or scalar render data may also write:

```text
disp_elem.npy
```

Evidence roles:

| Role | Downstream use |
| --- | --- |
| Verification | numerical consistency checks |
| Production timing | residual-clean, cap-free benchmark rows |
| Visual sample | topology images with solver status disclosed |
| Stress diagnostic | reported cap/residual failures |
| Invalid | excluded from claims |

Production rows require final residual `<= 1e-5`, zero Krylov cap hits, volume error `<= 1e-3`, complete history, timing/memory fields, and manifest hashes.
