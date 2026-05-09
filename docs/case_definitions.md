# Case Definitions

TopoGPU v0.1.0 supports structured-grid hexahedral SIMP cases. A case records:

- element counts `nel = (nelx, nely, nelz)`;
- volume fraction;
- density-filter radius;
- support description;
- load description;
- evidence role.

The public gallery currently exposes cantilever-style examples directly and bridges selected tool-paper cases through the verified `gpu_fem` case builder.

```python
import topogpu as tg

problem = tg.gallery.cantilever_3d(
    nel=(24, 12, 6),
    volfrac=0.30,
    support="xmin",
    load="tip_patch_z",
    filter_radius=1.5,
)
problem.validate()
```

The first release is intentionally narrow: it does not provide CAD import, unstructured meshing, nonlinear mechanics, or manufacturing constraints.
