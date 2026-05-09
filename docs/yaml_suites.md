# YAML Suites

YAML files provide a stable way to run examples and declare benchmark suites.

Single-case example:

```yaml
name: cantilever_3d
nel: [24, 12, 6]
volfrac: 0.30
filter_radius: 1.5
support: xmin
load: tip_point_y
role: example
```

Run a YAML case:

```bash
topogpu run cases/cantilever_3d.yaml --small --backend cpu --iters 2
```

Suite example:

```yaml
name: production_suite
cases:
  - tool_short_cantilever_vf25
criteria:
  residual_max: 1.0e-5
  cap_hits: 0
  volume_error_max: 1.0e-3
```

Declare the suite manifest:

```bash
topogpu benchmark cases/production_suite.yaml
```
