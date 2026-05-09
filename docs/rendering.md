# Rendering

TopoGPU stores render metadata alongside density arrays so figures can be traced to run artifacts.

```bash
topogpu render runs/cantilever_3d
```

The metadata records:

- density threshold;
- scalar field name;
- scalar normalization policy;
- source command.

The default density threshold is `0.5`. Figure colors are normalized per case unless a script explicitly records shared normalization.
