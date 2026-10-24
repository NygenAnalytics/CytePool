---
myst:
  html_meta:
    description: Build pseudobulk cubes from cell-level or bulk counts with CytePool, and keep sample, group and feature metadata in tables.
---

# Pseudobulk cubes and tables

A cube holds additive statistics of the cells of each sample and group (such as a
cell type), for every feature: counts, detected cells, and sums of normalized and
log-normalized expression. Because the statistics add up, groups can be merged and
samples pooled without going back to the cells. {func}`~cytepool.cube.from_cells`
pools a cell-level count matrix in one pass, and {func}`~cytepool.cube.from_bulk`
wraps a samples x features count matrix as a cube with one group. Sample, group and
feature metadata are {class}`~cytepool.table.Table` objects.

```{eval-rst}
.. automodule:: cytepool.cube
```

```{eval-rst}
.. automodule:: cytepool.table
```
