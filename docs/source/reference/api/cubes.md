---
myst:
  html_meta:
    description: Build pseudobulk cubes from cell-level or bulk counts with CytePool, keep sample, group and feature metadata in tables, record what each cube pooled with a provenance record and a content fingerprint, and align the cubes of several studies by sample, gene and cell type.
---

# Pseudobulk cubes and tables

A cube holds additive statistics of the cells of each sample and group (such as a
cell type), for every feature: counts, detected cells, and sums of normalized and
log-normalized expression. Because the statistics add up, groups can be merged and
samples pooled without going back to the cells. {func}`~cytepool.cube.from_cells`
pools a cell-level count matrix in one pass, and {func}`~cytepool.cube.from_bulk`
wraps a samples x features count matrix as a cube with one group. Sample, group and
feature metadata are {class}`~cytepool.table.Table` objects.
{class}`~cytepool.cube.CubeCollection` aligns the cubes of several studies by sample,
gene and cell type, with what each study measured, for analyses of each study and for
one cube of them all.

```{eval-rst}
.. automodule:: cytepool.cube
```

```{eval-rst}
.. automodule:: cytepool.table
```

## Fingerprints and provenance

A cube's fingerprint identifies its values, tables and cell numbers included, so a
result can name the cube it was computed from, and its provenance records how the cube
was made: the caller's record of the source, the settings of the function that built
it, and every operation that derived it, with what it kept or pooled and the
fingerprint of the cube it was applied to. {meth}`~cytepool.cube.Cube.history` reads
the record back in plain terms.

```{eval-rst}
.. automodule:: cytepool.cube.fingerprints
   :exclude-members: FINGERPRINT_VERSION
```

```{eval-rst}
.. py:data:: FINGERPRINT_VERSION
   :module: cytepool.cube.fingerprints
   :value: 1

   The version of the encoding that this module documents and computes.
```

```{eval-rst}
.. automodule:: cytepool.cube.provenance
```
