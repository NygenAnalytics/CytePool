---
myst:
  html_meta:
    description: Build pseudobulk cubes from cell-level or bulk counts with CytePool, keep sample, group and feature metadata in tables, record what each cube pooled with a provenance record and a content fingerprint, and align the cubes of several studies by sample, gene and cell type.
---

# Pseudobulk cubes and tables

Cubes, their tables, and the collections of the cubes of several studies;
[Pseudobulk profiles and cubes](../statistics/cubes/) explains cubes, and [Aligning
studies](../statistics/aligning-studies/) their collections.

```{eval-rst}
.. autoclass:: cytepool.Cube

.. automodule:: cytepool.cube.pool
   :no-members:
   :no-index:

.. autofunction:: cytepool.from_cells

.. autofunction:: cytepool.from_bulk

.. automodule:: cytepool.cube.collection
   :no-members:
   :no-index:

.. autoclass:: cytepool.CubeCollection
```

```{eval-rst}
.. automodule:: cytepool.table
   :no-members:
   :no-index:

.. autoclass:: cytepool.Table
```

## Fingerprints and provenance

The fingerprint that names a cube's values and the record of how it was made; [Cube
files and provenance records](../reference/cube-files/#fingerprint-encoding) specifies
both.

```{eval-rst}
.. automodule:: cytepool.cube.fingerprints
   :exclude-members: FINGERPRINT_VERSION
```

```{eval-rst}
.. py:data:: FINGERPRINT_VERSION
   :module: cytepool.cube.fingerprints
   :value: 1

   The version of the encoding of fingerprints that this module computes,
   https://docs.nygen.io/CytePool/reference/cube-files#fingerprint-encoding.
```

```{eval-rst}
.. automodule:: cytepool.cube.provenance
```
