---
myst:
  html_meta:
    description: Convert CytePool cubes to and from AnnData and MuData, the pseudobulk layouts of scanpy, decoupler and LIANA, keeping every value, table and the provenance, with anndata and mudata as optional dependencies.
---

# AnnData and MuData

scverse keeps pseudobulk profiles in two layouts, and CytePool reads and writes both.
{func}`~cytepool.to_anndata` and {func}`~cytepool.from_anndata` convert a cube to and
from a long AnnData, with a row per sample and cell type, as `scanpy.get.aggregate`
and `decoupler.pp.pseudobulk` make it; {func}`~cytepool.to_mudata` and
{func}`~cytepool.from_mudata` convert it to and from a MuData with a modality per cell
type and the samples as its rows, as LIANA's `adata_to_views` makes it. An import
takes only sums and counts of cells, so a cube made from aggregated profiles holds
what one made from the cells would hold, and a cube exported and read back has the
same fingerprint. anndata and mudata are optional: install CytePool's `anndata` or
`mudata` extra.

```{eval-rst}
.. automodule:: cytepool.io.anndata
   :no-members:
   :no-index:

.. autofunction:: cytepool.to_anndata

.. autofunction:: cytepool.from_anndata

.. autofunction:: cytepool.to_mudata

.. autofunction:: cytepool.from_mudata
```
