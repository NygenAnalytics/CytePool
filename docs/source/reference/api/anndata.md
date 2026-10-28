---
myst:
  html_meta:
    description: Convert CytePool cubes to and from AnnData and MuData, the pseudobulk layouts of scanpy, decoupler and LIANA, keeping every value, table and the provenance, with anndata and mudata as optional dependencies.
---

# AnnData and MuData

Conversions of cubes to and from the AnnData and MuData layouts of scverse;
[Pseudobulk profiles and cubes](../statistics/cubes/#profiles-from-other-tools)
explains what an import takes.

```{eval-rst}
.. automodule:: cytepool.io.anndata
   :no-members:
   :no-index:

.. autofunction:: cytepool.to_anndata

.. autofunction:: cytepool.from_anndata

.. autofunction:: cytepool.to_mudata

.. autofunction:: cytepool.from_mudata
```
