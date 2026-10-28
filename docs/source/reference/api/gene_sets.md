---
myst:
  html_meta:
    description: Gene-set collections in CytePool, read from and written to the GMT files of MSigDB, Reactome and WikiPathways and the GMT exports of the Gene Ontology, with each collection's name, version and licence, matched to the genes of an analysis and filtered by size.
---

# Gene-set files

Gene-set collections, read from and written to GMT files; [Gene-set
tests](../statistics/gene-set-tests/#gene-set-collections) explains how they are
matched to an analysis.

```{eval-rst}
.. automodule:: cytepool.gene_sets
   :no-members:
   :no-index:

.. autoclass:: cytepool.GeneSets

.. autofunction:: cytepool.read_gmt

.. autofunction:: cytepool.write_gmt
```
