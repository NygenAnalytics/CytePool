---
myst:
  html_meta:
    description: Rotation gene-set tests of one contrast in CytePool, limma's self-contained roast and mroast and its competitive, rank-based romer, of a cell type of a differential-expression result or of values with a design, with seeded rotations.
---

# Rotation gene-set tests

{func}`~cytepool.roast`, {func}`~cytepool.mroast` and {func}`~cytepool.romer` test gene
sets by rotations of the residuals on one contrast of a linear model: of a group of a
result of {func}`~cytepool.limma_voom` or {func}`~cytepool.limma_trend`, whose fit they
compute again, or of values with a design and a contrast. romer takes sample weights
only, so of the two results it tests those of limma-trend.

```{eval-rst}
.. automodule:: cytepool.rotation_gene_set_tests
   :no-members:
   :no-index:

.. autofunction:: cytepool.roast

.. autofunction:: cytepool.mroast

.. autofunction:: cytepool.romer
```
