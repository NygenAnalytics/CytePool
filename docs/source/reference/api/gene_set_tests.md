---
myst:
  html_meta:
    description: Gene-set tests of one contrast in CytePool, limma's competitive test camera and its self-contained test fry, of a cell type of a differential-expression result or of values with a design, with voom weights.
---

# Gene-set tests

{func}`~cytepool.camera` and {func}`~cytepool.fry` test gene sets on one contrast of a
linear model: of a group of a result of {func}`~cytepool.limma_voom` or
{func}`~cytepool.limma_trend`, whose fit they compute again, or of values with a design
and a contrast.

```{eval-rst}
.. automodule:: cytepool.gene_set_tests
   :no-members:
   :no-index:

.. autofunction:: cytepool.camera

.. autofunction:: cytepool.fry
```
