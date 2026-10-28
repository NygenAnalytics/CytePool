---
myst:
  html_meta:
    description: Differential expression of every cell type of a pseudobulk cube by limma-voom, limma-trend and dream in CytePool, with options for pseudobulk data, the samples of each fit and a provenance record.
---

# Differential expression

{func}`~cytepool.limma_voom`, {func}`~cytepool.limma_trend` and {func}`~cytepool.dream_voom` test the contrasts of a design in every cell type of a cube; [limma-voom and limma-trend](../statistics/limma-voom/) and [Mixed models and dream](../statistics/mixed-models/) explain them.

```{eval-rst}
.. automodule:: cytepool.differential_expression
   :no-members:
   :no-index:

.. autofunction:: cytepool.limma_voom

.. autofunction:: cytepool.limma_trend

.. autofunction:: cytepool.dream_voom

.. autoclass:: cytepool.DifferentialExpressionResult

.. autoclass:: cytepool.GroupFit
```
