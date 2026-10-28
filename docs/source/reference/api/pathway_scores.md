---
myst:
  html_meta:
    description: Scores of gene sets in every sample in CytePool, as the R package GSVA computes them, by GSVA, ssGSEA, combined z-scores and PLAGE, from an expression matrix or a cell type of a cube, ready for linear models and drawn as a heatmap.
---

# Pathway scores

Scores of gene sets in every sample by GSVA, ssGSEA, combined z-scores and PLAGE;
[Pathway scores per sample](../statistics/pathway-scores/#gsva) explains them.

```{eval-rst}
.. automodule:: cytepool.pathway_scores
   :no-members:
   :no-index:

.. autofunction:: cytepool.gsva

.. autofunction:: cytepool.ssgsea

.. autofunction:: cytepool.combined_z_scores

.. autofunction:: cytepool.plage

.. autoclass:: cytepool.PathwayScores
```
