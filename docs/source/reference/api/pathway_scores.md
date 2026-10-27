---
myst:
  html_meta:
    description: Scores of gene sets in every sample in CytePool, as the R package GSVA computes them, by GSVA, ssGSEA, combined z-scores and PLAGE, from an expression matrix or a cell type of a cube, ready for linear models and drawn as a heatmap.
---

# Pathway scores

{func}`~cytepool.gsva`, {func}`~cytepool.ssgsea`, {func}`~cytepool.combined_z_scores`
and {func}`~cytepool.plage` turn the expression values of every sample into one score
per gene set, as the four methods of the R package GSVA compute them, from a samples x
genes matrix or a cell type of a cube. The scores, a {class}`~cytepool.PathwayScores`,
are profiles x sets, so that {func}`~cytepool.limma` tests them with the design built
for the cube, and {func}`~cytepool.plotting.plot_pathway_scores` draws them as a
heatmap.

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
