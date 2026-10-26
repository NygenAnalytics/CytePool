---
myst:
  html_meta:
    description: Scores of gene sets in every sample in CytePool, as the R package GSVA computes them, by GSVA, ssGSEA, combined z-scores and PLAGE, from an expression matrix or a cell type of a cube, ready for linear models and drawn as a heatmap.
---

# Pathway scores

{func}`~cytepool.pathway_scores.gsva`, {func}`~cytepool.pathway_scores.ssgsea`,
{func}`~cytepool.pathway_scores.combined_z_scores` and
{func}`~cytepool.pathway_scores.plage` turn the expression values of every sample into
one score per gene set, as the four methods of the R package GSVA compute them, from a
samples x genes matrix or a cell type of a cube. The scores, a
{class}`~cytepool.pathway_scores.PathwayScores`, are samples x sets, so that
{func}`~cytepool.stats.linear_models.fit_linear_model` tests them with a design built
from their samples, and {func}`~cytepool.plotting.plot_pathway_scores` draws them as a
heatmap.

```{eval-rst}
.. automodule:: cytepool.pathway_scores
```
