---
myst:
  html_meta:
    description: Activities of transcription factors and pathways in CytePool, as decoupler computes them, by univariate and multivariate linear models (ULM, MLM) and the weighted aggregate, from an expression matrix, a cell type of a cube or the statistics of a differential-expression result.
---

# Pathway and regulator activity

{func}`~cytepool.activity.ulm`, {func}`~cytepool.activity.mlm` and
{func}`~cytepool.activity.weighted_aggregate` score every source of a network, a
transcription factor or a pathway, in every profile: the samples of an expression
matrix or of a cell type of a cube, or the contrasts of a differential-expression
result, as decoupler's `mt.ulm`, `mt.mlm` and `mt.waggr` compute them. The activities,
an {class}`~cytepool.activity.Activities`, are profiles x sources, so that
{func}`~cytepool.stats.linear_models.fit_linear_model` tests them with a design built
from their profiles; {func}`~cytepool.plotting.plot_activities` draws them as a
heatmap and {func}`~cytepool.plotting.plot_top_activities` the top sources of a
profile as bars. The networks come from {mod}`cytepool.networks`.

```{eval-rst}
.. automodule:: cytepool.activity
```
