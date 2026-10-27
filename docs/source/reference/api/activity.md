---
myst:
  html_meta:
    description: Activities of transcription factors and pathways in CytePool, as decoupler computes them, by univariate and multivariate linear models (ULM, MLM) and the weighted aggregate, from an expression matrix, a cell type of a cube or the statistics of a differential-expression result.
---

# Pathway and regulator activity

{func}`~cytepool.ulm`, {func}`~cytepool.mlm` and {func}`~cytepool.weighted_aggregate`
score every source of a network, a transcription factor or a pathway, in every
profile: the samples of an expression matrix or of a cell type of a cube, or the
contrasts of a differential-expression result, as decoupler's `mt.ulm`, `mt.mlm` and
`mt.waggr` compute them. The activities, an {class}`~cytepool.Activities`, are profiles
x sources, so that {func}`~cytepool.limma` tests the activities of samples with the
design built for the cube; {func}`~cytepool.plotting.plot_activities` draws them as a
heatmap and {func}`~cytepool.plotting.plot_top_activities` the top sources of a profile
as bars. The networks are {class}`~cytepool.Network` objects, such as those of
{func}`~cytepool.collectri` and {func}`~cytepool.progeny`.

```{eval-rst}
.. automodule:: cytepool.activity
   :no-members:
   :no-index:

.. autofunction:: cytepool.ulm

.. autofunction:: cytepool.mlm

.. autofunction:: cytepool.weighted_aggregate

.. autoclass:: cytepool.Activities
```
