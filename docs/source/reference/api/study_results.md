---
myst:
  html_meta:
    description: Analyse every study of a CubeCollection alike in CytePool, each on its aligned cube with a design built from its own samples, and keep the results together, lined up gene by gene or as a studies x genes matrix for combined p-values and SumRank.
---

# Results of several studies

Studies of one question are compared or combined once each is analysed on the
collection's axes: its aligned cube, with a design built from its own samples.
{func}`~cytepool.analyse_each_study` runs {func}`~cytepool.limma_voom`,
{func}`~cytepool.limma_trend` or {func}`~cytepool.dream_voom` on every study of a
{class}`~cytepool.CubeCollection` that way, with one formula, its levels and contrasts.
{class}`~cytepool.StudyResults` keeps the results with their collection: it lines them
up gene by gene, with the reason a study has no value for a gene, and gives one column
of every study, such as the p-values, as a studies x genes matrix, the form that
combined p-values and SumRank take. {func}`~cytepool.meta_analysis`,
{func}`~cytepool.replication`, {func}`~cytepool.sumrank` and
{func}`~cytepool.held_out_prediction` take it.

```{eval-rst}
.. automodule:: cytepool.study_results
   :no-members:
   :no-index:

.. autofunction:: cytepool.analyse_each_study

.. autoclass:: cytepool.StudyResults
```
