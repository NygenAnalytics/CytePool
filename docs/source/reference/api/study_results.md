---
myst:
  html_meta:
    description: Analyse every study of a CubeCollection alike in CytePool, each on its aligned cube with a design built from its own samples, and keep the results together, lined up gene by gene or as a studies x genes matrix for combined p-values and SumRank.
---

# Results of several studies

Studies of one question are compared or combined once each is analysed on the
collection's axes: its aligned cube, with a design built from its own samples.
{func}`~cytepool.study_results.analyse_each_study` runs
{func}`~cytepool.differential_expression.limma_voom`,
{func}`~cytepool.differential_expression.limma_trend` or
{func}`~cytepool.differential_expression.dream_voom` on every study of a
{class}`~cytepool.cube.CubeCollection` that way, with one formula, its levels and
contrasts. {class}`~cytepool.study_results.StudyResults` keeps the results with their
collection: it lines them up gene by gene, with the reason a study has no value for a
gene, and gives one column of every study, such as the p-values, as a studies x genes
matrix, the form that combined p-values and SumRank take.

```{eval-rst}
.. automodule:: cytepool.study_results
```
