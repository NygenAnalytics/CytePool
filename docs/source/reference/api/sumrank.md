---
myst:
  html_meta:
    description: SumRank in CytePool, the meta-analysis of Nakatsuka et al. (2025) that ranks every gene's differential expression within each study and sums its best ranks across studies, with Irwin-Hall p-values and p-values from permutations that re-run each study's analysis, and its Manhattan plot.
---

# SumRank

{func}`~cytepool.sumrank` ranks the genes of every study by a signed statistic, sums
each gene's best normalized ranks across the studies and caps the sum at its mean, for
genes up and genes down, as SumRank does (Nakatsuka et al. 2025), for one cell type
and contrast of the studies' results, a {class}`~cytepool.StudyResults`, with p-values
from the Irwin-Hall distribution or from permutations: it permutes the variable that
the contrast compares among the samples of each study's fit, re-runs the study's
analysis on its aligned cube, and pools the null distribution.
{func}`~cytepool.stats.sumrank.fit_sumrank` does it for a matrix of statistics, with
the statistics of permuted studies. {func}`~cytepool.plotting.plot_sumrank` draws the
p-value of every gene, the genes in alphabetical order, as the authors of SumRank draw
it.

```{eval-rst}
.. automodule:: cytepool.sumrank_across_studies
   :no-members:
   :no-index:

.. autofunction:: cytepool.sumrank

.. autoclass:: cytepool.SumRankResult
```

```{eval-rst}
.. automodule:: cytepool.stats.sumrank
```
