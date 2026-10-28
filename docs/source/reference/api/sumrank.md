---
myst:
  html_meta:
    description: SumRank in CytePool, the meta-analysis of Nakatsuka et al. (2025) that ranks every gene's differential expression within each study and sums its best ranks across studies, with Irwin-Hall p-values and p-values from permutations that re-run each study's analysis, and its Manhattan plot.
---

# SumRank

SumRank, every gene's best ranks summed across studies, with Irwin-Hall and permutation
p-values; [Replication](../statistics/replication/#sum-of-ranks) explains it.

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
