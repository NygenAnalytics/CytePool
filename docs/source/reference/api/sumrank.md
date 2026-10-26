---
myst:
  html_meta:
    description: SumRank in CytePool, the meta-analysis of Nakatsuka et al. (2025) that ranks every gene's differential expression within each study and sums its best ranks across studies, with Irwin-Hall p-values and p-values from permutations that re-run each study's analysis, and its Manhattan plot.
---

# SumRank

{func}`~cytepool.sumrank.sumrank` ranks the genes of every study by a signed statistic,
sums each gene's best normalized ranks across the studies and caps the sum at its mean,
for genes up and genes down, as SumRank does (Nakatsuka et al. 2025), with p-values from
the Irwin-Hall distribution and, given the statistics of permuted studies, from their
pooled null distribution. {func}`~cytepool.sumrank.differential_expression_sumrank`
does it for one cell type and contrast of the studies of a
{class}`~cytepool.cube.CubeCollection`, from each study's differential-expression
result, and runs the permutations itself: it permutes a variable of each study's design
among the samples of its fit, and re-runs the study's analysis on its aligned cube.
{func}`~cytepool.plotting.plot_sumrank` draws the p-value of every gene, the genes in
alphabetical order, as the authors of SumRank draw it.

```{eval-rst}
.. automodule:: cytepool.sumrank
```
