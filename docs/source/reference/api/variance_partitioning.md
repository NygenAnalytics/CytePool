---
myst:
  html_meta:
    description: Variance partitioning in CytePool, the fraction of every gene's variance that each variable of a design explains, as variancePartition's fitExtractVarPartModel computes it from linear mixed models fitted by maximum likelihood with precision weights, for a matrix of expression values or every cell type of a pseudobulk cube.
---

# Variance partitioning

{func}`~cytepool.variance_partitioning.variance_fractions` partitions the variance of
every gene among the variables of a {class}`~cytepool.design.Design` and the
residuals, as variancePartition's `fitExtractVarPartModel` does: each gene is fitted
with a linear mixed model, by maximum likelihood, whose random intercepts are the
categorical variables (donor, condition, batch) and whose fixed effects are the
continuous ones (age), with precision weights such as voom's, and the fraction of a
variable is its variance component, or the variance of its fitted effect, over their
sum. {func}`~cytepool.variance_partitioning.variance_partitioning` does it for every
cell type of a cube, after the steps of
{func}`~cytepool.differential_expression.dream_voom` up to the precision weights, and
records the samples each fit used. {func}`~cytepool.plotting.plot_variance_fractions`
and {func}`~cytepool.plotting.plot_gene_variance_fractions` draw the fractions as
variancePartition's `plotVarPart` and `plotPercentBars` do.

```{eval-rst}
.. automodule:: cytepool.variance_partitioning
```
