---
myst:
  html_meta:
    description: Variance partitioning in CytePool, the fraction of every gene's variance that each variable of a design explains, as variancePartition's fitExtractVarPartModel computes it from linear mixed models fitted by maximum likelihood with precision weights, for every cell type of a pseudobulk cube or a matrix of expression values.
---

# Variance partitioning

{func}`~cytepool.variance_partitioning` partitions the variance of every gene among the
variables of a {class}`~cytepool.Design` and the residuals, in every cell type of a
cube, as variancePartition's `fitExtractVarPartModel` does after the steps of
{func}`~cytepool.dream_voom` up to the precision weights: each gene is fitted with a
linear mixed model, by maximum likelihood, whose random intercepts are the categorical
variables (donor, condition, batch) and whose fixed effects are the continuous ones
(age), with voom's precision weights, and the fraction of a variable is its variance
component, or the variance of its fitted effect, over their sum. It records the samples
each fit used. {func}`~cytepool.stats.variance_fractions.variance_fractions` does it
for a matrix of expression values. {func}`~cytepool.plotting.plot_variance_fractions`
and {func}`~cytepool.plotting.plot_feature_variance_fractions` draw the fractions as
variancePartition's `plotVarPart` and `plotPercentBars` do.

```{eval-rst}
.. automodule:: cytepool.variance_components
   :no-members:
   :no-index:

.. autofunction:: cytepool.variance_partitioning

.. autoclass:: cytepool.VariancePartitioningResult
```

```{eval-rst}
.. automodule:: cytepool.stats.variance_fractions
```
