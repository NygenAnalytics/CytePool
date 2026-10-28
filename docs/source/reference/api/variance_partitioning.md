---
myst:
  html_meta:
    description: Variance partitioning in CytePool, the fraction of every gene's variance that each variable of a design explains, as variancePartition's fitExtractVarPartModel computes it from linear mixed models fitted by maximum likelihood with precision weights, for every cell type of a pseudobulk cube or a matrix of expression values.
---

# Variance partitioning

The share of each gene's variance that each variable of a design explains, in every cell type of a cube or in a matrix; [Mixed models and dream](../statistics/mixed-models/#variance-fractions) explains it.

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
