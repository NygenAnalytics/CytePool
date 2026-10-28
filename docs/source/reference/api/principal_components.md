---
myst:
  html_meta:
    description: Principal components of samples in CytePool, as DESeq2's plotPCA computes them from the genes of largest variance, for a matrix of log-expression values or a cell type of a pseudobulk cube, with their coordinates, shares of variance and loadings, and their figure.
---

# Principal components

Principal components of samples, as DESeq2's `plotPCA` computes them; [Normalization and
filtering](../statistics/normalization-and-filtering/#samples-at-a-glance) explains them.

```{eval-rst}
.. automodule:: cytepool.principal_component_analysis
   :no-members:
   :no-index:

.. autofunction:: cytepool.principal_components

.. autoclass:: cytepool.PrincipalComponents
```
