---
myst:
  html_meta:
    description: Principal components of samples in CytePool, as DESeq2's plotPCA computes them from the genes of largest variance, for a matrix of log-expression values or a cell type of a pseudobulk cube, with their coordinates, shares of variance and loadings, and their figure.
---

# Principal components

{func}`~cytepool.principal_components` places samples on the principal components of
their most variable genes, as DESeq2's `plotPCA` does: the `top` genes of largest
variance over the samples, each centred, decomposed by singular values. It takes a
samples x genes matrix of log-expression values or a cell type of a cube, whose samples
are prepared as edgeR prepares counts for `plotMDS`. The result, a
{class}`~cytepool.PrincipalComponents`, holds the samples' coordinates, each
component's share of the variance, the features used and their loadings, and
{func}`~cytepool.plotting.plot_pca` draws it.

```{eval-rst}
.. automodule:: cytepool.principal_component_analysis
   :no-members:
   :no-index:

.. autofunction:: cytepool.principal_components

.. autoclass:: cytepool.PrincipalComponents
```
