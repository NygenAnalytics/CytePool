---
myst:
  html_meta:
    description: Gene set enrichment analysis (GSEA) in CytePool, as fgsea computes it, of genes ranked by a statistic of a differential-expression result, a table of results or a mapping of gene id to statistic, with p-values by fgsea's multilevel method or by permutations, and its enrichment plot and table.
---

# Gene set enrichment analysis

{func}`~cytepool.gsea` ranks genes by a statistic, of a group and contrast of a
differential-expression result, of a table of results or of a mapping of gene id to
statistic, and asks of each gene set whether its genes gather at the top or at the
bottom of the ranking, as fgsea's `fgsea` computes it: the enrichment score and its
leading edge, the score normalized by those of random sets of the same size, and
p-values by fgsea's multilevel method or by permutations of the genes.
{func}`~cytepool.plotting.plot_enrichment` draws the running enrichment score of one
set, and {func}`~cytepool.plotting.plot_gsea_table` the top sets with their genes on
the ranking.

```{eval-rst}
.. automodule:: cytepool.gene_set_enrichment
   :no-members:
   :no-index:

.. autofunction:: cytepool.gsea
```
