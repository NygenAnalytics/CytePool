---
myst:
  html_meta:
    description: How well the studies of a CubeCollection replicate each other's differential expression in CytePool, gene by gene and study by study, with combined and partial conjunction p-values, sign agreement, heterogeneity, pi1 between studies and the genes found in at least k studies.
---

# Replication across studies

{func}`~cytepool.replication.differential_expression_replication` reads one cell type
and contrast of the studies of a {class}`~cytepool.cube.CubeCollection`, lined up by
{func}`~cytepool.meta_analysis.align_study_results`, and gives for every gene its
combined p-value, its partial conjunction p-value of an effect in at least u studies in
one direction, the studies that find it and the heterogeneity of its log2 fold changes;
for every pair of studies, pi1 of one study's p-values among the genes the other finds,
and their sign agreement; and the number of genes found in at least k studies.
{func}`~cytepool.plotting.plot_study_agreement` and
{func}`~cytepool.plotting.plot_replicated_genes` draw them.

```{eval-rst}
.. automodule:: cytepool.replication
```
