---
myst:
  html_meta:
    description: Differential expression of every cell type of a pseudobulk cube by limma-voom, limma-trend and dream in CytePool, with options for pseudobulk data, the samples behind each fit and a provenance record.
---

# Differential expression

{func}`~cytepool.differential_expression.limma_voom` and
{func}`~cytepool.differential_expression.limma_trend` test the contrasts of a
{class}`~cytepool.design.Design` in every group (cell type) of a cube, one group at a
time, as the limma and edgeR User's Guides do for RNA-seq counts, and
{func}`~cytepool.differential_expression.dream_voom` does so with the random
intercepts of the design, such as a donor's, as variancePartition's dream does. The design is
matched to the cube by `sample_id`; samples with too few cells are left out of a
group's fit, and a group whose design becomes rank deficient is fitted on its column
space, with the contrasts its samples can estimate. Options for pseudobulk data, each
off by default, add sample quality weights, prior weights from the number of cells, a
minimum total count per sample, and p-values adjusted across groups. The result holds
a table per group and contrast, a summary of every group, the samples used and left
out with the weights of each fit, and a record that traces every number to the
profiles of the cube that produced it.

```{eval-rst}
.. automodule:: cytepool.differential_expression
```
