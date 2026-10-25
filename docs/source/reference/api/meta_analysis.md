---
myst:
  html_meta:
    description: Meta-analysis of the effects of many genes across studies in CytePool, by fixed and random effects with the between-study variance by DerSimonian-Laird, REML or Paule-Mandel, Cochran's Q, I-squared, H-squared and the Knapp-Hartung adjustment, as metafor computes them, and of the differential expression of several cubes.
---

# Meta-analysis

{func}`~cytepool.meta_analysis.meta_analysis` combines, for every gene, the effects of
several studies and their standard errors into an estimate of a common effect, as
metafor's `rma.uni` does without moderators: the fixed-effect (inverse-variance)
estimate, or a random-effects estimate with the between-study variance by
DerSimonian-Laird, REML or Paule-Mandel, with Cochran's Q, I-squared, H-squared and
z-tests or the Knapp-Hartung adjustment. A gene measured in some studies only is
analysed over those.
{func}`~cytepool.meta_analysis.differential_expression_meta_analysis` does it for the
results of {func}`~cytepool.differential_expression.limma_voom` or
{func}`~cytepool.differential_expression.limma_trend` on several cubes, such as the
same cell types in several cohorts: it matches groups, contrasts and genes by their ids,
reads each table's log2 fold changes and moderated standard errors, and returns a table
per cell type and contrast with p-values adjusted over its genes, the studies used and
left out with the reason, and the provenance of every study.

```{eval-rst}
.. automodule:: cytepool.meta_analysis
```
