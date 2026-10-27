---
myst:
  html_meta:
    description: Meta-analysis of the differential expression of several studies in CytePool, gene by gene, by fixed and random effects with the between-study variance by DerSimonian-Laird, REML or Paule-Mandel, Cochran's Q, I-squared, H-squared and the Knapp-Hartung adjustment, as metafor computes them, from the studies' results or from arrays of effects and standard errors.
---

# Meta-analysis

{func}`~cytepool.meta_analysis` combines, for every gene, the log2 fold changes of
several studies and their standard errors into an estimate of a common effect, as
metafor's `rma.uni` does without moderators: the fixed-effect (inverse-variance)
estimate, or a random-effects estimate with the between-study variance by
DerSimonian-Laird, REML or Paule-Mandel, with Cochran's Q, I-squared, H-squared and
z-tests or the Knapp-Hartung adjustment. It takes the results of
{func}`~cytepool.limma_voom` or {func}`~cytepool.limma_trend` on several cubes, such as
the same cell types in several cohorts, kept together as a
{class}`~cytepool.StudyResults`: it matches groups, contrasts and genes by their ids,
reads each table's log2 fold changes and moderated standard errors, and returns a table
per cell type and contrast with p-values adjusted over its genes, the studies used and
left out with the reason, and the provenance of every study. A gene measured in some
studies only is analysed over those. {meth}`StudyResults.table
<cytepool.StudyResults.table>` lines up the studies' results gene by gene, with the
reason a study has no result for a gene, and
{func}`~cytepool.stats.meta_analysis.fit_meta_analysis` meta-analyses arrays of effects
and standard errors.

```{eval-rst}
.. automodule:: cytepool.meta_analysis_of_studies
   :no-members:
   :no-index:

.. autofunction:: cytepool.meta_analysis

.. autoclass:: cytepool.MetaAnalysisResult
```

```{eval-rst}
.. automodule:: cytepool.stats.meta_analysis
```
