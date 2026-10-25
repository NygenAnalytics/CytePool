---
myst:
  html_meta:
    description: Tests of differences in cell-type composition between conditions in CytePool, by propeller on the cell numbers of a pseudobulk cube.
---

# Cell-type composition

{func}`~cytepool.composition.propeller` tests whether the proportions of cell types
differ between conditions, as speckle's `propeller`, `propeller.ttest` and
`propeller.anova` do. It reads the number of cells of every sample in every group of a
cube (or a table of them), transforms the proportions (logit or arcsine square root),
fits a linear model with a {class}`~cytepool.design.Design` matched to the samples by
`sample_id`, moderates the variances of the cell types by empirical Bayes, and tests
the design's named contrasts by moderated t-statistics and the levels of a factor by
a moderated F-statistic. The result holds speckle's columns for every cell type, the
samples and cell types left out, the proportions, and a record that identifies the
cell numbers by the cube's fingerprint.

```{eval-rst}
.. automodule:: cytepool.composition
```
