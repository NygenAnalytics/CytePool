---
myst:
  html_meta:
    description: Test the pathway scores and regulator activities of each sample with a design in CytePool, by limma's linear models and moderated t-tests, in one call with the design built for the cube.
---

# Tests of pathway scores and activities

Pathway scores and the activities of regulators give every sample of a cell type one
value per gene set or source, and differ between conditions as expression values do.
{func}`~cytepool.score_tests.limma` tests a contrast of a
{class}`~cytepool.design.Design` on them as GSVA's and decoupler's guides do with
limma: a linear model of every set's scores, moderated t-tests of the contrast and
Benjamini-Hochberg adjusted p-values, with the design built for the cube matched to
the scored samples by `sample_id`. It takes
{class}`~cytepool.pathway_scores.PathwayScores`,
{class}`~cytepool.activity.Activities` of samples, or an array of values from another
platform.

```{eval-rst}
.. automodule:: cytepool.score_tests
```
