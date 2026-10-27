---
myst:
  html_meta:
    description: Voom precision weights from the mean-variance trend of mixed model fits, and empirical Bayes moderated Satterthwaite or Kenward-Roger t-tests and F-tests of their fixed effects with topTable's z-statistic, as variancePartition's dream computes them, in CytePool.
---

# dream

{func}`~cytepool.stats.dream.voom_with_dream_weights` gives voom's precision weights
from mixed models of every gene, fitted by maximum likelihood, as
variancePartition's `voomWithDreamWeights` does.
{func}`~cytepool.stats.dream.moderated_fixed_effect_tests` moderates the
Satterthwaite or Kenward-Roger t-tests of
{func}`~cytepool.stats.mixed_models.fixed_effect_tests` by empirical Bayes, as
variancePartition's `eBayes` does for `dream` fits, adds `topTable`'s z-statistic, and
tests the contrasts at once by the moderated F-statistic.
{func}`~cytepool.dream_voom` runs these steps on every cell type of a cube, with
dream's choice of the degrees of freedom.

```{eval-rst}
.. automodule:: cytepool.stats.dream
```
