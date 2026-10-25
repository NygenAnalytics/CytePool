---
myst:
  html_meta:
    description: Voom precision weights from the mean-variance trend of mixed model fits, and empirical Bayes moderated Satterthwaite t-tests of their fixed effects with topTable's z-statistic, as variancePartition's dream computes them, in CytePool.
---

# dream

{func}`~cytepool.stats.dream.voom_with_dream_weights` gives voom's precision weights
from mixed models of every gene, fitted by maximum likelihood, as
variancePartition's `voomWithDreamWeights` does.
{func}`~cytepool.stats.dream.moderated_fixed_effect_tests` moderates the
Satterthwaite t-tests of {func}`~cytepool.stats.mixed_models.fixed_effect_tests` by
empirical Bayes, as variancePartition's `eBayes` does for `dream` fits, and adds
`topTable`'s z-statistic. {func}`~cytepool.differential_expression.dream_voom` runs
these steps on every cell type of a cube.

```{eval-rst}
.. automodule:: cytepool.stats.dream
```
