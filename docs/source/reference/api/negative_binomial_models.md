---
myst:
  html_meta:
    description: Negative binomial generalized linear models fitted to every gene, with shrunk log fold changes and exact deviances, as edgeR's glmFit, nbinomDeviance and nbinomUnitDeviance compute them, in CytePool.
---

# Negative binomial models

{func}`~cytepool.stats.negative_binomial_models.fit_negative_binomial_model` fits the
same negative binomial log-linear model to every gene, with known dispersions, as
edgeR's `glmFit` does: one-way layouts group by group, other designs by Newton's method
with Levenberg damping. It is the first step of edgeR's quasi-likelihood pipeline;
dispersion estimation and the tests come next.

```{eval-rst}
.. automodule:: cytepool.stats.negative_binomial_models
```
