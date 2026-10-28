---
myst:
  html_meta:
    description: edgeR's quasi-likelihood F-tests in CytePool, as glmQLFit and glmQLFTest compute them by default in edgeR 4.4, with bias-adjusted deviances, the common dispersion of the top genes, average log CPM and exact moments of the negative binomial unit deviance.
---

# Quasi-likelihood F-tests

edgeR's quasi-likelihood F-tests, as `glmQLFit` and `glmQLFTest` run them by default in
edgeR 4.4; [edgeR and DESeq2](../statistics/count-models/#quasi-likelihood) explains
them.

```python
from cytepool import Design
from cytepool.design import make_contrasts
from cytepool.normalization import tmm_factors
from cytepool.stats.quasi_likelihood import (
    fit_quasi_likelihood,
    quasi_likelihood_f_test,
)

design = Design(samples, "~ 0 + group")
fit = fit_quasi_likelihood(counts, design, norm_factors=tmm_factors(counts))
contrasts = make_contrasts({"b_vs_a": "groupB - groupA"}, design)
test = quasi_likelihood_f_test(fit, contrasts=contrasts)
test.log2_fold_change, test.p_value, test.adjusted_p_value
```

```{eval-rst}
.. automodule:: cytepool.stats.quasi_likelihood
```
