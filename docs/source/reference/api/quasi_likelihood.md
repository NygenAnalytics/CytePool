---
myst:
  html_meta:
    description: edgeR's quasi-likelihood F-tests in CytePool, as glmQLFit and glmQLFTest compute them by default in edgeR 4, with bias-adjusted deviances, the common dispersion of the top genes, average log CPM and exact moments of the negative binomial unit deviance.
---

# Quasi-likelihood F-tests

{func}`~cytepool.stats.quasi_likelihood.fit_quasi_likelihood` and
{func}`~cytepool.stats.quasi_likelihood.quasi_likelihood_f_test` are edgeR's
`glmQLFit` and `glmQLFTest` as edgeR 4 runs them by default: the negative binomial
dispersion is the Cox-Reid common dispersion of the most highly expressed genes
({func}`~cytepool.stats.quasi_likelihood.common_dispersion`), each gene's
quasi-dispersion comes from deviances adjusted for small counts, and limma's prior for
unequal degrees of freedom moderates them, towards a trend in the average log CPM
({func}`~cytepool.stats.quasi_likelihood.average_log_cpm`). The tests give edgeR's
columns logFC, logCPM, F, PValue and FDR for coefficients or named contrasts of a
design.

```python
from cytepool.design import Design, make_contrasts
from cytepool.normalization import tmm_factors
from cytepool.stats.quasi_likelihood import (
    fit_quasi_likelihood,
    quasi_likelihood_f_test,
)

design = Design(samples, "~ 0 + group")
fit = fit_quasi_likelihood(counts, design, norm_factors=tmm_factors(counts))
contrasts = make_contrasts({"b_vs_a": "groupB - groupA"}, design)
test = quasi_likelihood_f_test(fit, contrasts=contrasts)
test.log_fold_change, test.p_value, test.adjusted_p_value
```

```{eval-rst}
.. automodule:: cytepool.stats.quasi_likelihood
```
