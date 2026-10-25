---
myst:
  html_meta:
    description: DESeq2's gene-wise, trended and maximum a posteriori dispersions, its negative binomial fits of every gene, Wald tests of coefficients and contrasts and likelihood ratio tests of reduced designs, with the independent filtering and Cook's distances of results(), in CytePool.
---

# DESeq2

{func}`~cytepool.stats.deseq2.fit_deseq2` estimates DESeq2's dispersions and fits every
gene as `DESeq()` does; {func}`~cytepool.stats.deseq2.deseq2_wald_test` tests a
coefficient or a contrast and {func}`~cytepool.stats.deseq2.deseq2_likelihood_ratio_test`
a reduced design against the full one, such as all the levels of a factor at once, each
with the columns of `results()`. {func}`~cytepool.stats.deseq2.deseq2_dispersions` gives
the dispersion estimates of every stage, and takes given values for each stage, as
DESeq2's lower-level functions do.

```{eval-rst}
.. automodule:: cytepool.stats.deseq2
```
