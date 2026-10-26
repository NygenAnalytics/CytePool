---
myst:
  html_meta:
    description: Over-representation analysis in CytePool, the one-sided hypergeometric test of limma's kegga and goana, for the genes that go up, down or either way in a differential-expression result, a table of results or a list of genes.
---

# Over-representation analysis

{func}`~cytepool.over_representation.over_representation` tests each gene set for
over-representation among the genes called significant, by the one-sided
hypergeometric test of limma's `kegga` and `goana`: for the genes that go up, those
that go down and those that change either way in a group and contrast of a
differential-expression result or in a table of results, or for a list of genes
against a universe. Its table gives each set's genes in the universe, the overlap, the
expected overlap, the p-value, the adjusted p-value and the genes of the overlap.
{func}`~cytepool.plotting.plot_over_representation` draws the top sets.

```{eval-rst}
.. automodule:: cytepool.over_representation
```
