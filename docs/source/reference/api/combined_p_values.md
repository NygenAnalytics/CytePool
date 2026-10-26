---
myst:
  html_meta:
    description: Combined p-values of several studies by the methods of Fisher, Stouffer (weighted), Tippett, Wilkinson, Pearson, Mudholkar and George and Simes, as scipy and metap compute them, one-sided or two-sided with the directions of the effects, and partial conjunction tests that an effect is found in at least u studies.
---

# Combined p-values

{func}`~cytepool.stats.combined_p_values.combine_p_values` combines, for every gene,
the p-values of the studies that tested it into one p-value of the hypothesis that no
study has an effect: by Fisher's, Stouffer's (with weights, such as the square roots of
the studies' sample sizes), Tippett's, Wilkinson's, Pearson's, Mudholkar and George's
or Simes's method, as scipy's `combine_pvalues` and the R package metap compute them.
With the directions of the effects, two-sided p-values become one-sided ones, and
effects in opposite directions count against each other.
{func}`~cytepool.stats.combined_p_values.partial_conjunction` gives the p-value that at
least u of the studies have an effect, by combining the largest p-values (Benjamini and
Heller 2008). A study that did not test a gene is left out of its combination.

```{eval-rst}
.. automodule:: cytepool.stats.combined_p_values
```
