---
myst:
  html_meta:
    description: The permutation check of CytePool, which permutes the dose labels of a dose series among the samples that its design treats alike, runs the analysis again and counts the genes that its gate passes, to show how often noise passes the gate at the series' own design.
---

# Permutation checks

How often the gate of a dose series passes genes when the doses carry no information, at the series' own design; [Could the calls come from noise?](../statistics/dose-response/#dose-permutation-check) explains the check.

```{eval-rst}
.. automodule:: cytepool.permutation_checks
   :no-members:
   :no-index:

.. autofunction:: cytepool.permutation_check

.. autoclass:: cytepool.PermutationCheck
```
