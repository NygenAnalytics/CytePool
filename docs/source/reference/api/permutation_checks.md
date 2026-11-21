---
myst:
  html_meta:
    description: The permutation check of CytePool, which permutes the dose labels of a dose series among the samples that its design treats alike, runs the analysis again and counts the genes that its gate passes, to test whether the true doses pass more genes than permuted ones.
---

# Permutation checks

Whether the true doses of a dose series pass more genes through its gate than permuted doses do, and how often the gate passes genes with the doses permuted; [Could the calls come from noise?](../statistics/dose-response/#dose-permutation-check) explains the check.

```{eval-rst}
.. automodule:: cytepool.permutation_checks
   :no-members:
   :no-index:

.. autofunction:: cytepool.permutation_check

.. autoclass:: cytepool.PermutationCheck
```
