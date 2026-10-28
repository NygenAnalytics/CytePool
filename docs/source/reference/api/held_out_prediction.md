---
myst:
  html_meta:
    description: Held-out prediction in CytePool, as Nakatsuka et al. (2025) measure how well genes found in some studies replicate, each study left out in turn, genes chosen from the others by SumRank and by a random-effects meta-analysis, its samples scored by UCell's rank score, and the AUC of cases against controls reported by study and method.
---

# Held-out prediction

The prediction of cases in held-out studies by genes chosen from the others; [Replication](../statistics/replication/#held-out-prediction) explains it.

```{eval-rst}
.. automodule:: cytepool.held_out_studies
   :no-members:
   :no-index:

.. autofunction:: cytepool.held_out_prediction

.. autoclass:: cytepool.HeldOutPrediction
```
