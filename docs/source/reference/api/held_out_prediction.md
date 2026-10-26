---
myst:
  html_meta:
    description: Held-out prediction in CytePool, as Nakatsuka et al. (2025) measure how well genes found in some studies replicate, each study left out in turn, genes chosen from the others by SumRank and by a random-effects meta-analysis, its samples scored by UCell's rank score, and the AUC of cases against controls reported by study and method.
---

# Held-out prediction

{func}`~cytepool.held_out_prediction.held_out_prediction` leaves each study of a
{class}`~cytepool.cube.CubeCollection` out in turn, chooses genes up and down in each
cell type from the results of the other studies, by SumRank and by a random-effects
meta-analysis, scores every case and control of the held-out study by how high its
profiles rank those genes, as UCell scores them, and gives the area under the ROC curve
of cases against controls, by study and method, as Nakatsuka et al. (2025) compare
methods. {func}`~cytepool.plotting.plot_held_out_auc` draws the AUC of every held-out
study by method.

```{eval-rst}
.. automodule:: cytepool.held_out_prediction
```
