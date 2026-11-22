---
myst:
  html_meta:
    description: The check of a dose series in CytePool, which reports for every cell type of a cube, or for the scores of one cell type's samples, the samples each dose group keeps after filtering, the dose groups that other terms or a batch absorb, a status of inferential, descriptive or refused, and ETAP's design minimums.
---

# Dose designs

Whether a dose series supports tests in each cell type, checked before anything is fitted; [Checking a dose design](../statistics/dose-response/#dose-design-check) explains it.

```{eval-rst}
.. automodule:: cytepool.dose_designs
   :no-members:
   :no-index:

.. autofunction:: cytepool.check_dose_design

.. autoclass:: cytepool.DoseDesignReport
```
