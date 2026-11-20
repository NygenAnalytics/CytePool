---
myst:
  html_meta:
    description: The dose tests of CytePool, which test in every cell type of a cube whether each gene responds to a dose, along a spline of dose, in its departure from a line, by its linear trend and by its dose groups, as contrasts of one moderated fit, with curves and 95% bands.
---

# Dose responses

Which genes respond to a dose in each cell type, by moderated tests of one model of the dose groups, with the fitted spline and its band; [Which genes respond to dose](../statistics/dose-response/#dose-response-tests) explains the tests.

```{eval-rst}
.. automodule:: cytepool.dose_responses
   :no-members:
   :no-index:

.. autofunction:: cytepool.dose_response

.. autoclass:: cytepool.DoseResponseResult
```
