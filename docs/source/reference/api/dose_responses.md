---
myst:
  html_meta:
    description: The dose tests of CytePool, which test in every cell type of a cube whether each gene responds to a dose, along a spline of dose, in its departure from a line, by its linear trend, by its dose groups and by Dunnett's, Williams-type and MCP contrasts, as contrasts of one moderated fit, with curves and 95% bands, a gate at a stated false discovery rate that decides which genes count as responding, and each gene's lowest effective dose, two-sided or in one direction, and the same tests of a cell type's signature scores.
---

# Dose responses

Which genes respond to a dose in each cell type, by moderated tests of one model of the dose groups, with the fitted spline and its band, multiple contrast tests of the doses, the gate that decides which genes count as responding and the lowest dose from which each gene responds, also of a cell type's pathway scores or activities; [Which genes respond to dose](../statistics/dose-response/#dose-response-tests) explains the tests, and [Signatures along dose](../statistics/dose-response/#dose-response-signatures) the tests of scores.

```{eval-rst}
.. automodule:: cytepool.dose_responses
   :no-members:
   :no-index:

.. autofunction:: cytepool.dose_response

.. autoclass:: cytepool.DoseResponseResult
```
