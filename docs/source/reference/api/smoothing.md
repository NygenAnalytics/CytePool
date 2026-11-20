---
myst:
  html_meta:
    description: LOWESS smoothing that matches R's lowess, limma's rule for the LOWESS span of a trend, and natural cubic spline bases that keep their knots, in R's B-spline form or Harrell's restricted cubic form, with knots placed as R's splines::ns and Hmisc's rcspline.eval place them, in CytePool.
---

# Smoothing and spline bases

LOWESS smoothing, limma's rule for its span, and natural cubic spline bases that keep their knots; [Spline columns](../statistics/linear-models/#spline-bases) explains the spline bases, and [The span](../statistics/limma-voom/#voom-span) the span of a LOWESS trend.

```{eval-rst}
.. automodule:: cytepool.stats.smoothing
```
