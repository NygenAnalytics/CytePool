---
myst:
  html_meta:
    description: Build design matrices from R formulas, name contrasts and check designs for problems before fitting, with CytePool.
---

# Designs and contrasts

A {class}`~cytepool.Design` joins a sample table, the fixed effects as an R formula,
optional random effects in lme4 notation and named contrasts. Its model matrix has the
columns and column names that R's `model.matrix` gives for the same level order.
Without explicit levels, factor levels are sorted by code point, as R sorts them with
`LC_COLLATE=C`; R's default follows the session's locale.
{meth}`~cytepool.Design.check` reports problems such as aliased columns before anything
is fitted.

```{eval-rst}
.. autoclass:: cytepool.Design
```

The parts of a design are in `cytepool.design`: its model matrix, its contrasts, its
formula and random effects as they are read, and the report of its check.

```{eval-rst}
.. automodule:: cytepool.design
```

```{eval-rst}
.. py:data:: INTERCEPT
   :module: cytepool.design
   :value: '(Intercept)'

   The name of the intercept column of a model matrix, as R names it.
```

Column names and term labels write a variable name that is not syntactic in R between
backticks, as R does: a factor `cell type` gives columns such as `` `cell type`B ``,
and a contrast refers to them by the same names.

```{eval-rst}
.. autofunction:: cytepool.design.formula.quote_name
```
