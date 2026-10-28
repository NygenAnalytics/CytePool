---
myst:
  html_meta:
    description: Build design matrices from R formulas, name contrasts and check designs for problems before fitting, with CytePool.
---

# Designs and contrasts

Designs, their formulas, model matrices, contrasts and checks; [Designs, linear models
and contrasts](../statistics/linear-models/) explains them.

```{eval-rst}
.. autoclass:: cytepool.Design

.. automodule:: cytepool.design
   :no-members:
```

## Formulas

```{eval-rst}
.. automodule:: cytepool.design.formula
   :no-members:
   :no-index:

.. autofunction:: cytepool.design.parse_formula

.. autoclass:: cytepool.design.FormulaTerms

.. autofunction:: cytepool.design.parse_random_effects

.. autoclass:: cytepool.design.RandomEffect

.. autofunction:: cytepool.design.formula.quote_name
```

## Model matrices

```{eval-rst}
.. automodule:: cytepool.design.model_matrix
   :no-members:
   :no-index:

.. autofunction:: cytepool.design.model_matrix

.. autoclass:: cytepool.design.ModelMatrix

.. py:data:: INTERCEPT
   :module: cytepool.design
   :value: '(Intercept)'

   The name of the intercept column of a model matrix, as R names it.
```

## Contrasts

```{eval-rst}
.. automodule:: cytepool.design.contrasts
   :no-members:
   :no-index:

.. autofunction:: cytepool.design.make_contrasts

.. autoclass:: cytepool.design.Contrasts
```

## Design checks

```{eval-rst}
.. automodule:: cytepool.design.checks
   :no-members:
   :no-index:

.. autoclass:: cytepool.design.DesignReport

.. autoclass:: cytepool.design.DesignIssue
```
