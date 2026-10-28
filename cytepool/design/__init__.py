"""Design matrices, contrasts and design checks.

:class:`cytepool.Design` joins a sample table, an R formula of the fixed effects,
random effects in lme4's notation and named contrasts, and its parts are here:
:func:`model_matrix` codes a formula as a :class:`ModelMatrix`, :func:`parse_formula`
and :func:`parse_random_effects` read formulas into :class:`FormulaTerms` and
:class:`RandomEffect` terms, :func:`make_contrasts` turns expressions into
:class:`Contrasts`, and ``Design.check`` returns a :class:`DesignReport` of
:class:`DesignIssue` entries. https://docs.nygen.io/CytePool/statistics/linear-models
explains designs, their contrasts and their checks.
"""

from .checks import DesignIssue, DesignReport
from .contrasts import Contrasts, make_contrasts
from .design import Design as Design
from .formula import FormulaTerms, RandomEffect, parse_formula, parse_random_effects
from .model_matrix import INTERCEPT, ModelMatrix, model_matrix

# Design is the top level's, cytepool.Design.
__all__ = [
    "INTERCEPT",
    "Contrasts",
    "DesignIssue",
    "DesignReport",
    "FormulaTerms",
    "ModelMatrix",
    "RandomEffect",
    "make_contrasts",
    "model_matrix",
    "parse_formula",
    "parse_random_effects",
]
