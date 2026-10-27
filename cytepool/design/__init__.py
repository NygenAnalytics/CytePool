"""Design matrices, contrasts and design checks."""

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
