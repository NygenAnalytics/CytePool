"""Design matrices, contrasts and design checks."""

from .checks import DesignIssue, DesignReport
from .contrasts import Contrasts, make_contrasts
from .design import Design
from .formula import FormulaTerms, RandomEffect, parse_formula, parse_random_effects
from .model_matrix import INTERCEPT, ModelMatrix, model_matrix

__all__ = [
    "INTERCEPT",
    "Contrasts",
    "Design",
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
