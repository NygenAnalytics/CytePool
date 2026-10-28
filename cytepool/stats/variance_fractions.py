"""Variance partitioning: the share of each gene's variance that each variable explains.

:func:`variance_fractions` fits every feature of a samples x features matrix of
log-expression values with one linear mixed model that holds every variable of a design
(Hoffman and Schadt 2016), with the model of :mod:`cytepool.stats.mixed_models` (Bates
et al. 2015), and returns, in a :class:`VarianceFractions`, the fraction of the
feature's variance that each variable explains and the residual fraction, as
variancePartition's ``fitExtractVarPartModel`` computes them;
:func:`cytepool.variance_partitioning` does it for every group (cell type) of a cube.
https://docs.nygen.io/CytePool/statistics/mixed-models#variance-fractions explains the
model and the fractions.

The fractions of a feature sum to one, and a random effect whose variance is estimated
at zero (a singular fit in lme4) has a fraction of exactly zero. They do not change when
the weights of a feature are multiplied by a constant, when the levels of a factor are
relabelled, or when the values of a feature are multiplied by a constant or shifted.

Matching variancePartition:
    The fractions match variancePartition 1.36.2 with lme4 1.1-38: its documentation
    gives the method, and running it on chosen inputs settled the conventions it leaves
    open.

    - The variances are estimated by maximum likelihood, as ``fitExtractVarPartModel``
      does by default (``REML = FALSE``): its help page says that maximum likelihood
      gave unbiased fractions, where REML (Patterson and Thompson 1971) can show
      substantial bias.
    - The variance of a fixed-effect term is the weighted variance of its fitted
      contribution, with the denominator of a maximum likelihood variance; in a mixed
      model the terms share the variance of all the fitted fixed effects in proportion
      to their own, and in a linear model each term has its own (``calcVarPart`` of
      ``lmer`` and ``lm`` fits).
    - ``fitExtractVarPartModel`` divides each gene's precision weights by their mean,
      as ``dream`` does (Hoffman and Roussos 2021), so the residual variance is that of
      an observation of average weight and the fractions do not depend on the scale of
      the weights. Without random effects the residual variance is the weighted mean of
      the squared residuals, the maximum likelihood estimate.
    - A design with random effects must give every categorical variable as a random
      effect: a factor among its fixed effects is refused, as variancePartition refuses
      it ("Must model either _all_ or _no_ categorical variables as random effects"),
      and so is a formula without an intercept. Without random effects, factors are
      fixed effects of the linear model.

Differences from variancePartition:
    https://docs.nygen.io/CytePool/statistics/mixed-models#mixed-models-limits lists
    them; :func:`variance_fractions` and :class:`VarianceFractions` give those that
    change the results.

References:
    Hoffman GE, Schadt EE (2016). variancePartition: interpreting drivers of variation
    in complex gene expression studies. BMC Bioinformatics 17, 483.
    doi:10.1186/s12859-016-1323-z

    Bates D, Maechler M, Bolker B, Walker S (2015). Fitting linear mixed-effects models
    using lme4. Journal of Statistical Software 67(1), 1-48. doi:10.18637/jss.v067.i01

    Hoffman GE, Roussos P (2021). Dream: powerful differential expression analysis for
    repeated measures designs. Bioinformatics 37, 192-201.
    doi:10.1093/bioinformatics/btaa687

    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29

    Patterson HD, Thompson R (1971). Recovery of inter-block information when block
    sizes are unequal. Biometrika 58, 545-554. doi:10.1093/biomet/58.3.545
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .._jit import thread_limit
from .._masked_arrays import real_array, reject_masked, unmasked_array
from .._messages import listing
from ..design import Design
from ..design.checks import _aliasing, _relation
from ._mixed_model_designs import (
    MixedDesign,
    maximum_likelihood,
    mixed_design,
    random_intercepts,
)
from .linear_models import LinearModelFit, fit_linear_model
from .mixed_models import MixedModelFit
from .mixed_models import _fit as _fit_mixed_model

__all__ = ["VarianceFractions", "variance_fractions"]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

RESIDUALS = "residuals"
"""The name of the residual variable, the last of every partition."""
# A feature whose variance is at most this fraction of the size of its values, in
# the norm the weights define, is constant to within rounding.
_CONSTANT = 1e-12


@dataclass(frozen=True, eq=False, repr=False)
class VarianceFractions:
    """The fraction of every feature's variance that each variable explains.

    Rows of the arrays are features, in the order of the columns of ``y``, and
    columns are variables, in the order of ``variables``. The arrays are read-only.
    A feature without a fit has NaN fractions: one that the fixed and random effects
    reproduce, as :class:`~cytepool.stats.mixed_models.MixedModelFit` describes.

    Attributes:
        fractions: ``(n_features, n_variables)`` fraction of each feature's
            variance that each variable explains; each row sums to one.
        variances: ``(n_features, n_variables)`` the variances the fractions are
            shares of (``calcVarPart(fit, returnFractions = FALSE)``): the variance
            component of each random effect, the variance of the fitted
            contribution of each fixed-effect term, and the residual variance of
            an observation of the feature's average weight, on the scale of the
            values.
        variables: The names of the variables: the random effects, as lme4 names
            their grouping factors (such as ``"donor"`` or ``"sample:donor"``), in
            the order of the design; the fixed-effect terms, as the formula labels
            them (such as ``"age"``); and ``"residuals"``, last.
        random_effects: The names of the random effects among ``variables``.
        fixed_effects: The names of the fixed-effect terms among ``variables``.
        method: ``"ml"`` or ``"reml"``, how the variances were estimated.
        converged: ``(n_features,)`` False where a feature was not fitted, or its
            mixed model fit did not converge (its fractions are those where the fit
            stopped).
        fit: The fits: a :class:`~cytepool.stats.mixed_models.MixedModelFit`, or
            for a design without random effects a
            :class:`~cytepool.stats.linear_models.LinearModelFit`.
    """

    fractions: FloatArray
    variances: FloatArray
    variables: tuple[str, ...]
    random_effects: tuple[str, ...]
    fixed_effects: tuple[str, ...]
    method: str
    converged: BoolArray
    fit: MixedModelFit | LinearModelFit

    def __repr__(self) -> str:
        return (
            f"VarianceFractions(n_features={self.fractions.shape[0]}, "
            f"variables={self.variables!r}, method={self.method!r})"
        )


def variance_fractions(
    y: npt.ArrayLike,
    design: Design,
    *,
    weights: npt.ArrayLike | None = None,
    method: Literal["ml", "reml"] = "ml",
    n_threads: int | None = None,
) -> VarianceFractions:
    """The fraction of every feature's variance that each variable of a design explains.

    variancePartition's ``fitExtractVarPartModel``: every feature (column of ``y``)
    is fitted with the design's fixed effects and random intercepts, by maximum
    likelihood as lme4's ``lmer(REML = FALSE)`` fits it (see
    :func:`~cytepool.stats.mixed_models.fit_mixed_model`), and its variance is
    partitioned among the random effects, the fixed-effect terms and the residual.
    Each variable contributes a variance, and its fraction is that variance over their
    sum: a random effect k its variance component sigma_k^2; a fixed-effect term j the
    variance of its fitted contribution X_j beta_j over the samples, weighted,
    sum_i w_i (v_i - v_w)^2 / sum_i w_i about the weighted mean v_w, and with random
    effects scaled so that the terms share the variance of all the fitted fixed
    effects, X beta, in proportion to their own; and the residual sigma^2 / mean(w),
    the residual variance of an observation of the feature's average weight.
    https://docs.nygen.io/CytePool/statistics/mixed-models#variance-fractions explains
    the fractions. Categorical variables are given as random
    effects and continuous ones as fixed effects, as in ``Design(samples, "~ age",
    random="(1 | donor) + (1 | condition)")``. A design without random effects is
    fitted by weighted least squares, and its factors are fixed effects.

    Args:
        y: ``(n_samples, n_features)`` expression values on a log scale, such as the
            log-CPM values of voom. Every value must be finite.
        design: A ``Design`` with an intercept, whose random effects are random
            intercepts, such as ``random="(1 | donor) + (1 | batch)"``, and whose
            fixed effects are then continuous covariates; or a design without
            random effects. Its fixed effects must have full column rank, and its
            random effects must pass ``design.check()``.
        weights: Precision weights, ``(n_samples,)`` for sample weights or
            ``(n_samples, n_features)`` for observation weights, such as those of
            :func:`~cytepool.stats.dream.voom_with_dream_weights`: the residual
            variance of an observation is sigma^2 / weight. They must be finite
            and positive; only their ratios within a feature matter.
        method: ``"ml"`` for maximum likelihood, variancePartition's default, or
            ``"reml"`` for restricted maximum likelihood. Without random effects,
            ``"reml"`` divides the residual sum of squares by n - p, where
            variancePartition divides it by n whatever its ``REML`` says.
        n_threads: The most threads for fitting features, as for
            ``fit_mixed_model``; the result does not depend on it.

    Returns:
        The fractions and variances of every feature, with the fits.

    Raises:
        TypeError: If ``design`` is not a ``Design``, an input is not real-valued,
            ``method`` is not a string, ``n_threads`` is not an integer, or an
            input is or holds a numpy masked array.
        ValueError: If shapes disagree, ``y`` is not finite, a weight is not finite
            and positive, ``method`` is unknown, the formula has no intercept, a
            design with random effects has a factor among its fixed effects or a
            random effect that is not a random intercept, the fixed effects are not
            of full column rank, a random effect fails the design checks, the
            fixed and random effects together span every sample, or a variable is
            named ``"residuals"`` or two variables have the same name.

    Example:
        >>> import numpy as np
        >>> from cytepool.design import Design
        >>> samples = {"donor": ["D1", "D1", "D2", "D2", "D3", "D3", "D4", "D4"],
        ...            "age": [31.0, 31.0, 45.0, 45.0, 52.0, 52.0, 60.0, 60.0],
        ...            "condition": ["ctrl", "stim"] * 4}
        >>> design = Design(samples, "~ age",
        ...                 random="(1 | donor) + (1 | condition)")
        >>> y = np.random.default_rng(1).normal(5.0, 1.0, (8, 3))
        >>> result = variance_fractions(y, design)
        >>> result.variables
        ('donor', 'condition', 'age', 'residuals')
    """
    ml = maximum_likelihood(method)
    terms = _checked_terms(design)
    with thread_limit(n_threads):
        if design.random:
            parts = mixed_design(design)
            return _mixed_fractions(y, weights, ml, parts, terms)
        x = _full_rank_matrix(design)
        values = _finite_values(y, x.shape[0])
        rows = _positive_weight_rows(weights, values.shape)
        fit = fit_linear_model(
            values,
            x,
            weights=None if weights is None else rows[0] if len(rows) == 1 else rows.T,
            coefficient_names=design.columns,
            n_threads=n_threads,
        )
        return _linear_fractions(values, x, rows, fit, ml, terms)


@dataclass(frozen=True)
class _Terms:
    """The variables of a design: random effects, fixed-effect terms and columns."""

    random: tuple[str, ...]  # names of the random effects
    fixed: tuple[str, ...]  # labels of the fixed-effect terms
    assign: npt.NDArray[np.intp]  # the term of each column, 0 for the intercept

    @property
    def variables(self) -> tuple[str, ...]:
        return (*self.random, *self.fixed, RESIDUALS)


def _checked_terms(design: Any) -> _Terms:
    """The variables of a design for variance partitioning, checked."""
    reject_masked(design, "design")
    if not isinstance(design, Design):
        raise TypeError(
            "design must be a Design, such as Design(samples, '~ age', random='(1 | "
            f"donor)'), got {type(design).__name__}"
        )
    if design.random:
        random_intercepts(design)
    model = design.model_matrix
    if not model.intercept:
        raise ValueError(
            f"the formula {design.formula!r} has no intercept: the fractions share the "
            "variance of each feature about its mean, which needs one, as "
            "variancePartition requires; write the formula with an intercept, such as "
            "'~ age' in place of '~ 0 + age'"
        )
    if design.random:
        factors = [
            label
            for index, label in enumerate(model.terms, start=1)
            if any(
                level is not None
                for column, term in enumerate(model.assign)
                if term == index
                for _, level in model.coding[column]
            )
        ]
        if factors:
            example = " + ".join(f"(1 | {name})" for name in factors if ":" not in name)
            raise ValueError(
                "with random effects, variance partitioning models every categorical "
                "variable as a random effect, as variancePartition does, but the "
                f"fixed effects hold the factor term(s) {listing(factors)}; give them "
                f"as random intercepts{f', such as {example!r}' if example else ''}, "
                "or give every variable as a fixed effect, without random effects"
            )
    random = tuple(effect.group for effect in design.random)
    names = [*random, *model.terms]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"the variable(s) {listing(repeated)} are both random and fixed effects; "
            "give each variable once"
        )
    if RESIDUALS in names:
        raise ValueError(
            f"a variable is named {RESIDUALS!r}, the name of the residual fraction; "
            "rename it in the sample table"
        )
    return _Terms(random, model.terms, np.asarray(model.assign, dtype=np.intp))


def _full_rank_matrix(design: Design) -> FloatArray:
    """The fixed-effects matrix of a design without random effects, of full rank."""
    x = np.array(design.matrix, dtype=np.float64, order="C")
    aliasing = _aliasing(x)
    if aliasing.aliased:
        relations = [
            _relation(design.model_matrix, aliasing, j) for j in aliasing.aliased
        ]
        raise ValueError(
            "the fixed effects are not of full column rank: "
            + "; ".join(relations)
            + "; remove the aliased variables or reparametrise the design (see "
            "design.check())"
        )
    if x.shape[0] <= x.shape[1]:
        raise ValueError(
            f"the design has {x.shape[1]} columns for {x.shape[0]} samples, so it fits "
            "them exactly and leaves no residual variance; add samples or remove "
            "variables"
        )
    return x


def _finite_values(y: npt.ArrayLike, n_samples: int) -> FloatArray:
    """The values as a finite 2-D float64 array of one row per sample."""
    values = real_array(unmasked_array(y, "y"), "y", boolean=False)
    if values.ndim != 2:
        raise ValueError(
            "y must be a 2-D array of shape (n_samples, n_features), "
            f"got shape {values.shape}"
        )
    if values.shape[0] != n_samples:
        raise ValueError(
            f"y has {values.shape[0]} rows but the design has {n_samples} samples; "
            "rows of both are samples"
        )
    if values.shape[1] == 0:
        raise ValueError("y must have at least one feature (column)")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"y must be finite, but it has {int(np.count_nonzero(np.isnan(values)))} "
            f"missing (NaN) and {int(np.count_nonzero(np.isinf(values)))} infinite "
            "values; remove the features with missing values first"
        )
    return np.asarray(values, dtype=np.float64)


def _positive_weight_rows(
    weights: npt.ArrayLike | None, shape: tuple[int, int]
) -> FloatArray:
    """The weights as rows of samples, one per feature or one for all; ones without."""
    if weights is None:
        return np.ones((1, shape[0]))
    w = real_array(unmasked_array(weights, "weights"), "weights")
    if w.shape == (shape[0],):
        rows = w[np.newaxis, :]
    elif w.shape == shape:
        rows = w.T
    else:
        raise ValueError(
            f"weights must have shape ({shape[0]},) for sample weights or {shape} "
            f"for observation weights, got {w.shape}"
        )
    with np.errstate(invalid="ignore"):
        invalid = int(np.count_nonzero(~(np.isfinite(w) & (w > 0.0))))
    if invalid:
        raise ValueError(
            f"weights must be finite and positive; {invalid} are zero, negative or not "
            "finite"
        )
    return np.ascontiguousarray(rows, dtype=np.float64)


def _mixed_fractions(
    y: npt.ArrayLike,
    weights: npt.ArrayLike | None,
    ml: bool,
    parts: MixedDesign,
    terms: _Terms,
) -> VarianceFractions:
    """The fractions of mixed model fits; call within ``thread_limit``."""
    fit = _fit_mixed_model(y, weights, ml, parts)
    n_features = fit.coefficients.shape[0]
    rows = (
        np.ones((1, parts.x.shape[0]))
        if weights is None
        else _positive_weight_rows(weights, (parts.x.shape[0], n_features))
    )
    # In a mixed model the fixed-effect terms share the variance of all the fitted
    # fixed effects; the residual variance is that of an observation of average
    # weight, as fitExtractVarPartModel divides the weights by their mean (found by
    # probing).
    fixed = _fixed_variances(parts.x, terms, fit.coefficients, rows, apportioned=True)
    residual = fit.residual_variance / fit.mean_weights
    variances = np.column_stack([fit.variance_components, fixed, residual])
    return _result(variances, terms, "ml" if ml else "reml", fit.converged, fit)


def _linear_fractions(
    values: FloatArray,
    x: FloatArray,
    rows: FloatArray,
    fit: LinearModelFit,
    ml: bool,
    terms: _Terms,
) -> VarianceFractions:
    """The fractions of weighted least squares fits of a design without random
    effects."""
    n, p = x.shape
    # In a linear model each term has the variance of its own contribution, as
    # calcVarPart gives it for an lm fit (found by probing).
    fixed = _fixed_variances(x, terms, fit.coefficients, rows, apportioned=False)
    # sigma^2 is the residual variance of an observation of weight one, sum w r^2 /
    # (n - p); maximum likelihood takes the sum over n.
    scale = 1.0 if not ml else (n - p) / n
    residual = fit.sigma**2 * scale / np.mean(rows, axis=1)
    variances = np.column_stack([fixed, residual])
    # A feature constant to within rounding has no variance to share; a mixed model
    # leaves such a feature unfitted too.
    spread = _weighted_variance(values.T, rows)
    size = np.sum(rows * values.T**2, axis=1) / np.sum(rows, axis=1)
    variances[spread <= _CONSTANT**2 * size] = np.nan
    converged = np.isfinite(residual)
    return _result(variances, terms, "ml" if ml else "reml", converged, fit)


def _fixed_variances(
    x: FloatArray,
    terms: _Terms,
    coefficients: FloatArray,
    rows: FloatArray,
    *,
    apportioned: bool,
) -> FloatArray:
    """The variance of the fitted contribution of every fixed-effect term.

    ``rows`` holds the weights of each feature, or one row for all. With
    ``apportioned``, the terms share the variance of all the fitted fixed effects in
    proportion to the variances of their own contributions.
    """
    n_features = coefficients.shape[0]
    variances = np.zeros((n_features, len(terms.fixed)))
    for t in range(len(terms.fixed)):
        columns = terms.assign == t + 1
        contribution = coefficients[:, columns] @ x[:, columns].T
        variances[:, t] = _weighted_variance(contribution, rows)
    if apportioned and len(terms.fixed) > 1:
        columns = terms.assign != 0
        total = _weighted_variance(coefficients[:, columns] @ x[:, columns].T, rows)
        sums = variances.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(sums > 0.0, total / sums, 0.0)
        variances *= share[:, np.newaxis]
    return variances


def _weighted_variance(values: FloatArray, rows: FloatArray) -> FloatArray:
    """sum_i w_i (v_i - v_w)^2 / sum_i w_i for every row of ``values``, about the
    weighted mean v_w; ``rows`` holds one row of weights per row of values, or one
    for all."""
    totals = np.sum(rows, axis=1)
    means = np.sum(rows * values, axis=1) / totals
    deviations = values - means[:, np.newaxis]
    variance: FloatArray = np.sum(rows * deviations**2, axis=1) / totals
    return variance


def _result(
    variances: FloatArray,
    terms: _Terms,
    method: str,
    converged: BoolArray,
    fit: MixedModelFit | LinearModelFit,
) -> VarianceFractions:
    total = np.sum(variances, axis=1, keepdims=True)
    # A feature without variance, such as a constant one, has no fractions.
    with np.errstate(invalid="ignore", divide="ignore"):
        fractions = np.where(total > 0.0, variances / total, np.nan)
    return VarianceFractions(
        fractions=_frozen(fractions),
        variances=_frozen(variances),
        variables=terms.variables,
        random_effects=terms.random,
        fixed_effects=terms.fixed,
        method=method,
        converged=_frozen(converged & np.all(np.isfinite(fractions), axis=1), bool),
        fit=fit,
    )


def _frozen(values: Any, dtype: type = np.float64) -> Any:
    array: npt.NDArray[Any] = np.array(values, dtype=dtype)
    array.flags.writeable = False
    return array
