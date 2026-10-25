"""Sample quality weights from a heteroscedastic linear model, as in limma's arrayWeights.

Samples differ in quality: pooled from fewer cells, sequenced less deeply, or taken
from an outlying donor, some follow the linear model less closely than others. Ritchie
et al. (2006) estimate a weight for each sample from the residuals of every gene, and
Liu et al. (2015) extend the model to groups of samples and combine it with voom.

For gene g and sample j, with design rows x_j and prior weights w_gj (one when none
are given), the model is

    E(y_gj) = x_j' b_g,    var(y_gj) = sigma_g^2 exp(u_j) / w_gj,

with the log variance factors u = Z gamma shared by every gene. Z is an orthonormal
basis of the columns of the variance design projected off the constant vector, so u
sums to zero and only the span of the variance design matters (found by probing: limma
gives the same weights for any parametrisation of it, with or without an intercept).
The default variance design gives each sample a variance of its own, and
``variance_groups`` one per group. The sample quality weights are exp(-u_j), whose
product is one.

gamma is estimated by residual maximum likelihood (REML) with the gene variances
sigma_g^2 profiled out. For a weighted least squares fit of gene g at weights
w_gj exp(-u_j) over its samples, with standardised residuals e_gj, leverages h_gj,
residual degrees of freedom d_g and s_g^2 = sum_j e_gj^2 / d_g, the score (twice the
gradient of the REML log likelihood) and the information are

    U_g = Z' z_g,  z_gj = e_gj^2 / s_g^2 - (1 - h_gj),
    I_g = Z' (V_g - (1 - h_g)(1 - h_g)' / d_g) Z,

where V_g has elements (delta_jk - h_g,jk)^2, h_g,jk being those of the hat matrix
(Smyth 2002). The weights are squeezed toward equality by a prior equivalent to
``prior_n`` genes at equal weights: it adds prior_n Z'(exp(-u) - 1) to the score and
prior_n Z'Z, the identity, to the information (found by probing). Two algorithms:

- ``"reml"``: Fisher scoring from equal weights, gamma <- gamma + d with the step
  d = I^-1 U, U and I summed over the genes and the prior, until the criterion
  d'U / ((G + prior_n) q) is below ``tolerance``, for q columns of Z and G genes;
  where the criterion rises instead, the iterations stop before that step (found by
  probing).
- ``"gene_by_gene"``: one pass through the genes in order (Ritchie et al. 2006). Each
  gene adds its information, with V_g replaced by the diagonal matrix of its row sums,
  diag(1 - h_g), to an information matrix that starts at prior_n Z'Z, and moves gamma
  by that matrix inverted times its score, evaluated at the current gamma. The result
  depends on the order of the genes.
- ``"auto"``: REML without prior weights or missing values, and gene by gene
  otherwise, as limma chooses.

Results match limma 3.62.1. Conventions that its documentation leaves open, found by
probing (running limma as a black box):

- Gene by gene, a gene is skipped where it has fewer than two residual degrees of
  freedom, or a residual variance below 1e-15 at the current weights. Its samples are
  those with an observed value and a positive weight.
- REML uses only the genes with every value observed and every prior weight positive.
  Without prior weights, it leaves out the genes whose residual variance at equal
  weights is below 1e-15. With prior weights, every gene adds its information, a gene
  whose residual variance at the current weights is below 1e-15 adds no score, and the
  iterations stop when d'U / ((G + prior_n)^2 q) is below ``tolerance``, going on
  where the criterion rises.
- Every weight is one when there are fewer than two genes, when the design leaves
  fewer than two residual degrees of freedom, or, for REML without prior weights, when
  fewer than two genes have a positive residual variance; CytePool warns.

REML stops where limma stops, so that the weights are limma's. With the default
``tolerance`` they are then within about half a percent of the REML estimate
without prior weights, and up to about 15% from it with prior weights, where the
criterion falls below the tolerance after one or two iterations (as with the voom
weights of :func:`~cytepool.stats.voom.voom_with_quality_weights`), or where the
criterion rises first. A smaller ``tolerance`` converges further where the iterations
converge. A step that would give weights that are not finite is not taken: REML stops
before it, with a warning.

Deliberate deviations, where limma returns values without meaning or fails:

- ``variance_design`` and ``variance_groups`` together are an error; limma uses
  ``var.group`` and ignores ``var.design``.
- ``prior_n`` must be at least zero; limma accepts a negative prior.
- A design that is not of full column rank is an error, as in
  :func:`~cytepool.stats.linear_models.fit_linear_model`; limma drops aliased columns.
- REML with fewer than two complete genes gives weights of one, with a warning; limma
  fails when one is left.
- An information matrix that is singular is an error: some sample carries no
  information about its variance, such as the only sample of a group of the design,
  and ``prior_n`` is zero or too small to make up for it. limma's REML returns weights
  that rounding error decides.

References:
    Ritchie ME, Diyagama D, Neilson J, van Laar R, Dobrovic A, Holloway A, Smyth GK
    (2006). Empirical array quality weights in the analysis of microarray data. BMC
    Bioinformatics 7, 261.

    Liu R, Holik AZ, Su S, Jansz N, Chen K, Leong HS, Blewitt ME, Asselin-Labat ML,
    Smyth GK, Ritchie ME (2015). Why weight? Modelling sample and observational level
    variability improves power in RNA-seq analyses. Nucleic Acids Research 43, e97.

    Smyth GK (2002). An efficient algorithm for REML in heteroscedastic regression.
    Journal of Computational and Graphical Statistics 11, 836-847.
"""

import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._linear_algebra import RANK_TOLERANCE, column_space
from cytepool._masked_arrays import reject_masked, unmasked_array
from cytepool.design.design import Design, _refuse_random_effects
from cytepool.stats.linear_models import (
    _aliased_columns,
    _design_matrix,
    _expression_matrix,
    _real_array,
)

__all__ = ["sample_quality_weights"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Method = Literal["auto", "gene_by_gene", "reml"]

# A gene whose residual variance is below this is left out (found by probing).
_ZERO_VARIANCE = 1e-15
# Genes need this many residual degrees of freedom (found by probing; Ritchie et al.
# 2006 restrict the method to experiments with at least two).
_MIN_RESIDUAL_DF = 2
# A pivot of the information matrix at most this multiple of the rounding error of
# its diagonal makes it singular.
_SINGULAR = 64.0 * np.finfo(np.float64).eps


class _SingularInformation(ValueError):
    """The information about the sample weights is singular, so a caller can say
    what to change in its own terms."""


@dataclass(frozen=True)
class _VarianceModel:
    """The variance model and the settings of its estimation, checked."""

    basis: FloatArray  # (n_samples, q) orthonormal, orthogonal to the constant
    prior_n: float
    method: Method
    max_iterations: int
    tolerance: float


def sample_quality_weights(
    y: npt.ArrayLike,
    design: Any,
    *,
    weights: npt.ArrayLike | None = None,
    variance_design: Any = None,
    variance_groups: npt.ArrayLike | None = None,
    prior_n: float = 10.0,
    method: Method = "auto",
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> FloatArray:
    """Estimate a quality weight for each sample from how well it fits the model.

    limma's ``arrayWeights`` (Ritchie et al. 2006; Liu et al. 2015); the module
    docstring gives the model, the algorithms and the conventions. Samples whose
    values have larger residuals across genes get lower weights. Multiply them into
    the precision weights of a fit, as
    :func:`~cytepool.stats.voom.voom_with_quality_weights` does for voom.

    Args:
        y: ``(n_samples, n_features)`` log-expression values, such as the log2
            counts per million of :func:`~cytepool.stats.voom.voom`. NaN marks a
            missing value; infinite values are rejected.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank,
            or a :class:`~cytepool.design.Design` without random effects. A column
            of ones treats the samples as replicates.
        weights: Prior weights: ``(n_samples,)`` sample weights,
            ``(n_samples, n_features)`` observation weights (such as voom's), or
            ``(1, n_features)`` feature weights. Finite and non-negative where ``y``
            is observed; a zero weight leaves the value out. A vector is always
            sample weights. limma's ``arrayWeights`` takes a matrix only.
        variance_design: ``(n_samples, n_columns)`` design of the variance model,
            a vector for one column, or a ``Design``. Only its span matters, and its
            constant part is ignored. Defaults to a variance for each sample.
        variance_groups: One label per sample: samples with the same label share a
            weight. Another way to give a variance design; give at most one of the
            two.
        prior_n: The number of genes at equal weights that the prior is worth; larger
            values squeeze the weights more toward one. At least 0.
        method: ``"reml"``, ``"gene_by_gene"``, or ``"auto"``, which chooses REML
            when there are no prior weights and no missing values.
        max_iterations: The most REML iterations, at least 1.
        tolerance: REML stops when the criterion of the module docstring falls
            below it, as limma's does. Positive; lower it to converge further than
            limma.

    Returns:
        ``(n_samples,)`` positive weights whose product is one.

    Warns:
        RuntimeWarning: If every weight is one for lack of genes or of residual
            degrees of freedom, if REML leaves out genes with missing values or zero
            weights, or if REML stops without converging: at ``max_iterations``, or
            where its criterion rises or its step is not finite.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If shapes disagree, ``y`` has infinite values, the design is not
            of full column rank or has random effects, a weight is negative or not
            finite where ``y`` is observed, the variance design is not finite, does
            not vary between samples, is given twice or is a ``Design`` with random
            effects, ``variance_groups`` has fewer than two groups, an option is out
            of range, or the information matrix is singular (some sample carries no
            information about its variance, and ``prior_n`` is too small).
    """
    x, names = _design_matrix(design, None)
    _refuse_random_effects(design, "sample_quality_weights")
    values = _expression_matrix(y, x.shape[0])
    _require_full_rank(x, names)
    prior = _prior_weights(weights, values.shape, ~np.isnan(values))
    model = _variance_model(
        variance_design,
        variance_groups,
        x.shape[0],
        prior_n=prior_n,
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        methods=("auto", "gene_by_gene", "reml"),
    )
    log_variances, notes = _estimate(values, x, prior, model)
    for note in notes:
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    return _weights_of(log_variances)


def _estimate(
    values: FloatArray, x: FloatArray, prior: FloatArray | None, model: _VarianceModel
) -> tuple[FloatArray, list[str]]:
    """The log variance factors u of the samples, and warnings for the caller.

    Args:
        values: ``(n_samples, n_features)`` values, NaN where missing.
        x: The design, checked.
        prior: Prior weights from :func:`_prior_weights`, or None.
        model: The variance model and settings.
    """
    n_samples, n_features = values.shape
    method = model.method
    if method == "auto":
        complete = prior is None and not np.isnan(values).any()
        method = "reml" if complete else "gene_by_gene"
    if n_features < 2:
        return np.zeros(n_samples), [
            f"sample quality weights need at least two features, got {n_features}; "
            "every weight is 1"
        ]
    if method == "gene_by_gene":
        return _gene_by_gene(values, x, prior, model)
    return _reml(values, x, prior, model)


def _gene_by_gene(
    values: FloatArray, x: FloatArray, prior: FloatArray | None, model: _VarianceModel
) -> tuple[FloatArray, list[str]]:
    n_samples = values.shape[0]
    if prior is None:
        weight_rows = np.ones((1, n_samples))
    elif prior.ndim == 1:
        weight_rows = prior[np.newaxis, :]
    else:
        weight_rows = prior.T
    gamma, used, singular = _gene_by_gene_pass(
        np.ascontiguousarray(values.T),
        np.ascontiguousarray(weight_rows, dtype=np.float64),
        x,
        model.basis,
        model.prior_n,
        RANK_TOLERANCE,
    )
    if singular:
        raise _SingularInformation(_singular_message())
    notes = []
    if used == 0:
        notes.append(
            "no feature has at least two residual degrees of freedom and a positive "
            "residual variance, so the sample quality weights cannot be estimated; "
            "every weight is 1"
        )
    log_variances: FloatArray = model.basis @ gamma
    return log_variances, notes


def _reml(
    values: FloatArray, x: FloatArray, prior: FloatArray | None, model: _VarianceModel
) -> tuple[FloatArray, list[str]]:
    n_samples, n_features = values.shape
    n_coefficients = x.shape[1]
    notes = []
    usable = ~np.isnan(values).any(axis=0)
    if prior is not None:
        if prior.ndim == 1:
            usable &= bool(np.all(prior > 0.0))
        else:
            usable &= np.all(prior > 0.0, axis=0)
    left_out = n_features - int(np.count_nonzero(usable))
    if left_out:
        notes.append(
            f"REML uses only the features with every value observed and every "
            f"weight positive, and leaves out {left_out} of {n_features}; "
            "method='gene_by_gene' uses them"
        )
    unit = np.zeros(n_samples)
    if n_samples - n_coefficients < _MIN_RESIDUAL_DF:
        notes.append(_too_few_df_message(n_samples - n_coefficients))
        return unit, notes
    values = values[:, usable]
    if values.shape[1] < 2:
        notes.append(
            "REML needs at least two features with every value observed and every "
            f"weight positive, got {values.shape[1]}; every weight is 1"
        )
        return unit, notes
    if prior is None:
        gamma, converged = _reml_scoring(values, x, None, model, notes)
    else:
        full = (
            prior if prior.ndim == 2 else np.repeat(prior[:, np.newaxis], n_features, 1)
        )
        gamma, converged = _reml_scoring(values, x, full[:, usable], model, notes)
    if not converged:
        notes.append(
            f"REML did not converge in {model.max_iterations} iterations (tolerance "
            f"{model.tolerance:g}); the sample weights are those of the last iteration"
        )
    log_variances: FloatArray = model.basis @ gamma
    return log_variances, notes


def _reml_scoring(
    values: FloatArray,
    x: FloatArray,
    prior: FloatArray | None,
    model: _VarianceModel,
    notes: list[str],
) -> tuple[FloatArray, bool]:
    """Fisher scoring for gamma from zero; returns gamma and whether it converged.

    Without prior weights every gene has the same hat matrix, so the information is
    that of one gene times their number; genes with no residual variance at equal
    weights are left out first. With prior weights each gene is fitted on its own.
    """
    basis = model.basis
    n_samples, n_coefficients = x.shape
    df = n_samples - n_coefficients
    n_parameters = basis.shape[1]
    if prior is None:
        fitted = column_space(x).basis
        residuals = values - fitted @ (fitted.T @ values)
        keep = np.einsum("ij,ij->j", residuals, residuals) / df >= _ZERO_VARIANCE
        values = values[:, keep]
        if values.shape[1] < 2:
            notes.append(
                "fewer than two features have a positive residual variance, so the "
                "sample quality weights cannot be estimated; every weight is 1"
            )
            return np.zeros(n_parameters), True
        scale = (values.shape[1] + model.prior_n) * n_parameters
    else:
        responses = np.ascontiguousarray(values.T)
        weight_rows = np.ascontiguousarray(prior.T)
        scale = (values.shape[1] + model.prior_n) ** 2 * n_parameters
    gamma = np.zeros(n_parameters)
    previous = math.inf
    for iteration in range(model.max_iterations):
        log_variances = basis @ gamma
        sample_weights = np.exp(-log_variances)
        if prior is None:
            score, information = _equal_prior_sums(values, x, sample_weights)
        else:
            score, information = _weighted_reml_sums(
                responses, weight_rows, x, log_variances, RANK_TOLERANCE
            )
            information = information + np.tril(information, -1).T
        score = basis.T @ (score + model.prior_n * (sample_weights - 1.0))
        information = basis.T @ information @ basis
        information[np.diag_indices(n_parameters)] += model.prior_n
        step = _solve(information, score)
        criterion = float(step @ score) / scale
        candidate = gamma + step
        with np.errstate(over="ignore"):
            usable = np.exp(-(basis @ candidate))
        if not (
            math.isfinite(criterion) and np.all(np.isfinite(usable) & (usable > 0))
        ):
            notes.append(
                f"REML stopped after {iteration} iterations: its next step gives "
                "weights that are not finite; the sample weights are those of the "
                "last iteration. method='gene_by_gene' or a larger prior_n may help"
            )
            return gamma, True
        if prior is None and criterion > previous:
            # limma stops where its criterion rises, without the step, and warns that
            # the tolerance is not achievable; with prior weights it goes on (found
            # by probing).
            notes.append(
                f"REML stopped after {iteration} iterations without reaching the "
                f"tolerance {model.tolerance:g}: its convergence criterion rose, as "
                "when rounding error limits the precision or the iterations "
                "oscillate; the sample weights are those of the last iteration, as "
                "limma's are"
            )
            return gamma, True
        gamma = candidate
        if criterion < model.tolerance:
            return gamma, True
        previous = criterion
    return gamma, False


def _equal_prior_sums(
    values: FloatArray, x: FloatArray, sample_weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """REML score per sample and information per pair of samples, summed over genes
    that share one hat matrix (no prior weights, no missing values)."""
    n_samples, n_coefficients = x.shape
    df = n_samples - n_coefficients
    root = np.sqrt(sample_weights)
    fitted = column_space(x * root[:, np.newaxis]).basis
    hat = fitted @ fitted.T
    remaining = 1.0 - np.diag(hat)
    scaled = values * root[:, np.newaxis]
    residuals = scaled - fitted @ (fitted.T @ scaled)
    variances = np.einsum("ij,ij->j", residuals, residuals) / df
    score = (residuals * residuals / variances).sum(axis=1)
    score -= values.shape[1] * remaining
    complement = np.eye(n_samples) - hat
    one_gene = complement * complement - np.outer(remaining, remaining) / df
    information: FloatArray = values.shape[1] * one_gene
    return score, information


def _solve(information: FloatArray, score: FloatArray) -> FloatArray:
    step = np.empty(score.size)
    if not _cholesky_solve(np.array(information), score.copy(), step):
        raise _SingularInformation(_singular_message())
    return step


def _weights_of(log_variances: FloatArray) -> FloatArray:
    """Weights exp(-u), scaled so that their product is one."""
    weights: FloatArray = np.exp(-(log_variances - log_variances.mean()))
    return weights


def _singular_message() -> str:
    return (
        "the information about the sample weights is singular: some samples carry "
        "no information about their variance, such as the only sample of a group "
        "in the design, whose residuals are always zero. Give prior_n > 0, which "
        "keeps their weights near 1, or a variance design that groups them with "
        "other samples"
    )


def _too_few_df_message(df: int) -> str:
    return (
        "sample quality weights need a design with at least two residual degrees of "
        f"freedom, got {df}; every weight is 1"
    )


def _require_full_rank(x: FloatArray, names: tuple[str, ...]) -> None:
    aliased = _aliased_columns(x)
    if aliased:
        listed = ", ".join(repr(names[j]) for j in aliased)
        raise ValueError(
            f"design is not of full column rank: column(s) {listed} are linear "
            "combinations of the preceding columns; remove them or reparametrise "
            "the design"
        )


def _prior_weights(
    weights: Any, shape: tuple[int, int], observed: npt.NDArray[np.bool_]
) -> FloatArray | None:
    """Prior weights: ``(n_samples,)`` as given, otherwise a full matrix.

    A ``(1, n_features)`` row of feature weights is repeated for every sample. The
    weights must be finite and non-negative wherever a value is observed.
    """
    if weights is None:
        return None
    array = _real_array(unmasked_array(weights, "weights"), "weights")
    n_samples, n_features = shape
    if array.shape == (n_samples,):
        kind, relevant = "sample weights", observed.any(axis=1)
    elif array.shape in ((1, n_features), shape):
        kind = "feature weights" if array.shape[0] == 1 else "observation weights"
        array, relevant = np.broadcast_to(array, shape), observed
    else:
        raise ValueError(
            f"weights must have shape ({n_samples},) for sample weights, "
            f"(1, {n_features}) for feature weights or {shape} for observation "
            f"weights, got {array.shape}"
        )
    with np.errstate(invalid="ignore"):
        invalid = int(
            np.count_nonzero(~(np.isfinite(array) & (array >= 0.0)) & relevant)
        )
    if invalid:
        raise ValueError(
            f"{kind} must be finite and non-negative wherever a value is observed; "
            f"{invalid} are negative or not finite"
        )
    return np.array(array, dtype=np.float64, order="C")


def _variance_model(
    variance_design: Any,
    variance_groups: npt.ArrayLike | None,
    n_samples: int,
    *,
    prior_n: Any,
    method: Any,
    max_iterations: Any,
    tolerance: Any,
    methods: tuple[str, ...],
) -> _VarianceModel:
    """Check the variance design and the settings of the estimation."""
    reject_masked(variance_design, "variance_design")
    if variance_design is not None and variance_groups is not None:
        raise ValueError(
            "give variance_design or variance_groups, not both: each defines the "
            "variance model"
        )
    if variance_groups is not None:
        matrix = _group_indicators(variance_groups, n_samples)
    elif variance_design is not None:
        matrix = _variance_design_matrix(variance_design, n_samples)
    else:
        matrix = np.eye(n_samples)
    # The constant column first: R's rank rule then drops any column that only adds
    # a constant, and the rest of the basis is orthogonal to it.
    space = column_space(np.column_stack([np.ones(n_samples), matrix]))
    basis = np.ascontiguousarray(space.basis[:, 1:])
    if basis.shape[1] == 0:
        raise ValueError(
            "the variance design must vary between samples: its columns are "
            "constant, so it gives every sample the same weight"
        )
    if method not in methods:
        listed = ", ".join(repr(m) for m in methods)
        raise ValueError(f"method must be one of {listed}, got {method!r}")
    return _VarianceModel(
        basis=basis,
        prior_n=_real_at_least(prior_n, "prior_n", 0.0),
        method=method,
        max_iterations=_positive_integer(max_iterations, "max_iterations"),
        tolerance=_positive_real(tolerance, "tolerance"),
    )


def _group_indicators(variance_groups: Any, n_samples: int) -> FloatArray:
    labels = unmasked_array(variance_groups, "variance_groups")
    if labels.shape != (n_samples,):
        raise ValueError(
            f"variance_groups must have one label per sample ({n_samples}), got "
            f"shape {labels.shape}"
        )
    if labels.dtype.kind == "O" and any(label is None for label in labels):
        raise ValueError("variance_groups must not contain None")
    if labels.dtype.kind in "fc" and np.isnan(labels).any():
        raise ValueError("variance_groups must not contain NaN")
    try:
        groups, codes = np.unique(labels, return_inverse=True)
    except TypeError as error:
        raise TypeError(
            "variance_groups must be comparable labels, such as integers or strings"
        ) from error
    if groups.size < 2:
        raise ValueError(
            "variance_groups must have at least two groups, got one: a single group "
            "gives every sample the same weight"
        )
    indicators: FloatArray = (
        codes.reshape(n_samples)[:, np.newaxis] == np.arange(groups.size)
    ).astype(np.float64)
    return indicators


def _variance_design_matrix(variance_design: Any, n_samples: int) -> FloatArray:
    if isinstance(variance_design, Design) and variance_design.random:
        raise ValueError(
            "variance_design must not have random effects: the variance model "
            "takes the span of fixed columns; build the Design without random="
        )
    matrix = getattr(variance_design, "matrix", variance_design)
    array = _real_array(unmasked_array(matrix, "variance_design"), "variance_design")
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2 or array.shape[0] != n_samples or array.shape[1] == 0:
        raise ValueError(
            f"variance_design must have one row per sample ({n_samples}) and at "
            f"least one column, got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("variance_design must be finite")
    return array


def _real_at_least(value: Any, name: str, lower: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not (math.isfinite(number) and number >= lower):
        raise ValueError(f"{name} must be finite and at least {lower:g}, got {number}")
    return number


def _positive_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {number}")
    return number


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return int(value)


@jit
def _gene_by_gene_pass(
    responses: FloatArray,
    weight_rows: FloatArray,
    design: FloatArray,
    basis: FloatArray,
    prior_n: float,
    tolerance: float,
) -> tuple[FloatArray, int, bool]:
    """The gene-by-gene update of Ritchie et al. (2006), one gene at a time in order.

    Args:
        responses: ``(n_features, n_samples)`` values, NaN where missing.
        weight_rows: ``(n_features, n_samples)`` prior weights, or one row shared by
            every feature.
        design: ``(n_samples, n_coefficients)`` design matrix.
        basis: ``(n_samples, q)`` orthonormal basis of the variance model.
        prior_n: Prior number of genes.
        tolerance: Relative norm at or below which a design column is aliased.

    Returns:
        gamma, the number of genes used, and whether the information was singular.
    """
    n_features, n_samples = responses.shape
    n_coefficients = design.shape[1]
    n_parameters = basis.shape[1]
    information = np.zeros((n_parameters, n_parameters))
    for k in range(n_parameters):
        information[k, k] = prior_n
    gamma = np.zeros(n_parameters)
    log_variances = np.zeros(n_samples)
    weights = np.empty(n_samples)
    rows = np.empty(n_samples, dtype=np.int64)
    gene_basis = np.empty((n_coefficients, n_samples))
    column = np.empty(n_samples)
    residuals = np.empty(n_samples)
    leverages = np.empty(n_samples)
    score = np.empty(n_parameters)
    spread = np.empty(n_parameters)
    factor = np.empty((n_parameters, n_parameters))
    step = np.empty(n_parameters)
    shared = weight_rows.shape[0] == 1
    used = 0
    for g in range(n_features):
        w = 0 if shared else g
        for i in range(n_samples):
            weights[i] = math.exp(-log_variances[i]) * weight_rows[w, i]
        m, rank = _project_gene(
            design,
            responses[g],
            weights,
            tolerance,
            rows,
            gene_basis,
            column,
            residuals,
            leverages,
        )
        df = m - rank
        if df < _MIN_RESIDUAL_DF:
            continue
        total = 0.0
        for t in range(m):
            total += residuals[t] * residuals[t]
        variance = total / df
        if variance < _ZERO_VARIANCE:
            continue
        used += 1
        score[:] = 0.0
        spread[:] = 0.0
        for t in range(m):
            i = rows[t]
            remaining = 1.0 - leverages[t]
            z = residuals[t] * residuals[t] / variance - remaining
            for k in range(n_parameters):
                score[k] += basis[i, k] * z
                spread[k] += basis[i, k] * remaining
            # The diagonal approximation of the information, lower triangle.
            for k in range(n_parameters):
                weighted = basis[i, k] * remaining
                for j in range(k + 1):
                    information[k, j] += weighted * basis[i, j]
        for k in range(n_parameters):
            for j in range(k + 1):
                information[k, j] -= spread[k] * spread[j] / df
        for k in range(n_parameters):
            for j in range(k + 1):
                factor[k, j] = information[k, j]
        if not _cholesky_solve(factor, score, step):
            return gamma, used, True
        for k in range(n_parameters):
            gamma[k] += step[k]
        for i in range(n_samples):
            total = 0.0
            for k in range(n_parameters):
                total += basis[i, k] * gamma[k]
            log_variances[i] = total
    return gamma, used, False


@jit
def _weighted_reml_sums(
    responses: FloatArray,
    weight_rows: FloatArray,
    design: FloatArray,
    log_variances: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, FloatArray]:
    """REML score per sample and information per pair of samples, summed over genes
    with prior weights, every value observed and every weight positive.

    Every gene adds its information; a gene whose residual variance is below
    ``_ZERO_VARIANCE`` adds no score (found by probing).

    Returns:
        The summed z_g, and the lower triangle of the sum of
        ``V_g - (1 - h_g)(1 - h_g)' / d_g``.
    """
    n_features, n_samples = responses.shape
    n_coefficients = design.shape[1]
    score = np.zeros(n_samples)
    information = np.zeros((n_samples, n_samples))
    weights = np.empty(n_samples)
    rows = np.empty(n_samples, dtype=np.int64)
    gene_basis = np.empty((n_coefficients, n_samples))
    column = np.empty(n_samples)
    residuals = np.empty(n_samples)
    leverages = np.empty(n_samples)
    for g in range(n_features):
        for i in range(n_samples):
            weights[i] = math.exp(-log_variances[i]) * weight_rows[g, i]
        m, rank = _project_gene(
            design,
            responses[g],
            weights,
            tolerance,
            rows,
            gene_basis,
            column,
            residuals,
            leverages,
        )
        df = m - rank
        for s in range(m):
            for t in range(s + 1):
                product = 0.0
                for a in range(rank):
                    product += gene_basis[a, s] * gene_basis[a, t]
                complement = (1.0 if s == t else 0.0) - product
                remaining = (1.0 - leverages[s]) * (1.0 - leverages[t])
                information[rows[s], rows[t]] += (
                    complement * complement - remaining / df
                )
        total = 0.0
        for t in range(m):
            total += residuals[t] * residuals[t]
        variance = total / df
        if variance < _ZERO_VARIANCE:
            continue
        for t in range(m):
            score[rows[t]] += residuals[t] * residuals[t] / variance - (
                1.0 - leverages[t]
            )
    return score, information


@jit
def _project_gene(
    design: FloatArray,
    response: FloatArray,
    weights: FloatArray,
    tolerance: float,
    rows: IntArray,
    gene_basis: FloatArray,
    column: FloatArray,
    residuals: FloatArray,
    leverages: FloatArray,
) -> tuple[int, int]:
    """Weighted least squares projection of one gene over its usable samples.

    The usable samples have an observed response and a positive weight. Their design
    rows and response, scaled by the square roots of the weights, are projected on
    the span of the design columns, taken in order and orthonormalised by
    Gram-Schmidt applied twice; a column whose part orthogonal to the kept columns
    has at most ``tolerance`` times its own norm is aliased, as by R's ``qr()``.

    Writes the usable samples to ``rows[:m]``, the orthonormal basis to
    ``gene_basis[:rank, :m]``, the residuals to ``residuals[:m]`` and the leverages
    to ``leverages[:m]``; ``column`` is workspace.

    Returns:
        m and the rank.
    """
    n_samples, n_coefficients = design.shape
    m = 0
    for i in range(n_samples):
        if not math.isnan(response[i]) and weights[i] > 0.0:
            rows[m] = i
            m += 1
    rank = 0
    for j in range(n_coefficients):
        if rank == m:
            break
        norm = 0.0
        for t in range(m):
            value = math.sqrt(weights[rows[t]]) * design[rows[t], j]
            column[t] = value
            norm += value * value
        norm = math.sqrt(norm)
        if norm == 0.0:
            continue
        for t in range(m):
            column[t] /= norm
        _orthogonalise(column, gene_basis, rank, m)
        remainder = 0.0
        for t in range(m):
            remainder += column[t] * column[t]
        remainder = math.sqrt(remainder)
        if remainder <= tolerance:
            continue
        for t in range(m):
            gene_basis[rank, t] = column[t] / remainder
        rank += 1
    for t in range(m):
        residuals[t] = math.sqrt(weights[rows[t]]) * response[rows[t]]
    _orthogonalise(residuals, gene_basis, rank, m)
    for t in range(m):
        total = 0.0
        for a in range(rank):
            total += gene_basis[a, t] * gene_basis[a, t]
        leverages[t] = total
    return m, rank


@jit
def _orthogonalise(
    vector: FloatArray, gene_basis: FloatArray, rank: int, m: int
) -> None:
    """Remove from ``vector[:m]`` its projection on ``gene_basis[:rank, :m]``, twice."""
    for _ in range(2):
        for a in range(rank):
            dot = 0.0
            for t in range(m):
                dot += gene_basis[a, t] * vector[t]
            for t in range(m):
                vector[t] -= dot * gene_basis[a, t]


@jit
def _cholesky_solve(
    matrix: FloatArray, right: FloatArray, solution: FloatArray
) -> bool:
    """Solve a symmetric positive definite system from its lower triangle.

    ``matrix`` is overwritten by its Cholesky factor. Returns False, leaving
    ``solution`` undefined, when a pivot is not positive beyond rounding error.
    """
    n = matrix.shape[0]
    largest = 0.0
    for k in range(n):
        largest = max(largest, abs(matrix[k, k]))
    for k in range(n):
        total = matrix[k, k]
        for j in range(k):
            total -= matrix[k, j] * matrix[k, j]
        if not total > _SINGULAR * n * largest:
            return False
        pivot = math.sqrt(total)
        matrix[k, k] = pivot
        for i in range(k + 1, n):
            total = matrix[i, k]
            for j in range(k):
                total -= matrix[i, j] * matrix[k, j]
            matrix[i, k] = total / pivot
    for i in range(n):
        total = right[i]
        for j in range(i):
            total -= matrix[i, j] * solution[j]
        solution[i] = total / matrix[i, i]
    for i in range(n - 1, -1, -1):
        total = solution[i]
        for j in range(i + 1, n):
            total -= matrix[j, i] * solution[j]
        solution[i] = total / matrix[i, i]
    return True
