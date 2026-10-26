"""Negative binomial generalized linear models fitted to every gene, as edgeR's glmFit.

For the count y_gi of gene g in sample i, the model is

    y_gi ~ NB(mu_gi, phi_gi),    var(y_gi) = mu_gi + phi_gi mu_gi^2,
    log(mu_gi) = x_i' beta_g + o_gi,

with x_i the design row of sample i, phi_gi a known dispersion and o_gi an offset,
usually the log effective library size log(N_i f_i) (McCarthy, Chen and Smyth 2012).
The coefficients beta_g, on the natural log scale, maximize the log-likelihood weighted
by prior weights w_gi; equivalently they minimize the deviance (Dunn and Smyth 2018)

    D_g = sum_i w_gi d(y_gi, mu_gi, phi_gi),
    d(y, mu, phi) = 2 [y log(y / mu) - (y + 1 / phi) log((1 + phi y) / (1 + phi mu))],

whose limit at phi = 0 is the Poisson unit deviance 2 [y log(y / mu) - (y - mu)]. The
log-likelihood is concave in beta_g, so its maximum is unique when it exists.

Each gene is fitted as edgeR's documentation describes ``glmFit``:

- When the rows of the design take as many distinct values as it has columns, the
  design is a one-way layout in some parametrization: group means, groups against a
  reference, an intercept alone, or one coefficient per sample. Each group's mean is
  then fitted on its own (edgeR's ``mglmOneGroup`` and ``mglmOneWay``), starting from
  the closed form of the Poisson fit, log(sum w y / sum w exp(o)), which is the answer
  when the group's offsets and dispersions are equal, and solving the group's score
  equation by Newton's method kept inside a bracket, to the last bits. The coefficients
  solve the distinct rows of the design for the group means.
- Other designs are fitted by Newton's method with Levenberg (1944) damping, from the
  fit of one group of all samples, edgeR's default start (the null model). The iteration
  works with an orthonormal basis Q of the design's column space, design = Q R, which
  McCarthy, Chen and Smyth (2012) describe for ``glmFit``, so that neither the scale of
  the columns nor how nearly collinear they are affects it; only the coefficients,
  R^(-1) times those of the basis, carry the conditioning of the design. Each step
  solves (X' W X) step = X' s, with s the scores and W the information of the
  observations, from the triangular factor of a Householder QR factorization (Golub and
  Van Loan 2013) of W^(1/2) X and with X' s formed directly; a least squares solution
  would carry the rounding of residuals W^(-1/2) s that can be far larger than their
  rows, as for zero counts with a large dispersion times mean. A column whose part
  orthogonal to the columns before it has a norm of at most 1e-11 times its own is
  aliased and not moved: a step along it would be rounding. Zero counts whose fitted
  values are below 1e-30 of the largest are left out of the fit, since the deviance no
  longer depends on them.
- A step that would increase the deviance is damped, by adding a multiple of the
  diagonal of the information matrix (Marquardt's scaling, Marquardt 1963), until it
  does not. A step is shortened so that it raises no log fitted value, and lowers none
  of a positive count, by more than 2, where the quadratic model of the step can fail;
  lowers no fitted value of a zero count below a thousandth of that 1e-30 level, beyond
  which lowering it further would only enlarge the coefficients, and with them the
  rounding of every fitted value; and changes no log fitted value by more than 1e3. A
  whole step that lowers the deviance by more than 1.1 times what it predicts is doubled
  while the deviance keeps falling: so it does where fitted values of zero counts tend
  to zero, and the deviance falls exponentially. The weights of each gene are divided by
  a power of two that brings the largest into [0.5, 1), which changes no fitted value,
  so that the tolerances refer to weights of order one.
- The steps are Newton's: they use the observed information where edgeR's
  documentation describes Fisher scoring, which uses the expected information. For
  this model the observed information, w mu (1 + phi y) / (1 + phi mu)^2 for each
  observation, is positive too, and it is the exact curvature of the log-likelihood.
  The expected information, w mu / (1 + phi mu), misjudges it by the factor
  (1 + phi y) / (1 + phi mu), so that with dispersions and weights that differ between
  observations Fisher scoring can stop far from the maximum, as edgeR's fits do on
  such genes (below). Both methods have the same maximum.
- A fit stops, as converged, when a step would change no log fitted value by more than
  1e-10 (leaving out zero counts below the 1e-30 level), when the decrease of the
  deviance that it predicts is below 1e-16 of the deviance, or when the deviance has
  fallen below 1e-16 of its starting value, as it does where every count can be fitted
  exactly in the limit. That last step is taken too unless it would raise the
  deviance by more than rounding: 1e-12 of it plus twice its rounding error, which the
  linear predictors' rounding sets, the same slack with which any step is accepted. A
  fit does not converge if its deviance is infinite, or if the deviance still falls
  along an aliased direction. A fit converges in about 4 to 8 steps when its maximum
  is finite.

Counts of zero:

- A group of a one-way layout whose counts are all zero has fitted values of zero. Its
  log mean, minus infinity, is represented by -1e8, as edgeR represents it (found by
  probing), so that the coefficients stay finite and reproduce the fitted values;
  coefficients that involve such a group are therefore near -1e8 or 1e8.
- In other designs the maximum lies at infinity when the fitted values of some samples
  with zero counts can tend to zero while the others stay, as when a group's counts
  are all zero, or a count is the only one at one end of a covariate. Each step then
  lowers those fitted values by a factor of e or more, until the decrease the step
  predicts falls below 1e-16 of the deviance: typically 10 to 20 steps, after which the
  fitted values that tend to zero are below about 1e-15 of the largest and the deviance
  is at its limit to rounding. The coefficients that reach them are large and depend on
  where the fit stops; the other fitted values and the deviance do not. edgeR stops
  earlier, ``glmFit`` with such fitted values near 1e-7 (found by probing).
- A gene without counts has fitted values and deviance of zero. Its coefficients are
  those of the constant log mean -1e8 when a vector of ones lies in the column space of
  the design, by the rank rule of R's qr() (an intercept, or group indicators that add
  up to one), as edgeR's one-way fit gives them; edgeR's Levenberg fit returns NA there
  (found by probing). Otherwise the iteration finds its maximum from coefficients of
  zero.

Deliberate differences from edgeR, found by probing:

- The deviance is exact. edgeR adds 1e-8 to each count and mean, so that the unit
  deviance of a zero count is up to about 5e-7 lower than it should be, and a small
  fraction of it for means below about 1e-8. Below a dispersion of 1e-4 it uses an
  approximation whose relative error grows with the dispersion times the mean: about
  1e-6 at 1e-3, 1e-4 at 1e-2 and 1e-2 at 0.1; from about 1 it is wrong by factors, and
  from about 10 it gives zero counts no deviance. From a dispersion times the mean of
  1e6 it uses the limit of the gamma distribution, which gives zero counts 1.6 to 2.6
  times their deviance.
- The fits of other designs than one-way layouts converge to the maximum. On the
  validation's data, edgeR's ``glmFit`` stops with coefficients up to about 3e-4 from
  it (1e-3 with a dispersion of 2e-5), 1e-2 with dispersions and weights per
  observation, and 1.6 with a covariate such as a date next to an intercept, where
  fitted values that should tend to zero stay near 1e2. ``mglmLevenberg`` with a
  lower tolerance (its ``tol``) stops about 2e-7 from it, but 2e-3 with dispersions
  and weights per observation, 2e-6 with sparse counts, 1e-5 with a date or for counts
  near 1e8, and 2e-3 with a dispersion of 2e-5, where the errors of its deviance are
  largest. With dispersions and weights per observation, ``glmFit`` also reports for
  a few genes a deviance that is not that of its own fitted values.
- Inputs that edgeR runs on without end, infinite dispersions or offsets, raise
  errors here.

Known limits:

- Where weights differ by many orders of magnitude between samples, the coefficients
  that only the samples of small weight determine are found to a relative accuracy of
  about 1e-16 times the factor between the weights (1e-6 for a factor of 1e10), as
  rounding in the scores of the other samples, which carry most of the information,
  swamps theirs. Beyond a factor of about 1e14 those coefficients are not determined,
  although the fit can report convergence. edgeR is less accurate still (found by
  probing).
- Where a zero count's dispersion times mean is so large, above about 1e22, that its
  information is below rounding next to that of the other samples while its deviance
  still falls with its fitted value, its direction is aliased from the start and the
  fit is reported as not converged.

References:
    McCarthy DJ, Chen Y, Smyth GK (2012). Differential expression analysis of
    multifactor RNA-Seq experiments with respect to biological variation. Nucleic Acids
    Research 40, 4288-4297. doi:10.1093/nar/gks042

    Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful
    differential analysis of sequencing data with expanded functionality and improved
    support for small counts and larger datasets. Nucleic Acids Research 53, gkaf018.
    doi:10.1093/nar/gkaf018

    Phipson B (2013). Empirical Bayes modelling of expression profiles and their
    associations. PhD thesis, University of Melbourne.
    https://hdl.handle.net/11343/38162

    Dunn PK, Smyth GK (2018). Generalized Linear Models With Examples in R. Springer,
    New York. doi:10.1007/978-1-4419-0118-7

    Golub GH, Van Loan CF (2013). Matrix Computations, 4th edition. Johns Hopkins
    University Press, Baltimore.

    Levenberg K (1944). A method for the solution of certain non-linear problems in
    least squares. Quarterly of Applied Mathematics 2, 164-168. doi:10.1090/qam/10666

    Marquardt DW (1963). An algorithm for least-squares estimation of nonlinear
    parameters. Journal of the Society for Industrial and Applied Mathematics 11,
    431-441. doi:10.1137/0111030
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import prange

from cytepool._counts import (
    as_counts,
    dispersion_matrix,
    offset_matrix,
    real_values,
    weight_matrix,
)
from cytepool._jit import by_chunks, jit, thread_limit
from cytepool._linear_algebra import RANK_TOLERANCE, householder_qr
from cytepool._masked_arrays import read_only, reject_masked
from cytepool._options import non_negative_number, positive_integer
from cytepool.design._design_matrices import (
    design_matrix,
    refuse_random_effects,
    require_full_rank,
)
from cytepool.stats._negative_binomial import (
    linear_predictors,
    unit_deviance,
    unit_deviances,
)

__all__ = [
    "NegativeBinomialFit",
    "fit_negative_binomial_model",
    "negative_binomial_deviance",
    "negative_binomial_unit_deviance",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

# edgeR's stand-in for the log mean of a group whose counts are all zero, minus
# infinity (found by probing). Its exponential is zero for any offset below 1e8.
_ZERO_LOG_MEAN = -1e8
# A fit stops when its Newton step would change no log fitted value by more than
# _STEP_TOLERANCE, or would lower the deviance by less than _DEVIANCE_TOLERANCE times
# the deviance, or when the deviance has fallen below _DEVIANCE_TOLERANCE times its
# starting value. Fitted values of zero counts below _NEGLIGIBLE of the largest are left
# out of the fit: the deviance no longer depends on them.
_STEP_TOLERANCE = 1e-10
_DEVIANCE_TOLERANCE = 1e-16
_NEGLIGIBLE = 1e-30
# A column of the weighted design whose part orthogonal to the columns before it has a
# norm of at most this times its own is aliased: steps along it would carry the rounding
# of the factorization, about the number of samples times epsilon relative to the
# column, magnified by the inverse of this ratio.
_ALIASING = 1e-11
# A step is shortened so that it raises no log fitted value, and lowers none of a
# positive count, by more than _LARGEST_STEP; lowers no fitted value of a zero count
# below _BELOW_NEGLIGIBLE times the negligible level; and changes no log fitted value
# by more than _LARGEST_CHANGE.
_LARGEST_STEP = 2.0
_BELOW_NEGLIGIBLE = 1e-3
_LARGEST_CHANGE = 1e3
# A step may raise the deviance by this fraction of it, plus twice the deviance's
# rounding error, and still be taken.
_ROUNDING_SLACK = 1e-12
# A whole Newton step that lowers the deviance by more than this times the decrease it
# predicts is doubled while the deviance keeps falling.
_EXPANSION = 1.1
# A score counts when it exceeds its rounding error, this multiple of epsilon times the
# sum of the absolute values of its terms, plus the rounding it inherits.
_SCORE_ROUNDING = 64.0
# Levenberg damping: the first value tried after a step that raises the deviance, the
# factor between successive values, the value below which it is dropped again, and the
# value at which the fit gives up.
_FIRST_DAMPING = 1e-3
_DAMPING_FACTOR = 10.0
_SMALLEST_DAMPING = 1e-8
_LARGEST_DAMPING = 1e10
# Newton steps of a group's log mean are at most this long.
_LARGEST_GROUP_STEP = 5.0
_EPSILON = 2.220446049250313e-16


@dataclass(frozen=True, eq=False, repr=False)
class NegativeBinomialFit:
    """Negative binomial generalized linear models of many genes, one design.

    Rows of the per-gene arrays are genes, in the order of the columns of the counts;
    arrays with a sample axis are samples x genes, as the counts are. The arrays are
    read-only.

    Attributes:
        coefficients: ``(n_features, n_coefficients)`` maximum likelihood estimates,
            on the natural log scale: ``fitted_values`` is ``exp(design @
            coefficients.T + offsets)``. edgeR reports them as
            ``unshrunk.coefficients``. A coefficient whose estimate is infinite is
            large and finite instead; see the module description.
        shrunk_coefficients: ``(n_features, n_coefficients)`` coefficients estimated
            after adding ``prior_count`` to the counts as edgeR's ``predFC`` does
            (Phipson 2013), which shrinks log fold changes towards zero and keeps them
            finite (edgeR's ``coefficients``). The same as ``coefficients`` when
            ``prior_count`` is 0.
        fitted_values: ``(n_samples, n_features)`` fitted means.
        deviance: ``(n_features,)`` deviances: ``negative_binomial_deviance`` of the
            counts at the fitted values.
        df_residual: ``(n_features,)`` residual degrees of freedom, the number of
            samples minus the number of coefficients, as floats.
        converged: ``(n_features,)`` whether each fit met its convergence criterion
            (and, with a prior count, so did the fit of the counts with the prior
            count added).
        iterations: ``(n_features,)`` Newton steps of each fit of the counts as
            given; for a one-way layout, the most that any group needed.
        offsets: ``(n_samples, n_features)`` offsets of the fits, a read-only view
            that repeats them along any axis where they do not change.
        dispersion: ``(n_samples, n_features)`` dispersions of the fits, a view as
            ``offsets`` is.
        weights: ``(n_samples, n_features)`` prior weights of the fits, a view as
            ``offsets`` is; ones when none were given.
        design: ``(n_samples, n_coefficients)`` design matrix.
        coefficient_names: Names of the coefficients, the design's columns.
        prior_count: The average count added to each observation for
            ``shrunk_coefficients``.
    """

    coefficients: FloatArray
    shrunk_coefficients: FloatArray
    fitted_values: FloatArray
    deviance: FloatArray
    df_residual: FloatArray
    converged: BoolArray
    iterations: IntArray
    offsets: FloatArray
    dispersion: FloatArray
    weights: FloatArray
    design: FloatArray
    coefficient_names: tuple[str, ...]
    prior_count: float

    def __repr__(self) -> str:
        n_samples, n_features = self.fitted_values.shape
        n_converged = int(np.count_nonzero(self.converged))
        return (
            f"NegativeBinomialFit(n_features={n_features}, n_samples={n_samples}, "
            f"coefficient_names={self.coefficient_names!r}, "
            f"converged={n_converged} of {n_features})"
        )


def fit_negative_binomial_model(
    counts: npt.ArrayLike,
    design: Any,
    *,
    dispersion: float | npt.ArrayLike,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    offsets: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    prior_count: float = 0.125,
    coefficient_names: Sequence[str] | None = None,
    max_iterations: int = 200,
    n_threads: int | None = None,
) -> NegativeBinomialFit:
    """Fit a negative binomial log-linear model to every gene, as edgeR's glmFit.

    Every gene is fitted with the same design, with its own dispersions, offsets and
    weights; the module description gives the model and the methods. The dispersions
    are taken as known: give estimates, such as
    :func:`~cytepool.stats.quasi_likelihood.common_dispersion`, or a
    value from similar data (edgeR's User's Guide gives biological coefficients of
    variation of 0.4 for human samples, 0.1 for genetically identical model organisms
    and 0.01 for technical replicates; the dispersion is their square).

    Args:
        counts: ``(n_samples, n_features)`` counts: finite and non-negative, integer or
            not.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank, an
            object with ``matrix`` and ``columns`` attributes such as a ``Design``, or
            a table with ``columns``. A ``Design`` with random effects is refused.
        dispersion: Negative binomial dispersions, finite and at least 0 (0 gives the
            Poisson model): one value, one per gene ``(n_features,)``, or one per
            observation ``(n_samples, n_features)``.
        library_sizes: Library size of each sample, positive; defaults to the total
            counts. The offsets are then ``log(library_sizes * norm_factors)``.
        norm_factors: Normalization factor of each sample, such as TMM factors from
            :func:`~cytepool.normalization.tmm_factors`, positive; defaults to ones.
        offsets: Offsets on the natural log scale, in place of library sizes and
            normalization factors: one value, one per sample ``(n_samples,)``, or one
            per observation ``(n_samples, n_features)``. Finite.
        weights: Prior weights, positive and finite: one value, one per sample
            ``(n_samples,)``, or one per observation ``(n_samples, n_features)``. They
            multiply each observation's log-likelihood.
        prior_count: Average count added to each observation for
            ``shrunk_coefficients``, at least 0. Each sample gets ``prior_count``
            times its library size over the mean library size, and its library size
            grows by twice that (edgeR's ``addPriorCount``), the library sizes being
            ``exp(offsets)``, per gene when the offsets differ between genes.
        coefficient_names: Names of the design columns, in order. Defaults to the
            design's column names when it has them, else ``x1, x2, ...``. They may
            rename the design's columns, but not give a column the name of another.
        max_iterations: The most Newton steps of a gene's fit (of each group's, for
            a one-way layout), at least 1. A fit that needs more is reported in
            ``converged``.
        n_threads: The most threads for fitting genes in parallel. ``None`` uses
            numba's setting for the calling thread (``numba.set_num_threads``). The
            result does not depend on it, and with one thread no parallel region runs.

    Returns:
        The fit of every gene, with its coefficients, fitted values, deviance,
        residual degrees of freedom and how it converged, and the offsets,
        dispersions and weights it used.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If shapes disagree, counts are negative or not finite, the design
            is not finite, has random effects or is not of full column rank (the
            message names the aliased columns), a dispersion is negative or not
            finite, an offset is not finite, a weight is not positive and finite, both
            offsets and library sizes or normalization factors are given, a library
            size is zero, or ``prior_count`` or ``max_iterations`` is out of range.
    """
    with thread_limit(n_threads):
        y = as_counts(counts)
        n_samples, n_features = y.shape
        x, names = design_matrix(design, coefficient_names)
        refuse_random_effects(design, "fit_negative_binomial_model")
        if x.shape[0] != n_samples:
            raise ValueError(
                f"counts have {n_samples} rows (samples) but the design has "
                f"{x.shape[0]}; both need one row per sample"
            )
        require_full_rank(x, names)
        phi = dispersion_matrix(dispersion, n_samples, n_features)
        log_sizes = offset_matrix(offsets, library_sizes, norm_factors, y)
        w = weight_matrix(weights, n_samples, n_features)
        prior = non_negative_number(prior_count, "prior_count")
        reject_masked(max_iterations, "max_iterations")
        limit = positive_integer(
            max_iterations, "max_iterations", expected="a positive integer"
        )
        layout = _layout(x)

        counts_by_gene = np.ascontiguousarray(y.T)
        coefficients, fitted, deviance, iterations, converged = _fit(
            counts_by_gene, log_sizes, phi, w, layout, limit
        )
        if prior > 0.0:
            augmented, augmented_offsets = _with_prior_counts(
                counts_by_gene, log_sizes, prior
            )
            shrunk, _, _, _, shrunk_converged = _fit(
                augmented, augmented_offsets, phi, w, layout, limit
            )
            converged &= shrunk_converged
        else:
            shrunk = coefficients

    shape = (n_samples, n_features)
    return NegativeBinomialFit(
        coefficients=read_only(coefficients),
        shrunk_coefficients=read_only(shrunk),
        fitted_values=read_only(fitted.T),
        deviance=read_only(deviance),
        df_residual=read_only(np.full(n_features, float(n_samples - x.shape[1]))),
        converged=read_only(converged),
        iterations=read_only(iterations),
        offsets=np.broadcast_to(log_sizes.T, shape),
        dispersion=np.broadcast_to(phi.T, shape),
        weights=np.broadcast_to(w.T, shape),
        design=read_only(x),
        coefficient_names=names,
        prior_count=prior,
    )


def negative_binomial_unit_deviance(
    counts: npt.ArrayLike, means: npt.ArrayLike, dispersion: npt.ArrayLike
) -> Any:
    """Unit deviances of the negative binomial distribution.

    d(y, mu, phi) = 2 [y log(y / mu) - (y + 1 / phi) log((1 + phi y) / (1 + phi mu))],
    with y log(y / mu) = 0 at y = 0; at phi = 0 it is the Poisson unit deviance
    2 [y log(y / mu) - (y - mu)], and as phi mu grows it tends to the gamma unit
    deviance (2 / phi) [y / mu - 1 - log(y / mu)]. It is computed without
    cancellation in every regime: as a series in (y - mu) / mu near y = mu, and from
    logarithms of ratios elsewhere, to a relative error of about 1e-14 over counts and
    means from 1e-30 to 1e30 and dispersions from 0 to 1e12. A count above a mean of 0
    gives infinity.

    If mu and phi are the true mean and dispersion, the unit deviance is
    approximately chi-squared on one degree of freedom. edgeR's
    ``nbinomUnitDeviance`` computes the same quantity with an offset and
    approximations; see the module description. The arguments broadcast with numpy's
    rules: with counts of shape ``(n_samples, n_features)``, one dispersion per gene
    has shape ``(n_features,)``.

    Args:
        counts: Counts, finite and non-negative.
        means: Means, finite and non-negative.
        dispersion: Dispersions, finite and non-negative.

    Returns:
        A float64 array of the broadcast shape, or a Python float when every argument
        is a scalar.

    Raises:
        TypeError: If an argument is not real-valued, or is a numpy masked array.
        ValueError: If an argument is negative or not finite, or the shapes do not
            broadcast.
    """
    y = _non_negative_array(counts, "counts")
    mu = _non_negative_array(means, "means")
    phi = _non_negative_array(dispersion, "dispersion")
    try:
        shape = np.broadcast_shapes(y.shape, mu.shape, phi.shape)
    except ValueError:
        raise ValueError(
            f"counts {y.shape}, means {mu.shape} and dispersion {phi.shape} do not "
            "broadcast to one shape"
        ) from None
    flat = [
        np.ascontiguousarray(np.broadcast_to(a, shape)).ravel() for a in (y, mu, phi)
    ]
    result: FloatArray = unit_deviances(*flat).reshape(shape)
    if result.ndim == 0:
        return float(result)
    return result


def negative_binomial_deviance(
    counts: npt.ArrayLike,
    means: npt.ArrayLike,
    dispersion: npt.ArrayLike,
    *,
    weights: npt.ArrayLike | None = None,
) -> FloatArray:
    """Deviance of each gene: the weighted sum of its unit deviances, as edgeR's
    nbinomDeviance.

    Args:
        counts: ``(n_samples, n_features)`` counts, finite and non-negative.
        means: ``(n_samples, n_features)`` means, finite and non-negative.
        dispersion: Dispersions, finite and non-negative: one value, one per gene
            ``(n_features,)``, or one per observation ``(n_samples, n_features)``.
        weights: Weights, finite and non-negative: one value, one per sample
            ``(n_samples,)`` or one per observation ``(n_samples, n_features)``; ones
            by default. An observation of weight zero adds nothing, even where its unit
            deviance is infinite.

    Returns:
        ``(n_features,)`` deviances, each sum_i w_i d(y_i, mu_i, phi_i) with d from
        :func:`negative_binomial_unit_deviance`.

    Raises:
        TypeError: If an argument is not real-valued, or is a numpy masked array.
        ValueError: If an argument is negative or not finite, or a shape is not one
            of those listed.
    """
    y = _non_negative_array(counts, "counts")
    mu = _non_negative_array(means, "means")
    if y.ndim != 2:
        raise ValueError(
            f"counts must be a 2-D array of samples x features, got shape {y.shape}"
        )
    if mu.shape != y.shape:
        raise ValueError(
            f"means must have the shape of counts {y.shape}, got {mu.shape}"
        )
    n_samples, n_features = y.shape
    phi = _non_negative_array(dispersion, "dispersion")
    if phi.shape not in ((), (n_features,), y.shape):
        raise ValueError(
            "dispersion must be one value, one per feature "
            f"({n_features},) or one per observation {y.shape}, got shape {phi.shape}"
        )
    unit = negative_binomial_unit_deviance(y, mu, phi)
    if weights is not None:
        w = _non_negative_array(weights, "weights")
        if w.shape == (n_samples,):
            w = w[:, np.newaxis]
        elif w.shape not in ((), y.shape):
            raise ValueError(
                f"weights must be one value, one per sample ({n_samples},) or one per "
                f"observation {y.shape}, got shape {w.shape}"
            )
        unit = np.where(w > 0.0, unit, 0.0) * w
    deviance: FloatArray = np.sum(unit, axis=0)
    return deviance


# Arguments.


def _non_negative_array(values: Any, name: str) -> FloatArray:
    array = real_values(values, name)
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(array) & (array >= 0.0))
    if bad.any():
        raise ValueError(
            f"{name} must be finite and non-negative; "
            f"{int(np.count_nonzero(bad))} values are not"
        )
    return array


# The design.


@dataclass(frozen=True)
class _Layout:
    """How the genes of one design are fitted.

    Attributes:
        one_way: Whether the design's rows take as many distinct values as it has
            columns, so that each group (distinct row) is fitted on its own.
        members: The samples of each group, consecutively; all samples, as one group,
            for other designs.
        group_starts: Where each group starts in ``members``, and the end.
        group_inverse: For a one-way layout, the inverse of the matrix of distinct
            rows, which maps group log means to coefficients; empty otherwise.
        basis: The design the iteration fits: for other designs than one-way
            layouts, an orthonormal basis Q of its column space, with design = Q R.
        triangular: R, which maps the coefficients g of the basis to those of the
            design, R b = g; the identity for a one-way layout.
        start: Coefficients of the basis whose linear predictor is the projection of
            a vector of ones on the column space, the direction of the null model.
        constant: Coefficients c of the design with design @ c as close as possible
            to a vector of ones, in whole numbers when those give it exactly: a gene
            without counts has coefficients -1e8 c.
        has_constant: Whether a vector of ones lies in the column space, by the rank
            rule of R's qr().
    """

    one_way: bool
    members: IntArray
    group_starts: IntArray
    group_inverse: FloatArray
    basis: FloatArray
    triangular: FloatArray
    start: FloatArray
    constant: FloatArray
    has_constant: bool


def _layout(x: FloatArray) -> _Layout:
    n_samples, n_coefficients = x.shape
    ones = np.ones(n_samples)
    constant, *_ = np.linalg.lstsq(x, ones, rcond=None)
    # Whole numbers, such as an intercept or group indicators that add up to one, are
    # kept exact, so that the coefficients of a gene without counts are -1e8 and 0.
    whole = np.round(constant)
    if np.array_equal(x @ whole, ones):
        constant = whole
    residual = float(np.linalg.norm(x @ constant - ones))
    # The rank rule of R's qr(), relative to the norm of the vector of ones.
    has_constant = residual <= RANK_TOLERANCE * math.sqrt(n_samples)
    distinct, groups = np.unique(x, axis=0, return_inverse=True)
    groups = groups.reshape(n_samples)
    if distinct.shape[0] == n_coefficients:
        order = np.argsort(groups, kind="stable")
        starts = np.searchsorted(groups[order], np.arange(n_coefficients + 1))
        inverse = np.linalg.solve(distinct, np.eye(n_coefficients))
        return _Layout(
            True,
            order.astype(np.int64),
            starts.astype(np.int64),
            np.ascontiguousarray(inverse),
            x,
            np.eye(n_coefficients),
            np.ascontiguousarray(constant),
            np.ascontiguousarray(constant),
            True,
        )
    # The iteration works with orthonormal columns, as McCarthy, Chen and Smyth
    # (2012) describe, so that neither the scale of the columns nor how nearly
    # collinear they are affects it; only the coefficients, R^(-1) times those of
    # the basis, carry the conditioning of the design.
    basis, triangular = np.linalg.qr(x)
    return _Layout(
        False,
        np.arange(n_samples, dtype=np.int64),
        np.array([0, n_samples], dtype=np.int64),
        np.empty((0, 0)),
        np.ascontiguousarray(basis),
        triangular,
        np.ascontiguousarray(basis.T @ ones),
        np.ascontiguousarray(constant),
        has_constant,
    )


def _with_prior_counts(
    counts: FloatArray, offsets: FloatArray, prior_count: float
) -> tuple[FloatArray, FloatArray]:
    """Counts and offsets (genes x samples) with prior counts added as edgeR's
    addPriorCount adds them.

    The library sizes are exp(offsets). Sample i of a gene gets prior_count times its
    library size over their mean, and its library size grows by twice that. The
    sizes are scaled by their largest value before they are averaged, so that large
    offsets do not overflow.
    """
    shift = offsets.max(axis=1, keepdims=True)
    scaled = np.exp(offsets - shift)
    prior = prior_count * scaled / scaled.mean(axis=1, keepdims=True)
    with np.errstate(over="ignore"):
        grown = np.where(
            offsets > 700.0,
            offsets + np.log1p(2.0 * prior * np.exp(-np.minimum(offsets, 709.0))),
            np.log(np.exp(np.minimum(offsets, 700.0)) + 2.0 * prior),
        )
    return np.ascontiguousarray(counts + prior), np.ascontiguousarray(grown)


def _fit(
    counts: FloatArray,
    offsets: FloatArray,
    dispersion: FloatArray,
    weights: FloatArray,
    layout: _Layout,
    max_iterations: int,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, BoolArray]:
    """Fit every gene (row of counts), on numba's threads when there are several.

    Returns:
        Coefficients (genes x coefficients), fitted values (genes x samples),
        deviances, iterations and whether each fit converged.
    """
    arguments = (
        counts,
        offsets,
        dispersion,
        weights,
        layout.basis,
        layout.members,
        layout.group_starts,
        layout.group_inverse,
        layout.start,
        layout.has_constant,
        layout.one_way,
        max_iterations,
    )
    fitted: tuple[FloatArray, FloatArray, FloatArray, IntArray, BoolArray] = by_chunks(
        _fit_in_parallel, _fit_in_order, arguments, counts.shape[0]
    )
    coefficients, values, deviance, iterations, converged = fitted
    if not layout.one_way:
        coefficients = np.linalg.solve(layout.triangular, coefficients.T).T
    if layout.has_constant:
        # Exactly -1e8 c, which the basis gives up to rounding.
        empty = ~counts.any(axis=1)
        coefficients[empty] = _ZERO_LOG_MEAN * layout.constant
    return coefficients, values, deviance, iterations, converged


# Compiled routines.


@jit
def _deviance(y: FloatArray, mu: FloatArray, phi: FloatArray, w: FloatArray) -> float:
    total = 0.0
    for i in range(y.size):
        total += w[i] * unit_deviance(y[i], mu[i], phi[i])
    return total


@jit
def _fill_row(matrix: FloatArray, g: int, out: FloatArray) -> None:
    """Row g of a genes x samples matrix that may repeat one row or one column."""
    row = g if matrix.shape[0] > 1 else 0
    if matrix.shape[1] > 1:
        for i in range(out.size):
            out[i] = matrix[row, i]
    else:
        for i in range(out.size):
            out[i] = matrix[row, 0]


@jit
def _group_log_mean(
    y: FloatArray,
    offsets: FloatArray,
    phi: FloatArray,
    w: FloatArray,
    members: IntArray,
    start: int,
    stop: int,
    max_iterations: int,
) -> tuple[float, int, bool]:
    """The log mean m of one group, whose samples are members[start:stop].

    Solves sum_i w_i (y_i - mu_i) / (1 + phi_i mu_i) = 0 with mu_i = exp(m + o_i) by
    Newton's method, from the Poisson solution, with every step kept inside the
    bracket of the root that the signs of the score give. The score decreases in m
    and is positive below the root, so the root is unique.

    Returns:
        The log mean (-1e8 when the group's counts are all zero), the number of
        Newton steps, and whether the last one was negligible.
    """
    total = 0.0
    top = -math.inf
    for k in range(start, stop):
        i = members[k]
        total += w[i] * y[i]
        top = max(top, offsets[i])
    if total == 0.0:
        return _ZERO_LOG_MEAN, 0, True
    size = 0.0
    for k in range(start, stop):
        i = members[k]
        size += w[i] * math.exp(offsets[i] - top)
    m = math.log(total) - (top + math.log(size))
    low = -math.inf
    high = math.inf
    for iteration in range(1, max_iterations + 1):
        score = 0.0
        information = 0.0
        for k in range(start, stop):
            i = members[k]
            mu = math.exp(m + offsets[i])
            denominator = 1.0 + phi[i] * mu
            score += w[i] * (y[i] - mu) / denominator
            information += (
                w[i] * mu * (1.0 + phi[i] * y[i]) / (denominator * denominator)
            )
        if score == 0.0:
            return m, iteration - 1, True
        if score > 0.0:
            low = m
        else:
            high = m
        step = score / information
        if not math.isfinite(step):
            step = _LARGEST_GROUP_STEP if score > 0.0 else -_LARGEST_GROUP_STEP
        step = min(max(step, -_LARGEST_GROUP_STEP), _LARGEST_GROUP_STEP)
        tolerance = 4.0 * _EPSILON * max(1.0, abs(m))
        if abs(step) <= tolerance:
            return m + step, iteration, True
        updated = m + step
        if not low < updated < high:
            # The step points towards the root, so it can only overshoot a bound
            # found before; both bounds are then finite.
            updated = 0.5 * (low + high)
            if high - low <= tolerance:
                return updated, iteration, True
        m = updated
    return m, max_iterations, False


@jit
def _fit_groups(
    y: FloatArray,
    offsets: FloatArray,
    phi: FloatArray,
    w: FloatArray,
    members: IntArray,
    group_starts: IntArray,
    group_inverse: FloatArray,
    max_iterations: int,
    log_means: FloatArray,
    beta: FloatArray,
    mu: FloatArray,
) -> tuple[float, int, bool]:
    """Fit a one-way layout: each group's log mean, then the coefficients."""
    n_groups = group_starts.size - 1
    most = 0
    converged = True
    for h in range(n_groups):
        m, iterations, done = _group_log_mean(
            y,
            offsets,
            phi,
            w,
            members,
            group_starts[h],
            group_starts[h + 1],
            max_iterations,
        )
        log_means[h] = m
        most = max(most, iterations)
        converged = converged and done
        for k in range(group_starts[h], group_starts[h + 1]):
            i = members[k]
            mu[i] = 0.0 if m == _ZERO_LOG_MEAN else math.exp(m + offsets[i])
    for j in range(n_groups):
        total = 0.0
        for h in range(n_groups):
            total += group_inverse[j, h] * log_means[h]
        beta[j] = total
    return _deviance(y, mu, phi, w), most, converged


@jit
def _means(
    design: FloatArray, beta: FloatArray, offsets: FloatArray, mu: FloatArray
) -> None:
    """mu = exp(design @ beta + offsets)."""
    linear_predictors(design, beta, offsets, mu)
    for i in range(design.shape[0]):
        mu[i] = math.exp(mu[i])


@jit
def _newton_system(
    y: FloatArray,
    mu: FloatArray,
    phi: FloatArray,
    w: FloatArray,
    design: FloatArray,
    beta: FloatArray,
    offsets: FloatArray,
    floor: float,
    work: FloatArray,
    scale: FloatArray,
    score: FloatArray,
    size: FloatArray,
) -> float:
    """The equations of a Newton step, column by column.

    Column j of ``work`` gets column j of W^(1/2) X, with W = w mu (1 + phi y) /
    (1 + phi mu)^2 the observed information of each observation, which is positive, and
    ``score`` gets X' s, with s = w (y - mu) / (1 + phi mu) the derivative of the
    log-likelihood with respect to each linear predictor; ``size`` gets
    sum_i |x_ij| (|s_i| + (p + 1) W_i (1 + |o_i| + sum_k |x_ik b_k|)), which bounds
    the rounding error of the score in units of epsilon: that of its terms, and that of
    the linear predictors, which reaches the score through the information. The Newton
    step solves
    (X' W X) step = X' s with the triangular factor of W^(1/2) X, and X' s is formed
    directly: as W^(1/2)' (W^(-1/2) s), a least squares solution would carry the
    rounding of residuals W^(-1/2) s that can be far larger than their rows, for zero
    counts whose dispersion times mean is large. Zero counts whose fitted values are
    below ``floor`` are left out, and so is the information of a positive count whose
    fitted value is zero. ``scale`` gets the diagonal of X' W X, floored at 1e-14 of its
    largest value so that damping reaches every direction.

    Returns:
        The rounding error of the deviance at these fitted values, from that of the
        linear predictors: 2 (p + 1) epsilon sum_i |s_i| (1 + |o_i| + sum_j |x_ij b_j|).
    """
    n_samples, n_coefficients = design.shape
    for j in range(n_coefficients):
        score[j] = 0.0
        size[j] = 0.0
    noise: float = 0.0
    for i in range(n_samples):
        m = mu[i]
        if m > 0.0 and (y[i] > 0.0 or m >= floor):
            denominator = 1.0 + phi[i] * m
            # Square roots taken one by one, so that neither a subnormal fitted value
            # nor a large one overflows.
            root = math.sqrt(w[i]) * math.sqrt(m)
            root *= math.sqrt(1.0 + phi[i] * y[i]) / denominator
            derivative = w[i] * (y[i] - m) / denominator
        else:
            root = 0.0
            derivative = w[i] * y[i]
        predictor = 1.0 + abs(offsets[i])
        for j in range(n_coefficients):
            predictor += abs(design[i, j] * beta[j])
        # The score of the sample carries the rounding of the score itself, and that
        # of the linear predictor through the information, root^2.
        terms = abs(derivative) + (n_coefficients + 1) * root * root * predictor
        for j in range(n_coefficients):
            work[i, j] = root * design[i, j]
            score[j] += design[i, j] * derivative
            size[j] += abs(design[i, j]) * terms
        noise += abs(derivative) * predictor
    largest = 0.0
    for j in range(n_coefficients):
        total = 0.0
        for i in range(n_samples):
            total += work[i, j] * work[i, j]
        scale[j] = total
        largest = max(largest, total)
    least = largest * 1e-14 if largest > 0.0 else 1.0
    for j in range(n_coefficients):
        scale[j] = max(scale[j], least)
    rounding: float = 2.0 * (n_coefficients + 1) * _EPSILON * noise
    return rounding


@jit
def _solve(
    work: FloatArray,
    n_coefficients: int,
    rank: int,
    kept: IntArray,
    score: FloatArray,
    size: FloatArray,
    z: FloatArray,
    rounding: FloatArray,
    step: FloatArray,
) -> tuple[float, float]:
    """Solve R' R step = score over the kept columns; aliased columns get 0.

    R' z = score is solved first, then R step = z; ``rounding`` gets the rounding
    error of each component of z.

    Returns:
        The decrease of the deviance that the step predicts, |z|^2, and the largest
        score of an aliased column along the part of it that the kept columns do not
        span, where that exceeds its rounding error (0 if none does).
    """
    for t in range(rank):
        j = kept[t]
        value, noise = _reduced_score(work, j, t, score, size, z, rounding)
        rounding[t] = noise / abs(work[t, j])
        z[t] = value / work[t, j]
    decrease = 0.0
    for t in range(rank):
        decrease += z[t] * z[t]
    for j in range(n_coefficients):
        step[j] = 0.0
    for t in range(rank - 1, -1, -1):
        total = z[t]
        for u in range(t + 1, rank):
            total -= work[t, kept[u]] * step[kept[u]]
        step[kept[t]] = total / work[t, kept[t]]
    reduced = 0.0
    t = 0
    for j in range(n_coefficients):
        if t < rank and kept[t] == j:
            t += 1
            continue
        value, noise = _reduced_score(work, j, t, score, size, z, rounding)
        if abs(value) > noise:
            reduced = max(reduced, abs(value))
    return decrease, reduced


@jit
def _reduced_score(
    work: FloatArray,
    j: int,
    count: int,
    score: FloatArray,
    size: FloatArray,
    z: FloatArray,
    rounding: FloatArray,
) -> tuple[float, float]:
    """The score of column j less its projections on the first ``count`` directions
    of the factorization times their components of z, and its rounding error: a
    multiple of epsilon times its terms, and the rounding of those components."""
    value = score[j]
    products = 0.0
    propagated = 0.0
    for u in range(count):
        product = work[u, j] * z[u]
        value -= product
        products += abs(product)
        propagated += abs(work[u, j]) * rounding[u]
    noise = _SCORE_ROUNDING * _EPSILON * (size[j] + products) + propagated
    return value, noise


@jit
def _largest_change(
    design: FloatArray, step: FloatArray, mu: FloatArray, floor: float
) -> float:
    """The largest change of a linear predictor, max |x_i' step|, over the samples
    whose fitted values mu_i are at least ``floor``; NaN for a NaN step."""
    n_samples, n_coefficients = design.shape
    largest = 0.0
    for i in range(n_samples):
        if mu[i] < floor:
            continue
        change = 0.0
        for j in range(n_coefficients):
            change += design[i, j] * step[j]
        if not abs(change) <= largest:
            largest = abs(change)
    return largest


@jit
def _longest(
    design: FloatArray, step: FloatArray, y: FloatArray, mu: FloatArray, floor: float
) -> float:
    """The largest multiple of ``step`` that the fit may take.

    It may raise no log fitted value, and lower none of a positive count, by more than
    2: far from the maximum the quadratic model of a Newton step can be poor, and a
    longer step would send fitted values beyond where it holds. It may lower the fitted
    value of a zero count to a thousandth of ``floor``, but no further: its deviance
    falls with it, so that fitted values that tend to zero reach the floor in few
    steps, while lowering them further would only enlarge the coefficients, and with
    them the rounding of every fitted value. It may change no log fitted value by more
    than 1e3, the only bound on zero counts already below ``floor``. NaN for a NaN
    step.
    """
    n_samples, n_coefficients = design.shape
    longest = math.inf
    for i in range(n_samples):
        change = 0.0
        for j in range(n_coefficients):
            change += design[i, j] * step[j]
        if change != change:
            return math.nan
        size = abs(change)
        if size == 0.0:
            continue
        bound = _LARGEST_CHANGE
        if y[i] > 0.0 or change > 0.0:
            if mu[i] >= floor or y[i] > 0.0:
                bound = _LARGEST_STEP
        elif mu[i] >= floor:
            bound = min(bound, math.log(mu[i] / floor) - math.log(_BELOW_NEGLIGIBLE))
        longest = min(longest, bound / size)
    return longest


@jit
def _expanded(
    design: FloatArray,
    beta: FloatArray,
    step: FloatArray,
    offsets: FloatArray,
    y: FloatArray,
    phi: FloatArray,
    w: FloatArray,
    longest: float,
    deviance: float,
    trial_beta: FloatArray,
    trial_mu: FloatArray,
    candidate_beta: FloatArray,
    candidate_mu: FloatArray,
) -> float:
    """Double a step that was taken whole while the deviance keeps falling and the
    step stays within ``longest``; ``trial_beta`` and ``trial_mu`` hold the step taken
    on entry, whose deviance is ``deviance``, and the longest on return.

    Returns:
        The deviance of the step taken.
    """
    n_coefficients = design.shape[1]
    length = 1.0
    best = deviance
    while 2.0 * length <= longest:
        for j in range(n_coefficients):
            candidate_beta[j] = beta[j] + 2.0 * length * step[j]
        _means(design, candidate_beta, offsets, candidate_mu)
        candidate = _deviance(y, candidate_mu, phi, w)
        if not candidate < best:
            break
        best = candidate
        length *= 2.0
        trial_beta[:] = candidate_beta
        trial_mu[:] = candidate_mu
    return best


@jit
def _shortened(
    design: FloatArray,
    beta: FloatArray,
    step: FloatArray,
    y: FloatArray,
    mu: FloatArray,
    floor: float,
    trial_beta: FloatArray,
) -> None:
    """trial_beta = beta + step, shortened to the longest step that :func:`_longest`
    allows when that is shorter."""
    shorten = min(1.0, _longest(design, step, y, mu, floor))
    for j in range(beta.size):
        trial_beta[j] = beta[j] + shorten * step[j]


@jit
def _fit_levenberg(
    y: FloatArray,
    offsets: FloatArray,
    phi: FloatArray,
    w: FloatArray,
    design: FloatArray,
    members: IntArray,
    start: FloatArray,
    has_constant: bool,
    max_iterations: int,
    beta: FloatArray,
    mu: FloatArray,
    work: FloatArray,
    scale: FloatArray,
    score: FloatArray,
    size: FloatArray,
    z: FloatArray,
    rounding: FloatArray,
    kept: IntArray,
    qr_norms: FloatArray,
    qr_aliased: BoolArray,
    qr_products: FloatArray,
    step: FloatArray,
    trial_beta: FloatArray,
    trial_mu: FloatArray,
    candidate_beta: FloatArray,
    candidate_mu: FloatArray,
) -> tuple[float, int, bool]:
    """Fit one gene by Newton's method with Levenberg damping.

    ``design`` is the orthonormal basis of the layout, and ``start`` the coefficients
    of its projection of a vector of ones.

    Returns:
        The deviance, the number of steps taken, and whether the fit converged.
    """
    n_samples, n_coefficients = design.shape
    counted = False
    for i in range(n_samples):
        if y[i] != 0.0:
            counted = True
            break
    if not counted and has_constant:
        for j in range(n_coefficients):
            beta[j] = _ZERO_LOG_MEAN * start[j]
        for i in range(n_samples):
            mu[i] = 0.0
        return 0.0, 0, True
    level = 0.0
    if counted:
        level = _group_log_mean(
            y, offsets, phi, w, members, 0, n_samples, max_iterations
        )[0]
    for j in range(n_coefficients):
        beta[j] = level * start[j]
    _means(design, beta, offsets, mu)
    deviance = _deviance(y, mu, phi, w)
    # The deviance at the start, or the first finite one: where every count can be
    # fitted exactly in the limit, the deviance itself tends to zero.
    reference = deviance
    damping = 0.0
    for iteration in range(max_iterations + 1):
        floor = _NEGLIGIBLE * _largest(mu)
        noise = _newton_system(
            y, mu, phi, w, design, beta, offsets, floor, work, scale, score, size
        )
        rank = householder_qr(
            work,
            n_samples,
            n_coefficients,
            _ALIASING,
            True,
            0.0,
            qr_norms,
            qr_aliased,
            kept,
            qr_products,
        )
        decrease, reduced = _solve(
            work, n_coefficients, rank, kept, score, size, z, rounding, step
        )
        moving = _largest_change(design, step, mu, floor)
        # The steps come from the scores, which rounding in the deviance does not
        # reach: it only widens the slack with which a step may raise the deviance.
        limit = _DEVIANCE_TOLERANCE * deviance
        slack = _ROUNDING_SLACK * deviance + 2.0 * noise
        negligible = deviance <= _DEVIANCE_TOLERANCE * reference
        if deviance < math.inf and (
            moving <= _STEP_TOLERANCE or decrease <= limit or negligible
        ):
            # Converged, unless the deviance still falls along an aliased direction,
            # whose information is below rounding: as for a zero count whose
            # dispersion times mean is so large that its deviance grows only with the
            # log of its fitted value.
            aliased = 2.0 * reduced > max(_DEVIANCE_TOLERANCE * reference, noise)
            if iteration == max_iterations or aliased:
                return deviance, iteration, not aliased
            # The last step still converges quadratically, and moves fitted values
            # that tend to zero further down: it is taken unless it would raise the
            # deviance by more than rounding.
            _shortened(design, beta, step, y, mu, floor, trial_beta)
            _means(design, trial_beta, offsets, trial_mu)
            trial_deviance = _deviance(y, trial_mu, phi, w)
            if trial_deviance <= deviance + slack:
                beta[:] = trial_beta
                mu[:] = trial_mu
                return trial_deviance, iteration + 1, True
            return deviance, iteration, True
        if iteration == max_iterations:
            return deviance, iteration, False
        while True:
            if damping > 0.0:
                # The damped step solves the equations with damping * scale added to
                # the diagonal of X' W X: the factorization gets p rows,
                # sqrt(damping * scale_j) in column j.
                _newton_system(
                    y,
                    mu,
                    phi,
                    w,
                    design,
                    beta,
                    offsets,
                    floor,
                    work,
                    scale,
                    score,
                    size,
                )
                for j in range(n_coefficients):
                    for k in range(n_coefficients):
                        work[n_samples + j, k] = 0.0
                    work[n_samples + j, j] = math.sqrt(damping * scale[j])
                rows = n_samples + n_coefficients
                rank = householder_qr(
                    work,
                    rows,
                    n_coefficients,
                    _ALIASING,
                    True,
                    0.0,
                    qr_norms,
                    qr_aliased,
                    kept,
                    qr_products,
                )
                _solve(work, n_coefficients, rank, kept, score, size, z, rounding, step)
            longest = _longest(design, step, y, mu, floor)
            length = min(1.0, longest)
            for j in range(n_coefficients):
                trial_beta[j] = beta[j] + length * step[j]
            _means(design, trial_beta, offsets, trial_mu)
            trial_deviance = _deviance(y, trial_mu, phi, w)
            # A fit that starts with an infinite deviance, a positive count whose
            # fitted value is zero, takes only steps that make it finite.
            if trial_deviance <= deviance + slack and trial_deviance < math.inf:
                if (
                    damping == 0.0
                    and length == 1.0
                    and deviance - trial_deviance > _EXPANSION * decrease
                ):
                    # The deviance fell by more than the Newton step predicts, as it
                    # does along a direction in which fitted values of zero counts
                    # tend to zero, where it falls exponentially: the step is doubled
                    # while the deviance keeps falling and the longest step allows.
                    trial_deviance = _expanded(
                        design,
                        beta,
                        step,
                        offsets,
                        y,
                        phi,
                        w,
                        longest,
                        trial_deviance,
                        trial_beta,
                        trial_mu,
                        candidate_beta,
                        candidate_mu,
                    )
                beta[:] = trial_beta
                mu[:] = trial_mu
                deviance = trial_deviance
                if not reference < math.inf:
                    reference = deviance
                damping /= _DAMPING_FACTOR
                if damping < _SMALLEST_DAMPING:
                    damping = 0.0
                break
            damping = _FIRST_DAMPING if damping == 0.0 else damping * _DAMPING_FACTOR
            if damping > _LARGEST_DAMPING:
                return deviance, iteration, False
    return deviance, max_iterations, False


@jit
def _largest(values: FloatArray) -> float:
    largest = 0.0
    for value in values:
        largest = max(largest, value)
    return largest


@jit
def _fit_genes(
    counts: FloatArray,
    offsets: FloatArray,
    dispersion: FloatArray,
    weights: FloatArray,
    design: FloatArray,
    members: IntArray,
    group_starts: IntArray,
    group_inverse: FloatArray,
    start: FloatArray,
    has_constant: bool,
    one_way: bool,
    max_iterations: int,
    first: int,
    last: int,
    coefficients: FloatArray,
    fitted: FloatArray,
    deviance: FloatArray,
    iterations: IntArray,
    converged: BoolArray,
) -> None:
    """Fit genes first..last-1 into the output arrays, with work arrays of their own."""
    n_samples, n_coefficients = design.shape
    o = np.empty(n_samples)
    phi = np.empty(n_samples)
    w = np.empty(n_samples)
    beta = np.empty(n_coefficients)
    log_means = np.empty(n_coefficients)
    # W^(1/2) X for a Newton step, with room for the rows of the damping, and the
    # workspace of its factorization.
    work = np.empty((n_samples + n_coefficients, n_coefficients))
    qr_norms = np.empty(n_coefficients)
    qr_aliased = np.empty(n_coefficients, dtype=np.bool_)
    qr_products = np.empty(n_coefficients)
    scale = np.empty(n_coefficients)
    score = np.empty(n_coefficients)
    size = np.empty(n_coefficients)
    z = np.empty(n_coefficients)
    rounding = np.empty(n_coefficients)
    kept = np.empty(n_coefficients, dtype=np.int64)
    step = np.empty(n_coefficients)
    trial_beta = np.empty(n_coefficients)
    trial_mu = np.empty(n_samples)
    candidate_beta = np.empty(n_coefficients)
    candidate_mu = np.empty(n_samples)
    for g in range(first, last):
        _fill_row(offsets, g, o)
        _fill_row(dispersion, g, phi)
        _fill_row(weights, g, w)
        # The weights are divided by a power of two that brings the largest into
        # [0.5, 1): exactly, so that the maximum does not change, while the
        # tolerances of the fit refer to weights of order one. The deviance is
        # multiplied back.
        largest = 0.0
        for i in range(n_samples):
            largest = max(largest, w[i])
        exponent = math.frexp(largest)[1]
        for i in range(n_samples):
            w[i] = math.ldexp(w[i], -exponent)
        mu = fitted[g]
        if one_way:
            gene_deviance, gene_iterations, done = _fit_groups(
                counts[g],
                o,
                phi,
                w,
                members,
                group_starts,
                group_inverse,
                max_iterations,
                log_means,
                beta,
                mu,
            )
        else:
            gene_deviance, gene_iterations, done = _fit_levenberg(
                counts[g],
                o,
                phi,
                w,
                design,
                members,
                start,
                has_constant,
                max_iterations,
                beta,
                mu,
                work,
                scale,
                score,
                size,
                z,
                rounding,
                kept,
                qr_norms,
                qr_aliased,
                qr_products,
                step,
                trial_beta,
                trial_mu,
                candidate_beta,
                candidate_mu,
            )
        for j in range(n_coefficients):
            coefficients[g, j] = beta[j]
        deviance[g] = math.ldexp(gene_deviance, exponent)
        iterations[g] = gene_iterations
        converged[g] = done


@jit(parallel=True)
def _fit_in_parallel(
    counts: FloatArray,
    offsets: FloatArray,
    dispersion: FloatArray,
    weights: FloatArray,
    design: FloatArray,
    members: IntArray,
    group_starts: IntArray,
    group_inverse: FloatArray,
    start: FloatArray,
    has_constant: bool,
    one_way: bool,
    max_iterations: int,
    n_chunks: int,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, BoolArray]:
    """The fits of :func:`_fit_genes`, in ``n_chunks`` runs of consecutive genes."""
    n_genes, n_samples = counts.shape
    coefficients = np.empty((n_genes, design.shape[1]))
    fitted = np.empty((n_genes, n_samples))
    deviance = np.empty(n_genes)
    iterations = np.empty(n_genes, dtype=np.int64)
    converged = np.empty(n_genes, dtype=np.bool_)
    for chunk in prange(n_chunks):
        _fit_genes(
            counts,
            offsets,
            dispersion,
            weights,
            design,
            members,
            group_starts,
            group_inverse,
            start,
            has_constant,
            one_way,
            max_iterations,
            chunk * n_genes // n_chunks,
            (chunk + 1) * n_genes // n_chunks,
            coefficients,
            fitted,
            deviance,
            iterations,
            converged,
        )
    return coefficients, fitted, deviance, iterations, converged


@jit
def _fit_in_order(
    counts: FloatArray,
    offsets: FloatArray,
    dispersion: FloatArray,
    weights: FloatArray,
    design: FloatArray,
    members: IntArray,
    group_starts: IntArray,
    group_inverse: FloatArray,
    start: FloatArray,
    has_constant: bool,
    one_way: bool,
    max_iterations: int,
) -> tuple[FloatArray, FloatArray, FloatArray, IntArray, BoolArray]:
    """The fits of :func:`_fit_in_parallel` on the calling thread alone."""
    n_genes, n_samples = counts.shape
    coefficients = np.empty((n_genes, design.shape[1]))
    fitted = np.empty((n_genes, n_samples))
    deviance = np.empty(n_genes)
    iterations = np.empty(n_genes, dtype=np.int64)
    converged = np.empty(n_genes, dtype=np.bool_)
    _fit_genes(
        counts,
        offsets,
        dispersion,
        weights,
        design,
        members,
        group_starts,
        group_inverse,
        start,
        has_constant,
        one_way,
        max_iterations,
        0,
        n_genes,
        coefficients,
        fitted,
        deviance,
        iterations,
        converged,
    )
    return coefficients, fitted, deviance, iterations, converged
