"""Gene-set tests of one contrast of a linear model: limma's camera and fry.

A gene set, such as a pathway, is tested as a unit for one contrast of the linear model
fitted to every gene. :func:`camera` is competitive: it asks whether the genes of a set
are more differentially expressed than the other genes, allowing for the correlation
between the genes of the set (Wu and Smyth 2012). :func:`fry` is self-contained: it asks
whether the genes of a set are differentially expressed at all, as the rotation test of
Wu et al. (2010) with the mean statistic does, without drawing rotations (Giner and
Smyth 2016). Results match limma 3.62.1 except where noted under "Deliberate
differences". The rotation tests, which draw rotations (limma's roast, mroast and
romer), are in :mod:`cytepool.stats.rotation_gene_set_tests`.

Effects:
    Both tests work with the effects of each gene's least squares fit (Wu et al. 2010,
    section 4.1). The design X, n samples by p coefficients, is reparametrised so that
    the contrast is its last coefficient: a contrast given as a column of the design
    moves that column last; a contrast given as weights c becomes the last column, X c /
    (c'c), after the columns X h_2, ..., X h_p, where h_1, ..., h_p are the columns of
    the Householder reflection H of c, so that the last coefficient is c'alpha (limma's
    ``contrastAsCoef``, found by probing). With the Householder QR decomposition (Golub
    and Van Loan 2013) of this design in R's sign convention, a gene's data y give the
    effects Q'y; the p-th is the contrast effect u_0, which divided by the residual
    standard deviation is the gene's t-statistic, and the last d = n - p are its
    residual effects u_1, ..., u_d, independent with variance sigma^2 under the model.
    With precision weights W each gene's data and design are whitened by W^(1/2) first.
    Every sample must be observed, with a positive weight, in every gene, so that all
    genes have the same residual degrees of freedom d.

    The residual effects are coordinates in a basis of the residual space, which the
    QR decomposition chooses. With observation weights each gene has a basis of its
    own, and the inter-gene correlation of camera and the statistics of fry depend on
    them; fry's standardizations other than "none" depend on the basis even without
    weights, through the largest effect. The bases are limma's: they depend on how the
    contrast is given, and a pivot of the decomposition that is zero up to rounding
    (below 1e-12 of its column), as in indicator designs with the samples sorted by
    group, counts as zero, which R's ``qr()`` gives the positive sign. limma computes
    most such zeros exactly, but leaves rounding noise in a few percent of genes, a
    different few in camera and in fry, so that their bases, and limma's results, change
    with the last bits of the data there; CytePool's do not (found by probing).

camera:
    Genes get moderated t-statistics, the contrast effect over the posterior standard
    deviation of :func:`~cytepool.stats.empirical_bayes.squeeze_variances` on
    min(d + d0, G d) degrees of freedom (as ``eBayes`` caps them, for G genes), turned
    into z-scores by Hill's (1970) approximation (Brophy 1987, equation 5), as limma's
    ``zscoreT(approx = TRUE, method = "hill")`` does (found by probing). With the mean
    z-score of a set of m genes and the mean and variance of all G z-scores, the test
    statistic is the two-sample t-statistic of the set against the other G - m genes
    with the variance of the set's mean inflated by VIF = 1 + (m - 1) rho (Wu and Smyth
    2012). The inter-gene correlation rho is a value given for every set, or estimated
    for each set from the residual effects: each gene's are divided by its residual
    standard deviation (its variance bounded below by 1e-8, found by probing), the m x d
    matrix is averaged over genes, and m/d times the sum of squares of that mean
    estimates the VIF. Unless negative correlations are allowed, rho enters the VIF as
    max(rho, 0); a given negative correlation that makes the VIF of a set negative,
    which no correlation matrix allows, gives the set a NaN p-value. The p-value is
    two-sided, from the t distribution on G - 2 degrees of freedom for a given
    correlation and on min(d, G - 2) for an estimated one (found by probing). The rank
    version replaces the t-statistic by the Wilcoxon-Mann-Whitney rank sum of the set's
    z-scores among all genes, with the variance of the rank sum when the set's values
    share a correlation rho (Wu and Smyth 2012, from Barry et al. 2008, equation 4.5),
    times Zar's (1999, section 8.10) correction for ties, 1 - sum(t^3 - t) / (G^3 - G),
    and a continuity correction of 1/2; it is referred to the normal distribution for a
    given correlation and to the t distribution on min(d, G - 2) degrees of freedom for
    an estimated one (found by probing).

fry:
    The effects of each gene are divided by a scale (see ``standardize``), and the set
    statistic is the mean b over the set's genes of these (d + 1)-vectors. Under the
    null hypothesis the d + 1 effects of a gene are independent with equal variances,
    so one rotation applied to the effects of every gene leaves the joint distribution
    of the data unchanged (Wu et al. 2010, section 3.4). Under a uniformly random
    rotation b turns uniformly over its sphere, so T = b_0 / (sum_k b_k^2 / d)^(1/2) has
    the t distribution on d degrees of freedom: the p-value the rotation test of the
    mean statistic would give with infinitely many rotations, exactly so when every
    gene has the same scale. The mixed (non-directional) statistic is the sum over
    genes of the squared standardized contrast effects. Under rotation it is r'(E'E)r
    for the set's m x (d + 1) matrix E of standardized effects and r uniform on the unit
    sphere, so it equals sum_k lambda_k v_k with lambda_k the d + 1 eigenvalues of E'E
    and (v_k) Dirichlet(1/2, ..., 1/2); scaled to [0, 1] by the smallest and largest
    eigenvalue, it is approximated by the beta distribution with its first two moments
    (found by probing to be limma's approximation, for sets of more than d genes). With
    one gene the approximation is exact: the mixed p-value is the two-sided p-value of
    the gene's t-statistic. For more genes it can be much too small in the tail, as
    limma's is: 6.3e-4 where a simulation of the rotation test gives 3.2e-3.

    The scales of ``standardize="residual_sd"`` and ``"posterior_sd"`` start from the
    mean square of a gene's d + 1 effects without the largest, which leaves out the
    contrast effect of a strongly differentially expressed gene (found by probing).
    ``"posterior_sd"`` divides it by its expected value under the null hypothesis,
    which makes it unbiased for sigma^2, and moderates it with the prior of the
    residual variances as a variance on 0.92 d degrees of freedom, fewer than d since
    the largest effect is left out (limma's rule, found by probing).

Deliberate differences from limma:
    - fry's mixed p-value of a set of 2 to d genes. limma uses only the set's m nonzero
      eigenvalues and treats them as an m-dimensional problem (found by probing), which
      can be far from the rotation p-value; here all d + 1 dimensions are used, as for
      larger sets. ``validation/gene_set_tests.py`` compares both with the rotation
      p-value by simulation.
    - fry's ``standardize="posterior_sd"``. limma's expected value of the trimmed mean
      square is about 7.3e-5 (d + 1) / d above the exact one used here (found by
      probing), which moves p-values by about 1e-4 of their value.
    - camera's rank test with continuity correction can give both tails above 1/2;
      limma then reports twice the smaller, a p-value above 1, and here it is 1.
    - A gene listed twice in a set counts once; limma counts it twice.
    - A set without genes, or containing every gene in camera, has NaN p-values and no
      direction, and is left out of the adjusted p-values of the other sets. limma
      reports the direction "Up", in camera's rank test the p-value 2, which it adjusts
      with the others, and in fry a mixed p-value of 0 for an empty set.
    - camera takes ``robust``: the moderated t-statistics of the robust empirical Bayes
      estimator (Phipson et al. 2016), each turned into a z-score on its own degrees of
      freedom. limma's camera has no such option and ignores it.

References:
    Wu D, Smyth GK (2012). Camera: a competitive gene set test accounting for inter-gene
    correlation. Nucleic Acids Research 40, e133. doi:10.1093/nar/gks461

    Wu D, Lim E, Vaillant F, Asselin-Labat ML, Visvader JE, Smyth GK (2010). ROAST:
    rotation gene set tests for complex microarray experiments. Bioinformatics 26,
    2176-2182. doi:10.1093/bioinformatics/btq401

    Giner G, Smyth GK (2016). FRY: a fast approximation to ROAST gene set test with mean
    aggregated set statistics [version 1; not peer reviewed]. F1000Research 5, 2605
    (slides). doi:10.7490/f1000research.1113351.1

    Goeman JJ, Buehlmann P (2007). Analyzing gene expression data in terms of gene sets:
    methodological issues. Bioinformatics 23, 980-987. doi:10.1093/bioinformatics/btm051

    Barry WT, Nobel AB, Wright FA (2008). A statistical framework for testing functional
    categories in microarray data. Annals of Applied Statistics 2, 286-315.
    doi:10.1214/07-AOAS146

    Hill GW (1970). ACM Algorithm 395: Student's t-distribution. Communications of the
    ACM 13, 617-619. doi:10.1145/355598.355599

    Brophy AL (1987). Efficient estimation of probabilities in the t distribution.
    Behavior Research Methods, Instruments, & Computers 19, 462-466.
    doi:10.3758/BF03205616

    Zar JH (1999). Biostatistical Analysis, 4th edition. Prentice Hall, Upper Saddle
    River.

    Phipson B, Lee S, Majewski IJ, Alexander WS, Smyth GK (2016). Robust hyperparameter
    estimation protects against hypervariable genes and improves power to detect
    differential expression. Annals of Applied Statistics 10, 946-963.
    doi:10.1214/16-AOAS920

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Golub GH, Van Loan CF (2013). Matrix Computations, 4th edition. Johns Hopkins
    University Press, Baltimore.
"""

import math
import numbers
from collections.abc import Mapping, Set
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._linear_algebra import RANK_TOLERANCE, householder_qr
from cytepool._masked_arrays import real_array, reject_masked, unmasked_array
from cytepool._options import checked_flag
from cytepool._ranks import average_ranks
from cytepool.stats._moderation import posterior_variances
from cytepool._summation import observed_means
from cytepool.design.contrasts import _is_named_contrasts, _weights_on_columns
from cytepool.design.design import Design, _refuse_random_effects
from cytepool.stats.distributions import beta_sf, norm_sf, t_sf
from cytepool.stats.empirical_bayes import squeeze_variances
from cytepool.stats.linear_models import (
    _aliased_columns,
    _design_matrix,
)
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.voom import VoomResult
from cytepool.table import Table

__all__ = ["camera", "fry"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

_STANDARDIZATIONS = ("posterior_sd", "residual_sd", "none")
# camera bounds each gene's residual variance below by this before dividing its
# residual effects by the standard deviation (found by probing).
_CORRELATION_VARIANCE_FLOOR = 1e-8
# A pivot of the QR decomposition this small relative to the norm of its column is
# zero up to rounding; see _effects_of_last_coefficient.
_ZERO_PIVOT = 1e-12
# fry's posterior standardization moderates the trimmed mean square as a variance on
# this fraction of the residual degrees of freedom (limma's rule, found by probing).
_TRIMMED_DF_FRACTION = 0.92
# Composite Gauss-Legendre rule over [0, 12] for E[max of k chi-squared(1)]: beyond 12
# the integrand, 2 z (1 - (1 - 2 Q(z))^k), is below 1e-21 for any k up to 1e9.
_PANELS = 48
_PANEL_WIDTH = 0.25
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(16)


def camera(
    y: Any,
    design: Any,
    contrast: Any,
    gene_sets: Mapping[str, Any],
    *,
    gene_ids: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    inter_gene_correlation: float | None = 0.01,
    allow_negative_correlation: bool = False,
    use_ranks: bool = False,
    trend: bool = False,
    robust: bool = False,
) -> Table:
    """Competitive test of gene sets for one contrast, allowing for their correlation.

    limma's ``camera`` (Wu and Smyth 2012): tests whether the genes of each set are more
    differentially expressed for ``contrast`` than the other genes, by a two-sample test
    of their moderated z-scores against those of the other genes, with the variance of
    the set's mean inflated by the correlation between its genes. The module docstring
    gives the method. The default correlation of 0.01 is limma's (since limma 3.29.6):
    it ranks co-regulated, interpretable sets higher than a correlation estimated for
    each set, whose error rate control is strict.

    Args:
        y: ``(n_samples, n_genes)`` log-expression values, finite, or a
            :class:`~cytepool.stats.voom.VoomResult`, whose log-CPM values are tested
            with its weights.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank, or
            anything else that ``fit_linear_model`` accepts, such as a
            :class:`~cytepool.design.Design`. It needs residual degrees of freedom.
        contrast: The contrast to test: the name of a named contrast of a ``Design``,
            the name or 0-based position of a design column (a coefficient; negative
            positions count from the last column), or weights on the design columns: a
            vector in their order, a mapping of column name to weight (columns left out
            weigh 0), a pandas Series indexed by them, or a ``Contrasts`` holding one
            contrast.
        gene_sets: Mapping of set name to its genes: gene ids (matched to
            ``gene_ids``; ids not among them are left out, as limma's ``ids2indices``
            does), 0-based positions of columns of ``y`` when ``gene_ids`` is not
            given, or a boolean mask over the genes. A gene listed twice counts once.
        gene_ids: One id per gene (column of ``y``), unique, to name genes in
            ``gene_sets`` by id.
        weights: Precision weights, positive and finite: ``(n_samples,)`` for sample
            weights or ``(n_samples, n_genes)`` for observation weights. Not with a
            ``VoomResult``, which carries its own. limma's per-gene weights (a vector
            of one weight per gene) have no counterpart.
        inter_gene_correlation: The correlation between the genes of every set, in
            (-1, 1), or None to estimate it for each set from its residuals.
        allow_negative_correlation: Whether a negative correlation, given or
            estimated, may shrink the variance of a set's mean; otherwise it counts as
            zero.
        use_ranks: Whether to test the ranks of the z-scores (a correlation-adjusted
            Wilcoxon-Mann-Whitney test) rather than their values.
        trend: Whether the prior variance of the moderation follows the mean
            log-expression of the genes (limma's ``trend.var``).
        robust: Whether to moderate the variances with the robust estimator, which
            limma's camera does not offer.

    Returns:
        A table with one row per gene set, in the order of ``gene_sets``: ``gene_set``;
        ``n_genes``, the number of the set's genes among those tested; ``correlation``,
        the estimated inter-gene correlation (NaN for a set of one gene) or the given
        one; ``direction``, ``"up"`` or ``"down"`` (None where the p-value is NaN);
        ``p_value``, two-sided; and ``adjusted_p_value``, Benjamini-Hochberg over the
        sets (Benjamini and Hochberg 1995). A set with no genes, or with every gene,
        has NaN p-values.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If ``y`` is not finite or has fewer than three genes, the design
            does not have one row per sample, is not of full column rank or leaves no
            residual degrees of freedom, the contrast or a gene set is invalid, a weight
            is not positive and finite, or no gene id of any set is among ``gene_ids``.
    """
    rho = _correlation_option(inter_gene_correlation)
    allow_negative = checked_flag(
        allow_negative_correlation, "allow_negative_correlation", expected="a bool"
    )
    ranks = checked_flag(use_ranks, "use_ranks", expected="a bool")
    with_trend = checked_flag(trend, "trend", expected="a bool")
    with_robust = checked_flag(robust, "robust", expected="a bool")
    values, precision = _expression(y, weights)
    n_genes = values.shape[1]
    if n_genes < 3:
        raise ValueError(
            "camera compares the genes of each set with the other genes and needs at "
            f"least 3 genes, got {n_genes}"
        )
    members = _gene_set_members(gene_sets, gene_ids, n_genes)
    effects, mean_expression, d = _contrast_effects(
        values, design, contrast, precision, "camera"
    )
    residual = effects[:, 1:]
    variances = np.sum(residual**2, axis=1) / d
    squeezed = squeeze_variances(
        variances,
        d,
        covariate=mean_expression if with_trend else None,
        robust=with_robust,
    )
    prior_df = np.broadcast_to(squeezed.prior_df, variances.shape)
    df_total = np.minimum(d + prior_df, float(d * n_genes))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = effects[:, 0] / np.sqrt(squeezed.posterior_variance)
    z = _hill_z_scores(t, df_total)

    if rho is None:
        scale = np.sqrt(np.maximum(variances, _CORRELATION_VARIANCE_FLOOR))
        standardized = residual / scale[:, np.newaxis]
        correlations = np.array(
            [_estimated_correlation(standardized[rows], d) for _, rows in members]
        )
    else:
        correlations = np.full(len(members), rho)
    used = np.where(np.isnan(correlations), 0.0, correlations)
    if not allow_negative:
        used = np.maximum(used, 0.0)
    sizes = np.array([rows.size for _, rows in members], dtype=np.int64)
    # A given correlation is exact, so the rank sum is then normal and the t-statistic
    # has the G - 2 degrees of freedom of a two-sample test. An estimated one carries d
    # degrees of freedom, and both tests use min(d, G - 2) (found by probing).
    estimated_df = float(min(d, n_genes - 2))
    if ranks:
        df = estimated_df if rho is None else math.inf
        lower, upper = _rank_sum_tails(z, members, used, df)
    else:
        df = estimated_df if rho is None else float(n_genes - 2)
        lower, upper = _two_sample_tails(z, members, used, df)
    p_value, direction = _two_sided(lower, upper)
    return Table(
        {
            "gene_set": [name for name, _ in members],
            "n_genes": sizes,
            "correlation": correlations,
            "direction": direction,
            "p_value": p_value,
            "adjusted_p_value": adjust_p_values(p_value, "bh"),
        }
    )


def fry(
    y: Any,
    design: Any,
    contrast: Any,
    gene_sets: Mapping[str, Any],
    *,
    gene_ids: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    standardize: str = "posterior_sd",
    trend: bool = False,
    robust: bool = False,
) -> Table:
    """Self-contained test of gene sets for one contrast, without rotations.

    limma's ``fry`` (Giner and Smyth 2016): tests whether the genes of each set are
    differentially expressed for ``contrast``, in the same direction (the directional
    p-value) or in either (the mixed p-value), by the limit of the rotation test of Wu
    et al. (2010) with the mean statistic over infinitely many rotations. The module
    docstring gives the method and the deliberate differences from limma. limma's
    gene contribution weights (``gene.weights``, or weights inside a set), which give
    genes signs and sizes in the set statistic, are not implemented.

    Args:
        y: ``(n_samples, n_genes)`` log-expression values, finite, or a
            :class:`~cytepool.stats.voom.VoomResult`, whose log-CPM values are tested
            with its weights.
        design: The design, as for :func:`camera`.
        contrast: The contrast to test, as for :func:`camera`.
        gene_sets: Mapping of set name to its genes, as for :func:`camera`.
        gene_ids: One id per gene, unique, to name genes in ``gene_sets`` by id.
        weights: Precision weights, as for :func:`camera`.
        standardize: How each gene's effects are scaled before they are averaged:
            ``"posterior_sd"``, by the square root of a posterior variance that combines
            the prior of the empirical Bayes moderation with the mean square of the
            gene's d + 1 effects without the largest (limma's ``"posterior.sd"``);
            ``"residual_sd"``, by the square root of that mean square alone (limma's
            ``"residual.sd"``); or ``"none"``, unscaled, which suits genes of equal
            variance.
        trend: Whether the prior variance follows the mean log-expression of the
            genes. Only with ``standardize="posterior_sd"``.
        robust: Whether to estimate the prior robustly. Only with
            ``standardize="posterior_sd"``.

    Returns:
        A table with one row per gene set, in the order of ``gene_sets``: ``gene_set``;
        ``n_genes``; ``direction``, ``"up"`` or ``"down"`` (None where the p-value is
        NaN); ``p_value``, the two-sided directional p-value; ``adjusted_p_value``,
        Benjamini-Hochberg over the sets; ``mixed_p_value``, the non-directional
        p-value; and ``adjusted_mixed_p_value``. A set with no genes has NaN p-values,
        and so, with ``standardize="residual_sd"``, has a set containing a gene whose
        d + 1 effects are zero but for at most one, which cannot be scaled.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: As :func:`camera` (one gene suffices), if ``standardize`` is
            unknown, or if ``trend`` or ``robust`` is given with another
            standardization, which has no prior.
    """
    if not isinstance(standardize, str) or standardize not in _STANDARDIZATIONS:
        raise ValueError(
            "standardize must be one of "
            f"{', '.join(map(repr, _STANDARDIZATIONS))}, got {standardize!r}"
        )
    with_trend = checked_flag(trend, "trend", expected="a bool")
    with_robust = checked_flag(robust, "robust", expected="a bool")
    if (with_trend or with_robust) and standardize != "posterior_sd":
        raise ValueError(
            "trend and robust shape the prior of the posterior standard deviations, so "
            "they apply only with standardize='posterior_sd'"
        )
    values, precision = _expression(y, weights)
    members = _gene_set_members(gene_sets, gene_ids, values.shape[1])
    effects, mean_expression, d = _contrast_effects(
        values, design, contrast, precision, "fry"
    )
    squares = effects**2
    # The mean square of the d + 1 effects without the largest (found by probing).
    trimmed = (np.sum(squares, axis=1) - np.max(squares, axis=1)) / d
    if standardize == "none":
        scale = np.ones(effects.shape[0])
    elif standardize == "residual_sd":
        scale = np.sqrt(trimmed)
    else:
        variances = np.sum(squares[:, 1:], axis=1) / d
        squeezed = squeeze_variances(
            variances,
            d,
            covariate=mean_expression if with_trend else None,
            robust=with_robust,
        )
        scale = np.sqrt(
            posterior_variances(
                trimmed / _trimmed_mean_square_expectation(d),
                _TRIMMED_DF_FRACTION * d,
                np.broadcast_to(squeezed.prior_variance, trimmed.shape),
                np.broadcast_to(squeezed.prior_df, trimmed.shape),
            )
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = effects / scale[:, np.newaxis]
    standardized[scale == 0.0] = np.nan

    n_sets = len(members)
    statistic = np.full(n_sets, np.nan)
    mixed = np.full(n_sets, np.nan)
    for i, (_, rows) in enumerate(members):
        if rows.size == 0:
            continue
        block = standardized[rows]
        if not np.all(np.isfinite(block)):
            continue
        mean = np.mean(block, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic[i] = mean[0] / math.sqrt(float(np.sum(mean[1:] ** 2)) / d)
        mixed[i] = _mixed_p_value(block, d)
    lower, upper = _t_tails(statistic, float(d))
    p_value, direction = _two_sided(lower, upper)
    return Table(
        {
            "gene_set": [name for name, _ in members],
            "n_genes": np.array([rows.size for _, rows in members], dtype=np.int64),
            "direction": direction,
            "p_value": p_value,
            "adjusted_p_value": adjust_p_values(p_value, "bh"),
            "mixed_p_value": mixed,
            "adjusted_mixed_p_value": adjust_p_values(mixed, "bh"),
        }
    )


# ---------------------------------------------------------------------------------------
# Effects of the reparametrised fits
# ---------------------------------------------------------------------------------------


def _contrast_effects(
    values: FloatArray,
    design: Any,
    contrast: Any,
    weights: FloatArray | None,
    function: str,
) -> tuple[FloatArray, FloatArray, int]:
    """Each gene's contrast effect and residual effects, its mean, and d.

    Returns:
        ``(n_genes, d + 1)`` effects, the contrast effect first, signed as the
        estimated contrast; the mean of each gene's values; and the residual degrees
        of freedom d.
    """
    reparametrised, d = _contrast_design(values, design, contrast, function)
    return _effects(values, reparametrised, weights), observed_means(values), d


def _contrast_design(
    values: FloatArray, design: Any, contrast: Any, function: str
) -> tuple[FloatArray, int]:
    """The design, checked, with the contrast as its last coefficient, and d."""
    _refuse_random_effects(design, function)
    x, names = _design_matrix(design, None)
    n_samples, n_coefficients = x.shape
    if n_samples != values.shape[0]:
        raise ValueError(
            f"y has {values.shape[0]} rows but the design has {n_samples}; rows "
            "of both are samples"
        )
    aliased = _aliased_columns(x)
    if aliased:
        listed = ", ".join(repr(names[j]) for j in aliased)
        raise ValueError(
            f"design is not of full column rank: column(s) {listed} are linear "
            "combinations of the preceding columns; remove them or reparametrise the "
            "design"
        )
    d = n_samples - n_coefficients
    if d < 1:
        raise ValueError(
            f"the design has {n_coefficients} columns for {n_samples} samples and "
            f"leaves no residual degrees of freedom, which {function} needs to "
            "estimate the variances"
        )
    return _reparametrised_design(x, _contrast(contrast, design, names)), d


def _effects(
    values: FloatArray,
    reparametrised: FloatArray,
    weights: FloatArray | None,
    *,
    signed_residuals: bool = False,
) -> FloatArray:
    """The ``(n_genes, d + 1)`` effects of each gene's fit of the design.

    ``signed_residuals`` multiplies the residual effects, like the contrast effect, by
    the sign of the last diagonal element of R: the residual basis of limma's romer
    (found by probing), which camera, fry and roast do not use.
    """
    n_samples = reparametrised.shape[0]
    if weights is None:
        weight_rows = np.ones((1, n_samples))
    elif weights.ndim == 1:
        weight_rows = weights[np.newaxis, :]
    else:
        weight_rows = weights.T
    effects: FloatArray = _effects_of_last_coefficient(
        np.ascontiguousarray(values.T),
        np.ascontiguousarray(weight_rows),
        np.ascontiguousarray(reparametrised),
        RANK_TOLERANCE,
        signed_residuals,
    )
    deficient = np.flatnonzero(np.isnan(effects[:, 0]))
    if deficient.size:
        raise ValueError(
            f"the weights make the design rank deficient for {deficient.size} gene(s), "
            f"such as gene {deficient[0]}; weights that differ by many orders of "
            "magnitude between samples cannot be used"
        )
    return effects


@jit
def _effects_of_last_coefficient(
    responses: FloatArray,
    weight_rows: FloatArray,
    design: FloatArray,
    tolerance: float,
    signed_residuals: bool,
) -> FloatArray:
    """Contrast and residual effects of each gene's weighted least squares fit.

    Householder QR decomposition of each gene's whitened design, in the sign convention
    of R's ``qr()``: column j is reflected onto -s r e_j, with r the norm of its part
    from row j on and s the sign of its element in row j, or s = 1 when that element is
    zero. An element below ``_ZERO_PIVOT`` times r counts as zero: it is zero in exact
    arithmetic, as in an indicator design with samples sorted by group, and its sign
    is rounding noise, which would otherwise decide the basis of the residual effects.

    Args:
        responses: ``(n_genes, n_samples)`` values.
        weight_rows: ``(n_genes, n_samples)`` precision weights, or one row shared by
            all genes; positive.
        design: ``(n_samples, p)`` design whose last coefficient is the contrast.
        tolerance: Relative norm at or below which a column is aliased.
        signed_residuals: Whether the residual effects are multiplied by the same
            sign as the contrast effect.

    Returns:
        ``(n_genes, n_samples - p + 1)`` effects: the p-th element of Q'y for the
        whitened values y, times the sign of the p-th diagonal element of R, so that it
        has the sign of the estimated contrast, then the n - p residual effects. A row
        is NaN when the whitened design of that gene is rank deficient.
    """
    n_genes, n_samples = responses.shape
    p = design.shape[1]
    effects = np.empty((n_genes, n_samples - p + 1))
    work = np.empty((n_samples, p + 1))
    norms = np.empty(p)
    aliased = np.empty(p, dtype=np.bool_)
    kept = np.empty(p, dtype=np.int64)
    products = np.empty(p + 1)
    shared = weight_rows.shape[0] == 1
    for g in range(n_genes):
        w = 0 if shared else g
        for i in range(n_samples):
            scale = math.sqrt(weight_rows[w, i])
            for j in range(p):
                work[i, j] = scale * design[i, j]
            work[i, p] = scale * responses[g, i]
        # The norm after the earlier reflections is the reference of the rank rule, a
        # head within _ZERO_PIVOT of zero counts as zero, and v'v is summed.
        rank = householder_qr(
            work, n_samples, p, tolerance, True, _ZERO_PIVOT, True, norms, aliased,
            kept, products,
        )  # fmt: skip
        if rank < p:
            effects[g, :] = np.nan  # aliased: the design is rank deficient
            continue
        # The diagonal element of R of the contrast, by R's sign convention.
        sign = -1.0 if work[p - 1, p - 1] < 0.0 else 1.0
        effects[g, 0] = sign * work[p - 1, p]
        residual_sign = sign if signed_residuals else 1.0
        for i in range(p, n_samples):
            effects[g, i - p + 1] = residual_sign * work[i, p]
    return effects


def _contrast(contrast: Any, design: Any, names: tuple[str, ...]) -> int | FloatArray:
    """The contrast: the position of a design column, or weights on the columns."""
    reject_masked(contrast, "contrast")
    n_columns = len(names)
    if isinstance(contrast, str):
        if isinstance(design, Design) and contrast in design.contrasts:
            weights = np.array(design.contrasts[contrast], dtype=np.float64)
            return _checked_weights(weights, contrast)
        if contrast in names:
            return names.index(contrast)
        named = list(design.contrasts.names) if isinstance(design, Design) else []
        raise ValueError(
            f"contrast {contrast!r} is neither a named contrast of the design nor "
            "one of its columns; contrasts: "
            f"{', '.join(named) if named else '(none)'}; columns: {', '.join(names)}"
        )
    if isinstance(contrast, np.ndarray) and contrast.ndim == 0:
        contrast = contrast.item()  # a 0-d array is its value
    if isinstance(contrast, (bool, np.bool_)) or (
        isinstance(contrast, numbers.Real)
        and not isinstance(contrast, numbers.Integral)
    ):
        raise TypeError(
            "contrast must be a contrast name, a column name or position (an integer), "
            f"or weights on the design columns, got {type(contrast).__name__}"
        )
    if isinstance(contrast, numbers.Integral):
        position = int(contrast)
        if not -n_columns <= position < n_columns:
            raise ValueError(
                f"contrast {position} is not the position of a design column; the "
                f"design has {n_columns} columns, positions 0 to {n_columns - 1}"
            )
        return position % n_columns
    if isinstance(contrast, Mapping) and not _is_named_contrasts(contrast):
        return _weights_by_column(contrast, names)
    remedy = "give one weight per design column, in their order or named by them"
    weights = _weights_on_columns(
        contrast,
        names,
        label="the contrast",
        target="the design's columns",
        remedy=remedy,
    )
    try:
        vector = np.array(weights, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(
            "contrast must be a contrast name, a column name or position, or weights "
            f"on the design columns, got {type(contrast).__name__}"
        ) from None
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    elif vector.ndim == 2:
        raise ValueError(
            f"{vector.shape[1]} contrasts were given; one contrast is tested at a "
            "time, so select one, for example with contrasts.select(name)"
        )
    if vector.shape != (n_columns,):
        raise ValueError(
            f"the contrast must have one weight per design column ({n_columns}: "
            f"{', '.join(names)}), got shape {vector.shape}"
        )
    return _checked_weights(vector, "the contrast")


def _weights_by_column(
    weights: Mapping[Any, Any], names: tuple[str, ...]
) -> FloatArray:
    """Weights given as a mapping of column name to weight; columns left out weigh 0."""
    vector = np.zeros(len(names))
    for column, weight in weights.items():
        if column not in names:
            raise ValueError(
                f"the contrast weighs {column!r}, which is not a design column; "
                f"columns are: {', '.join(names)}"
            )
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, numbers.Real):
            raise TypeError(
                "a contrast given as a mapping takes one number per design column, got "
                f"{type(weight).__name__} for {column!r}"
            )
        vector[names.index(column)] = float(weight)
    return _checked_weights(vector, "the contrast")


def _checked_weights(vector: FloatArray, label: str) -> FloatArray:
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} has weights that are not finite")
    if not np.any(vector != 0.0):
        raise ValueError(f"{label} is zero on every column, so it compares nothing")
    return vector


def _reparametrised_design(x: FloatArray, contrast: int | FloatArray) -> FloatArray:
    """The design with the contrast as its last coefficient, as limma forms it.

    A column moves to the end. Weights c give the columns X h_2, ..., X h_p and then
    X c / (c'c), where h_j are the columns of the Householder reflection H of c in R's
    sign convention, so that the last coefficient is c'alpha (limma's
    ``contrastAsCoef``, found by probing).
    """
    if isinstance(contrast, int):
        order = [j for j in range(x.shape[1]) if j != contrast] + [contrast]
        return x[:, order]
    reflection = _householder_reflection(contrast)
    last = x @ contrast / float(contrast @ contrast)
    return np.column_stack([x @ reflection[:, 1:], last])


def _householder_reflection(c: FloatArray) -> FloatArray:
    """H = I - u u' / u_0 with u = c / norm + e_0, norm = |c| with the sign of c_0.

    H c = -norm e_0, so the reflection maps c to the first axis as R's ``qr()``
    (LINPACK) does, with a positive norm when c_0 is zero.
    """
    norm = float(np.linalg.norm(c))
    if c[0] != 0.0:
        norm = math.copysign(norm, c[0])
    u = c / norm
    u[0] += 1.0
    reflection: FloatArray = np.eye(c.size) - np.outer(u, u) / u[0]
    return reflection


# ---------------------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------------------


def _hill_z_scores(t: FloatArray, df: FloatArray) -> FloatArray:
    """Hill's (1970) normal approximation to t-statistics on df degrees of freedom.

    z = w (1 + (w^2 + 3) / b - (4 w^6 + 33 w^4 + 240 w^2 + 855) / (10 b (b + 0.8 w^4
    + 100))) with w^2 = (df - 1/2) log(1 + t^2 / df) and b = 48 (df - 1/2)^2 (Brophy
    1987, equation 5). Infinite t stay infinite.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a = df - 0.5
        b = 48.0 * a * a
        w2 = a * np.log1p(t * t / df)
        correction = (
            (((-0.4 * w2 - 3.3) * w2 - 24.0) * w2 - 85.5) / (0.8 * w2 * w2 + 100.0 + b)
            + w2
            + 3.0
        ) / b
        z: FloatArray = np.copysign((1.0 + correction) * np.sqrt(w2), t)
    return np.where(np.isinf(t), t, z)


def _estimated_correlation(standardized: FloatArray, d: int) -> float:
    """The mean correlation of a set's genes from its standardized residual effects.

    The VIF is m/d times the sum of squares of the mean over the m genes, and
    VIF = 1 + (m - 1) rho (Wu and Smyth 2012); NaN for fewer than two genes. The VIF
    lies in [0, m], so rho at most 1; genes whose residuals are equal reach 1, which
    rounding can exceed, so rho is bounded by 1.
    """
    m = int(standardized.shape[0])
    if m < 2:
        return math.nan
    mean = np.mean(standardized, axis=0)
    vif = m * float(np.sum(mean * mean)) / d
    return min((vif - 1.0) / (m - 1), 1.0)


def _two_sample_tails(
    z: FloatArray,
    members: list[tuple[str, IntArray]],
    correlations: FloatArray,
    df: float,
) -> tuple[FloatArray, FloatArray]:
    """P(T <= t) and P(T > t) of the correlation-adjusted two-sample t-statistics.

    With delta = (mean of the set - mean of all) G / m2 and s_p^2 = ((G - 1) s^2 -
    delta^2 m m2 / G) / (G - 2), the pooled variance of the two groups written with the
    mean and variance s^2 of all G z-scores, t = delta / (s_p (VIF / m + 1 / m2)^(1/2))
    for a set of m genes and the m2 = G - m others (Wu and Smyth 2012).
    """
    n_genes = z.size
    overall = float(np.mean(z))
    variance = float(np.var(z, ddof=1))
    statistic = np.full(len(members), np.nan)
    for i, (_, rows) in enumerate(members):
        m = rows.size
        others = n_genes - m
        if m == 0 or others == 0:
            continue
        vif = 1.0 + (m - 1) * correlations[i]
        delta = (float(np.mean(z[rows])) - overall) * n_genes / others
        pooled = ((n_genes - 1) * variance - delta * delta * m * others / n_genes) / (
            n_genes - 2
        )
        # A negative correlation can make the VIF, and with it the variance, negative:
        # no such correlation exists for a set this large.
        squared_error = pooled * (vif / m + 1.0 / others)
        if squared_error > 0.0:
            statistic[i] = delta / math.sqrt(squared_error)
    return _t_tails(statistic, df)


def _rank_sum_tails(
    z: FloatArray,
    members: list[tuple[str, IntArray]],
    correlations: FloatArray,
    df: float,
) -> tuple[FloatArray, FloatArray]:
    """The left and right tails of the correlation-adjusted rank sum test of each set.

    The rank sum of a set of m1 genes among all G, with m2 = G - m1 others, has
    variance m1 m2 / (2 pi) (asin(1) + (m2 - 1) asin(1/2) + (m1 - 1)(m2 - 1)
    asin(rho/2) + (m1 - 1) asin((rho + 1)/2)) when the set's values share correlation
    rho (Wu and Smyth 2012), times the tie correction of Zar (1999, 8.10). With
    U = m1 m2 + m1 (m1 + 1)/2 - R1 and a continuity correction of 1/2, the left tail
    (the set ranked low) is P(T > (U - m1 m2/2 - 1/2) / sd) and the right tail
    P(T < (U - m1 m2/2 + 1/2) / sd), for T on ``df`` degrees of freedom.
    """
    n_genes = z.size
    ranks = average_ranks(z)
    _, counts = np.unique(z, return_counts=True)
    ties = counts.astype(np.float64)
    tie_factor = 1.0 - float(np.sum(ties**3 - ties)) / (float(n_genes) ** 3 - n_genes)
    low = np.full(len(members), np.nan)
    high = np.full(len(members), np.nan)
    for i, (_, rows) in enumerate(members):
        m1 = rows.size
        m2 = n_genes - m1
        if m1 == 0 or m2 == 0:
            continue
        rho = correlations[i]
        variance = (
            m1
            * m2
            / (2.0 * math.pi)
            * (
                math.asin(1.0)
                + (m2 - 1) * math.asin(0.5)
                + (m1 - 1) * (m2 - 1) * math.asin(rho / 2.0)
                + (m1 - 1) * math.asin((rho + 1.0) / 2.0)
            )
            * tie_factor
        )
        if not variance > 0.0:
            continue
        sd = math.sqrt(variance)
        u = m1 * m2 + m1 * (m1 + 1) / 2.0 - float(np.sum(ranks[rows]))
        centre = m1 * m2 / 2.0
        low[i] = (u - centre - 0.5) / sd
        high[i] = (u - centre + 0.5) / sd
    # P(T < high) as P(T > -high), so that a centred rank sum gives equal tails.
    left = np.asarray(t_sf(low, df), dtype=np.float64)
    right = np.asarray(t_sf(-high, df), dtype=np.float64)
    return left, right


# ---------------------------------------------------------------------------------------
# fry
# ---------------------------------------------------------------------------------------


@cache
def _trimmed_mean_square_expectation(d: int) -> float:
    """E[(sum - max) / d] of d + 1 independent chi-squared(1) values.

    That is (d + 1 - E[max]) / d, with E[max of k chi-squared(1)] = integral over
    z > 0 of 2 z (1 - (1 - 2 Q(z))^k), Q the normal upper tail, by composite
    Gauss-Legendre quadrature; 1 - 2/pi for d = 1.
    """
    k = float(d + 1)
    total = 0.0
    for panel in range(_PANELS):
        z = (panel + (_NODES + 1.0) / 2.0) * _PANEL_WIDTH
        tail = 2.0 * np.asarray(norm_sf(z), dtype=np.float64)
        integrand = -2.0 * z * np.expm1(k * np.log1p(-tail))
        total += _PANEL_WIDTH / 2.0 * float(np.dot(_WEIGHTS, integrand))
    return (k - total) / d


def _mixed_p_value(standardized: FloatArray, d: int) -> float:
    """The beta approximation to the rotation p-value of the sum of squared statistics.

    The statistic is sum_g E_g0^2 = e_0'(E'E)e_0; under rotation e_0 becomes r, uniform
    on the unit sphere of dimension d + 1, so the statistic is sum_k lambda_k v_k with
    (v_k) Dirichlet(1/2, ..., 1/2): each v_k has mean 1/(d + 1) and variance
    d / ((d + 1)^2 ((d + 1)/2 + 1)), and two of them covariance -1/d times that. Scaled
    to [0, 1] by the smallest and largest of the d + 1 eigenvalues lambda_k of E'E, it
    is given the beta distribution with the same mean and variance.
    """
    n_effects = d + 1
    singular = np.linalg.svd(standardized, compute_uv=False)
    eigenvalues = np.zeros(n_effects)
    eigenvalues[: singular.size] = singular * singular
    smallest = float(eigenvalues.min())
    largest = float(eigenvalues.max())
    if not largest > smallest:
        return math.nan
    spread = largest - smallest
    observed = float(np.sum(standardized[:, 0] ** 2))
    variance_v = d / (n_effects * n_effects * (n_effects / 2.0 + 1.0))
    total = float(np.sum(eigenvalues))
    squares = float(np.sum(eigenvalues * eigenvalues))
    mean = (total / n_effects - smallest) / spread
    variance = variance_v * ((1.0 + 1.0 / d) * squares - total * total / d) / spread**2
    shape_sum = mean * (1.0 - mean) / variance - 1.0
    alpha = shape_sum * mean
    beta = shape_sum - alpha
    return float(beta_sf((observed - smallest) / spread, alpha, beta))


# ---------------------------------------------------------------------------------------
# Results and arguments
# ---------------------------------------------------------------------------------------


def _t_tails(statistic: FloatArray, df: float) -> tuple[FloatArray, FloatArray]:
    """P(T <= t) and P(T > t) for T on df degrees of freedom, as P(T > -t) and
    P(T > t), so that t = 0 gives equal tails."""
    lower = np.asarray(t_sf(-statistic, df), dtype=np.float64)
    upper = np.asarray(t_sf(statistic, df), dtype=np.float64)
    return lower, upper


def _two_sided(
    lower: FloatArray, upper: FloatArray
) -> tuple[FloatArray, list[str | None]]:
    """Twice the smaller tail, at most 1, and the direction of the smaller tail.

    Equal tails count as "up", as in limma (found by probing).
    """
    p_value = np.minimum(1.0, 2.0 * np.minimum(lower, upper))
    direction: list[str | None] = [
        None if math.isnan(p) else ("down" if low < high else "up")
        for p, low, high in zip(
            p_value.tolist(), lower.tolist(), upper.tolist(), strict=True
        )
    ]
    return p_value, direction


def _expression(y: Any, weights: Any) -> tuple[FloatArray, FloatArray | None]:
    """The values (samples x genes) and the precision weights, checked."""
    if isinstance(y, VoomResult):
        if weights is not None:
            raise ValueError(
                "weights cannot be given with a VoomResult, which carries the "
                "precision weights of its log-CPM values"
            )
        return np.asarray(y.log_cpm, dtype=np.float64), np.asarray(
            y.weights, dtype=np.float64
        )
    values = real_array(unmasked_array(y, "y"), "y", boolean=False, objects=False)
    if values.ndim != 2:
        raise ValueError(
            "y must be a 2-D array of shape (n_samples, n_genes), or a VoomResult, got "
            f"shape {values.shape}"
        )
    if values.shape[1] == 0:
        raise ValueError("y must have at least one gene (column)")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "y must be finite: gene-set tests fit every gene to every sample, so "
            "values cannot be missing (NaN) or infinite"
        )
    if weights is None:
        return values, None
    w = real_array(
        unmasked_array(weights, "weights"), "weights", boolean=False, objects=False
    )
    n_samples, n_genes = values.shape
    if w.shape not in ((n_samples,), (n_samples, n_genes)):
        raise ValueError(
            f"weights must have shape ({n_samples},) for sample weights or "
            f"{values.shape} for observation weights, got {w.shape}"
        )
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(w) & (w > 0.0)
    if not np.all(valid):
        raise ValueError(
            "weights must be positive and finite: every sample enters every gene's "
            "fit, so that all genes have the same residual degrees of freedom; "
            f"{int(np.count_nonzero(~valid))} are not"
        )
    return values, np.ascontiguousarray(w)


def _gene_set_members(
    gene_sets: Any, gene_ids: Any, n_genes: int
) -> list[tuple[str, IntArray]]:
    """Each set's name and the sorted positions of its genes among the columns of y."""
    index = _checked_gene_sets(gene_sets, gene_ids, n_genes)
    members: list[tuple[str, IntArray]] = []
    matched = 0
    for name, genes in gene_sets.items():
        positions, found = _set_positions(genes, _set_label(name), index, n_genes)
        matched += found
        members.append((name, positions))
    _check_matched(index, matched)
    return members


def _checked_gene_sets(
    gene_sets: Any, gene_ids: Any, n_genes: int
) -> dict[Any, int] | None:
    """Check that ``gene_sets`` is a non-empty mapping; the index of ``gene_ids``."""
    reject_masked(gene_sets, "gene_sets")
    if not isinstance(gene_sets, Mapping):
        raise TypeError(
            "gene_sets must be a mapping of set name to genes, such as "
            f"{{'interferon': ['ISG15', 'IFIT1']}}; got {type(gene_sets).__name__}"
        )
    if not gene_sets:
        raise ValueError("gene_sets is empty; give at least one gene set")
    return None if gene_ids is None else _gene_index(gene_ids, n_genes)


def _set_label(name: Any) -> str:
    """How messages name a set, after checking its name."""
    if not isinstance(name, str):
        raise TypeError(f"gene set names must be strings, got {name!r}")
    if not name:
        raise ValueError("gene set names must not be empty")
    return f"gene_sets[{name!r}]"


def _set_positions(
    genes: Any, label: str, index: dict[Any, int] | None, n_genes: int
) -> tuple[IntArray, int]:
    """The sorted positions of a set's genes, and how many ids were found."""
    if isinstance(genes, (str, bytes)):
        raise TypeError(
            f"{label} is a single string; give the set's genes as a list, such as "
            f"[{genes!r}]"
        )
    if isinstance(genes, Set):
        genes = list(genes)
    array = unmasked_array(genes, label)
    if array.ndim != 1:
        raise ValueError(
            f"{label} must be a list of genes (a 1-D sequence), got shape {array.shape}"
        )
    found = 0
    if array.dtype == np.bool_:
        if array.size != n_genes:
            raise ValueError(
                f"{label} is a boolean mask of {array.size} values; a mask "
                f"needs one value per gene ({n_genes})"
            )
        positions = np.flatnonzero(array)
    elif index is None:
        positions = _positions(array, label, n_genes)
    else:
        positions = _id_positions(array.tolist(), label, index)
        found = positions.size
    return np.unique(positions).astype(np.intp), found


def _id_positions(ids: list[Any], label: str, index: dict[Any, int]) -> IntArray:
    """The positions of the ids that are among the gene ids; others are left out."""
    try:
        found = [index[gene] for gene in ids if gene in index]
    except TypeError:
        raise TypeError(
            f"{label} holds values that cannot be gene ids, such as {ids[0]!r}"
        ) from None
    return np.array(found, dtype=np.intp)


def _check_matched(index: dict[Any, int] | None, matched: int) -> None:
    if index is not None and matched == 0:
        raise ValueError(
            "no gene of any set is among gene_ids; the sets and gene_ids must name "
            "genes the same way (for example both by symbol, or both by Ensembl id)"
        )


def _gene_index(gene_ids: Any, n_genes: int) -> dict[Any, int]:
    ids = unmasked_array(gene_ids, "gene_ids")
    if ids.shape != (n_genes,):
        raise ValueError(
            f"gene_ids must have one id per gene ({n_genes}), got shape {ids.shape}"
        )
    index: dict[Any, int] = {}
    try:
        for position, gene in enumerate(ids.tolist()):
            if gene in index:
                raise ValueError(
                    f"gene_ids must be unique, but {gene!r} names genes "
                    f"{index[gene]} and {position}"
                )
            index[gene] = position
    except TypeError:
        raise TypeError("gene ids must be hashable values, such as strings") from None
    return index


def _positions(array: npt.NDArray[Any], label: str, n_genes: int) -> IntArray:
    if array.size == 0:
        return np.empty(0, dtype=np.intp)
    if array.dtype.kind not in "iu":
        raise TypeError(
            f"{label} must hold 0-based gene positions (integers) when gene_ids is not "
            f"given; give gene_ids to name genes by id (got dtype {array.dtype})"
        )
    outside = (array < 0) | (array >= n_genes)
    if np.any(outside):
        raise ValueError(
            f"{label} has position {int(array[outside][0])}, but there are {n_genes} "
            f"genes, at positions 0 to {n_genes - 1}"
        )
    return array.astype(np.intp)


def _correlation_option(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(
            "inter_gene_correlation must be a real number or None, got "
            f"{type(value).__name__}"
        )
    rho = float(value)
    if not -1.0 < rho < 1.0:
        raise ValueError(
            "inter_gene_correlation must lie in (-1, 1), or be None to estimate "
            f"it, got {rho}"
        )
    return rho
