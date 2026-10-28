"""Gene-set tests of one contrast of a linear model: limma's camera and fry.

:func:`~cytepool.camera` asks whether the genes of each set are more differentially
expressed for a contrast than the other genes, allowing for the correlation between the
genes of the set (Wu and Smyth 2012), and :func:`~cytepool.fry` whether they are
differentially expressed at all, in one direction or in either, by the limit of the
rotation test of Wu et al. (2010) with the mean statistic over infinitely many rotations
(Giner and Smyth 2016). Each returns a table with one row per set. The rotation tests,
which draw rotations, are :func:`~cytepool.roast`, :func:`~cytepool.mroast` and
:func:`~cytepool.romer`. https://docs.nygen.io/CytePool/statistics/gene-set-tests#camera
explains the tests, and
https://docs.nygen.io/CytePool/statistics/gene-set-tests#gene-set-tests-limits how they
differ from limma's.

Both tests work with the effects of each gene's least squares fit (Wu et al. 2010,
section 4.1): with the design reparametrised so that the contrast is its last
coefficient, the QR decomposition of the design, whitened by the gene's precision
weights, gives the gene's contrast effect u_0 and its d residual effects
(https://docs.nygen.io/CytePool/statistics/gene-set-tests#gene-effects). Every sample
must be observed, with a positive weight, in every gene, so that all genes have the same
residual degrees of freedom d, and the design must be of fixed effects only, of full
column rank, with d of at least 1.

Matching limma:
    The results match limma 3.62.1. limma's documentation gives some of the conventions
    below, and running limma on chosen inputs gave the others.

    - The design is reparametrised as limma's ``contrastAsCoef`` does: a contrast given
      as a column of the design moves that column last; a contrast given as weights c
      becomes the last column, X c / (c'c), after the columns X h_2, ..., X h_p, where
      h_1, ..., h_p are the columns of the Householder reflection H of c, so that the
      last coefficient is c'alpha.
    - The QR decomposition is Householder's (Golub and Van Loan 2013) in R's sign
      convention, so that the residual effects are in limma's basis. With observation
      weights each gene has a basis of its own, and the correlation that camera
      estimates and the statistics of fry depend on it; fry's standardizations other
      than "none" depend on the basis even without weights, through the largest
      effect. The basis depends on how the contrast is given, and a pivot that is zero
      up to rounding (below 1e-12 of its column), as in indicator designs with the
      samples sorted by group, counts as zero, which R's ``qr()`` gives the positive
      sign.
    - camera's moderated t-statistics are the contrast effects over the posterior
      standard deviations of :func:`~cytepool.stats.empirical_bayes.squeeze_variances`,
      on min(d + d0, G d) degrees of freedom (as ``eBayes`` caps them, for G genes),
      turned into z-scores by Hill's (1970) approximation (Brophy 1987, equation 5), as
      limma's ``zscoreT(approx = TRUE, method = "hill")`` does.
    - camera's estimated correlation divides each gene's residual effects by its
      residual standard deviation (its variance bounded below by 1e-8), averages the
      m x d matrix over the set's m genes, and takes m/d times the sum of squares of
      that mean as the variance inflation factor, VIF = 1 + (m - 1) rho (Wu and Smyth
      2012). Unless negative correlations are allowed, rho enters the VIF as
      max(rho, 0); a given negative correlation that makes the VIF of a set negative,
      which no correlation matrix allows, gives the set a NaN p-value.
    - camera's p-value is two-sided, from the t distribution on G - 2 degrees of
      freedom for a given correlation and on min(d, G - 2) for an estimated one, and a
      set whose two tails are equal goes "up". The rank version uses the variance of
      the rank sum when the set's values share a correlation rho (Wu and Smyth 2012,
      from Barry et al. 2008, equation 4.5), times Zar's (1999, section 8.10)
      correction for ties, 1 - sum(t^3 - t) / (G^3 - G), with a continuity correction
      of 1/2, referred to the normal distribution for a given correlation and to the t
      distribution on min(d, G - 2) degrees of freedom for an estimated one.
    - fry's mixed p-value is limma's beta approximation for sets of more than d genes.
      Under a rotation r, the sum of the set's squared standardized contrast effects
      is r'(E'E)r for the set's m x (d + 1) matrix E of standardized effects, so it
      equals sum_k lambda_k v_k, with lambda_k the d + 1 eigenvalues of E'E and (v_k)
      Dirichlet(1/2, ..., 1/2); scaled to [0, 1] by the smallest and largest
      eigenvalue, it is given the beta distribution with its first two moments.
    - fry's scales for ``standardize="residual_sd"`` and ``"posterior_sd"`` start from
      the mean square of a gene's d + 1 effects without the largest, which leaves out
      the contrast effect of a strongly differentially expressed gene.
      ``"posterior_sd"`` divides it by its expected value under the null hypothesis,
      which makes it unbiased for sigma^2, and moderates it with the prior of the
      residual variances as a variance on 0.92 d degrees of freedom, fewer than d since
      the largest effect is left out.

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
from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._options import checked_flag
from cytepool._ranks import average_ranks
from cytepool._summation import observed_means
from cytepool._gene_sets import (
    check_matched,
    checked_gene_sets,
    contrast_design,
    gene_effects,
    gene_set_data,
    set_label,
    set_positions,
)
from cytepool.stats._moderation import posterior_variances
from cytepool.stats.distributions import beta_sf, norm_sf, t_sf
from cytepool.stats.empirical_bayes import squeeze_variances
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.table import Table

__all__ = ["camera", "fry"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

_STANDARDIZATIONS = ("posterior_sd", "residual_sd", "none")
# camera bounds each gene's residual variance below by this before dividing its
# residual effects by the standard deviation (found by probing).
_CORRELATION_VARIANCE_FLOOR = 1e-8
# fry's posterior standardization moderates the trimmed mean square as a variance on
# this fraction of the residual degrees of freedom (limma's rule, found by probing).
_TRIMMED_DF_FRACTION = 0.92
# Composite Gauss-Legendre rule over [0, 12] for E[max of k chi-squared(1)]: beyond 12
# the integrand, 2 z (1 - (1 - 2 Q(z))^k), is below 1e-21 for any k up to 1e9.
_PANELS = 48
_PANEL_WIDTH = 0.25
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(16)


def camera(
    data: Any,
    gene_sets: Mapping[str, Any],
    *,
    group: str | None = None,
    contrast: Any = None,
    design: Any = None,
    id_column: str = "feature_id",
    feature_ids: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    inter_gene_correlation: float | None = 0.01,
    allow_negative_correlation: bool = False,
    use_ranks: bool = False,
    trend: bool | None = None,
    robust: bool | None = None,
) -> Table:
    """Competitive test of gene sets for one contrast, allowing for their correlation.

    limma's ``camera`` (Wu and Smyth 2012): tests whether the genes of each set are more
    differentially expressed for ``contrast`` than the other genes, by a two-sample test
    of their moderated z-scores against those of the other genes, with the variance of
    the mean of a set of m genes inflated by VIF = 1 + (m - 1) rho for a correlation
    rho between them.
    https://docs.nygen.io/CytePool/statistics/gene-set-tests#camera explains the test,
    the module documentation the conventions that match limma, and
    https://docs.nygen.io/CytePool/statistics/gene-set-tests#gene-set-tests-limits the
    differences from it. The default correlation of 0.01 is limma's (since limma
    3.29.6): it ranks co-regulated, interpretable sets higher than a correlation
    estimated for each set, whose error rate control is strict.

    A result of :func:`~cytepool.limma_voom` or :func:`~cytepool.limma_trend` is tested
    as limma's camera tests the fit of one group: its values and weights, the design
    rows of its samples and the contrast's weights on them, as
    :meth:`~cytepool.DifferentialExpressionResult.fit` gives them, and the moderation
    the result used. ``camera(result, sets, id_column="symbol")`` names the genes by
    the symbols of the result's features.

    Args:
        data: A :class:`~cytepool.DifferentialExpressionResult` of limma-voom or
            limma-trend; a :class:`~cytepool.stats.voom.VoomResult`, whose log2-CPM
            values are tested with its weights; or ``(n_samples, n_genes)``
            log-expression values, finite.
        gene_sets: Mapping of set name to its genes: gene ids (matched to the ids of
            the data; ids not among them are left out, as limma's ``ids2indices``
            does), 0-based positions of columns of an array when ``feature_ids`` is
            not given, or a boolean mask over the genes. A gene listed twice counts
            once. A :class:`~cytepool.GeneSets`, such as one read from a GMT file by
            :func:`~cytepool.read_gmt`, is such a mapping, of gene ids.
        group: For a result, the group (cell type) whose fit is tested; it may be
            left out when the result has one.
        contrast: The contrast to test. For a result, a contrast name of the result,
            which may be left out when it tests one. For an array or a VoomResult:
            the name of a named contrast of a ``Design``, the name or 0-based
            position of a design column (a coefficient; negative positions count
            from the last column), or weights on the design columns: a vector in
            their order, a mapping of column name to weight (columns left out weigh
            0), a pandas Series indexed by them, or a ``Contrasts`` holding one
            contrast.
        design: For an array or a VoomResult, ``(n_samples, n_coefficients)``
            design matrix of full column rank, or anything else that
            ``fit_linear_model`` accepts, such as a :class:`~cytepool.Design`. It
            needs residual degrees of freedom.
        id_column: For a result, the column of its features whose ids the sets name,
            such as ``"feature_id"`` (the default) or ``"symbol"``.
        feature_ids: For an array or a VoomResult, one id per gene (column),
            unique, to name genes in ``gene_sets`` by id.
        weights: For an array, precision weights, positive and finite:
            ``(n_samples,)`` for sample weights or ``(n_samples, n_genes)`` for
            observation weights. limma's per-gene weights (a vector of one weight
            per gene) have no counterpart.
        inter_gene_correlation: The correlation between the genes of every set, in
            (-1, 1), or None to estimate it for each set from its residuals.
        allow_negative_correlation: Whether a negative correlation, given or
            estimated, may shrink the variance of a set's mean; otherwise it counts as
            zero.
        use_ranks: Whether to test the ranks of the z-scores (a correlation-adjusted
            Wilcoxon-Mann-Whitney test) rather than their values. Where the
            continuity correction leaves both tails above 1/2, the p-value is 1;
            limma reports twice the smaller tail, above 1.
        trend: Whether the prior variance of the moderation follows the mean
            log-expression of the genes (limma's ``trend.var``); None takes the
            moderation of a result (limma-trend's trend), and False otherwise.
        robust: Whether to moderate the variances with the robust estimator
            (Phipson et al. 2016), which limma's camera does not offer; each gene's
            moderated t-statistic then becomes a z-score on its own degrees of
            freedom. None takes the result's setting, and False otherwise.

    Returns:
        A table with one row per gene set, in the order of ``gene_sets``: ``gene_set``;
        ``n_genes``, the number of the set's genes among those tested; ``correlation``,
        the estimated inter-gene correlation (NaN for a set of one gene) or the given
        one; ``direction``, ``"up"`` or ``"down"`` (None where the p-value is NaN);
        ``p_value``, two-sided; and ``adjusted_p_value``, Benjamini-Hochberg over the
        sets (Benjamini and Hochberg 1995). A set with no genes, or with every gene,
        has NaN p-values and no direction, and the adjustment leaves it out; limma
        reports "Up" for it, and in the rank test the p-value 2, which it adjusts
        with the others.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the values are not finite or have fewer than three genes, the
            design is missing for an array, does not have one row per sample, is not
            of full column rank or leaves no residual degrees of freedom, the
            contrast or a gene set is invalid, a weight is not positive and finite,
            no gene id of any set is among the data's ids, an argument applies to
            another kind of data, or the result is from ``dream_voom``.
        KeyError: If a result has no such group, contrast or column of its features,
            or the group cannot estimate the contrast.
    """
    rho = _correlation_option(inter_gene_correlation)
    allow_negative = checked_flag(
        allow_negative_correlation, "allow_negative_correlation"
    )
    ranks = checked_flag(use_ranks, "use_ranks")
    read = gene_set_data(
        data,
        group=group,
        contrast=contrast,
        design=design,
        id_column=id_column,
        feature_ids=feature_ids,
        weights=weights,
        trend=trend,
        robust=robust,
        function="camera",
    )
    values, precision = read.values, read.precision
    with_trend, with_robust = bool(read.trend), bool(read.robust)
    n_genes = values.shape[1]
    if n_genes < 3:
        raise ValueError(
            "camera compares the genes of each set with the other genes and needs at "
            f"least 3 genes, got {n_genes}"
        )
    members = _gene_set_members(gene_sets, read.ids, n_genes, read.ids_name)
    effects, mean_expression, d = _contrast_effects(
        values, read.design, read.contrast, precision, "camera"
    )
    residual = effects[:, 1:]
    variances = np.sum(residual**2, axis=1) / d
    squeezed = squeeze_variances(
        variances,
        d,
        covariate=mean_expression if with_trend else None,
        robust=with_robust,
    )
    df_prior = np.broadcast_to(squeezed.df_prior, variances.shape)
    df_total = np.minimum(d + df_prior, float(d * n_genes))
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
    data: Any,
    gene_sets: Mapping[str, Any],
    *,
    group: str | None = None,
    contrast: Any = None,
    design: Any = None,
    id_column: str = "feature_id",
    feature_ids: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    standardize: str = "posterior_sd",
    trend: bool | None = None,
    robust: bool | None = None,
) -> Table:
    """Self-contained test of gene sets for one contrast, without rotations.

    limma's ``fry`` (Giner and Smyth 2016): tests whether the genes of each set are
    differentially expressed for ``contrast``, in the same direction (the directional
    p-value) or in either (the mixed p-value), by the limit of the rotation test of Wu
    et al. (2010) with the mean statistic over infinitely many rotations: the mean b of
    the set's standardized effects gives T = b_0 / (sum_k b_k^2 / d)^(1/2) on d degrees
    of freedom, and the mixed p-value is a beta approximation to the rotation p-value
    of the sum of the set's squared standardized contrast effects, from all d + 1
    dimensions of the rotation, also when the set has 2 to d genes, where limma uses
    only the set's own m. https://docs.nygen.io/CytePool/statistics/gene-set-tests#fry
    explains the test, the module documentation the conventions, and
    https://docs.nygen.io/CytePool/statistics/gene-set-tests#gene-set-tests-limits the
    differences from limma. limma's gene contribution weights (``gene.weights``, or
    weights inside a set), which give genes signs and sizes in the set statistic, are
    not implemented. A result of :func:`~cytepool.limma_voom` or
    :func:`~cytepool.limma_trend` is tested as for :func:`camera`.

    Args:
        data: A differential-expression result, a VoomResult or log-expression
            values, as for :func:`camera`.
        gene_sets: Mapping of set name to its genes, as for :func:`camera`.
        group: For a result, the group whose fit is tested, as for :func:`camera`.
        contrast: The contrast to test, as for :func:`camera`.
        design: For an array or a VoomResult, the design, as for :func:`camera`.
        id_column: For a result, the column of its features whose ids the sets name.
        feature_ids: For an array or a VoomResult, one id per gene, unique, to name
            genes in ``gene_sets`` by id.
        weights: For an array, precision weights, as for :func:`camera`.
        standardize: How each gene's effects are scaled before they are averaged:
            ``"posterior_sd"``, by the square root of a posterior variance that combines
            the prior of the empirical Bayes moderation with the mean square of the
            gene's d + 1 effects without the largest (limma's ``"posterior.sd"``);
            ``"residual_sd"``, by the square root of that mean square alone (limma's
            ``"residual.sd"``); or ``"none"``, unscaled, which suits genes of equal
            variance.
        trend: Whether the prior variance follows the mean log-expression of the
            genes. Only with ``standardize="posterior_sd"``; None takes the
            moderation of a result, and False otherwise.
        robust: Whether to estimate the prior robustly. Only with
            ``standardize="posterior_sd"``; None takes the result's setting, and
            False otherwise.

    Returns:
        A table with one row per gene set, in the order of ``gene_sets``: ``gene_set``;
        ``n_genes``; ``direction``, ``"up"`` or ``"down"`` (None where the p-value is
        NaN); ``p_value``, the two-sided directional p-value; ``adjusted_p_value``,
        Benjamini-Hochberg over the sets; ``mixed_p_value``, the non-directional
        p-value; and ``mixed_adjusted_p_value``. A set with no genes has NaN p-values,
        and so, with ``standardize="residual_sd"``, has a set containing a gene whose
        d + 1 effects are zero but for at most one, which cannot be scaled; such a set
        has no direction, and the adjustment leaves it out. For a set with no genes,
        limma reports "Up" and a mixed p-value of 0.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: As :func:`camera` (one gene suffices), if ``standardize`` is
            unknown, or if ``trend`` or ``robust`` is given with another
            standardization, which has no prior.
        KeyError: As :func:`camera`.
    """
    if not isinstance(standardize, str) or standardize not in _STANDARDIZATIONS:
        raise ValueError(
            "standardize must be one of "
            f"{', '.join(map(repr, _STANDARDIZATIONS))}, got {standardize!r}"
        )
    given_trend = None if trend is None else checked_flag(trend, "trend")
    given_robust = None if robust is None else checked_flag(robust, "robust")
    if (given_trend or given_robust) and standardize != "posterior_sd":
        raise ValueError(
            "trend and robust shape the prior of the posterior standard deviations, so "
            "they apply only with standardize='posterior_sd'"
        )
    read = gene_set_data(
        data,
        group=group,
        contrast=contrast,
        design=design,
        id_column=id_column,
        feature_ids=feature_ids,
        weights=weights,
        trend=given_trend,
        robust=given_robust,
        function="fry",
    )
    values, precision = read.values, read.precision
    # A result's moderation shapes the prior, which only posterior_sd has.
    posterior = standardize == "posterior_sd"
    with_trend = posterior and bool(read.trend)
    with_robust = posterior and bool(read.robust)
    members = _gene_set_members(gene_sets, read.ids, values.shape[1], read.ids_name)
    effects, mean_expression, d = _contrast_effects(
        values, read.design, read.contrast, precision, "fry"
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
                np.broadcast_to(squeezed.df_prior, trimmed.shape),
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
            "mixed_adjusted_p_value": adjust_p_values(mixed, "bh"),
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
    reparametrised, d = contrast_design(values, design, contrast, function)
    return gene_effects(values, reparametrised, weights), observed_means(values), d


# ---------------------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------------------


def _hill_z_scores(t: FloatArray, df: FloatArray) -> FloatArray:
    """Hill's (1970) normal approximation to t-statistics on df degrees of freedom.

    z = w (1 + (w^2 + 3) / b - (4 w^6 + 33 w^4 + 240 w^2 + 855) / (10 b (b + 0.8 w^4
    + 100))) with w^2 = (df - 1/2) log(1 + t^2 / df) and b = 48 (df - 1/2)^2 (Brophy
    1987, equation 5). Infinite t stay infinite.
    """
    # camera's z-scores, as limma's zscoreT(approx = TRUE, method = "hill") gives them
    # (found by probing).
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
    # The exact expectation; limma's is about 7.3e-5 (d + 1) / d above it (found by
    # probing), a deliberate difference.
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
    # limma's approximation for sets of more than d genes (found by probing). For
    # sets of 2 to d genes limma uses only the set's m nonzero eigenvalues, as an
    # m-dimensional problem (found by probing); all d + 1 are used here, on purpose.
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


def _gene_set_members(
    gene_sets: Any, gene_ids: Any, n_genes: int, ids_name: str
) -> list[tuple[str, IntArray]]:
    """Each set's name and the sorted positions of its genes among the genes of the
    data."""
    index = checked_gene_sets(gene_sets, gene_ids, n_genes, ids_name)
    members: list[tuple[str, IntArray]] = []
    matched = 0
    for name, genes in gene_sets.items():
        positions, found = set_positions(genes, set_label(name), index, n_genes)
        matched += found
        members.append((name, positions))
    check_matched(index, matched, ids_name)
    return members


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
