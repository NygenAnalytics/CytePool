"""Rotation gene-set tests of one contrast of a linear model: limma's roast, mroast and
romer.

A rotation test (Langsrud 2005) draws random rotations of the residual space of the
linear model fitted to every gene, which leave the joint distribution of the data
unchanged under the null hypothesis, and compares a set statistic with its values
under the rotations. :func:`roast` and :func:`mroast` are self-contained tests of one
gene set and of many (Wu et al. 2010): whether the genes of a set are differentially
expressed for the contrast at all. :func:`romer` is a competitive test of many sets
(Majewski et al. 2010): whether the genes of a set rank higher than the other genes.
:func:`~cytepool.stats.gene_set_tests.fry` gives the limit of ``mroast`` with the mean
statistic over infinitely many rotations without drawing any. Results match limma
3.62.1 within Monte Carlo error, and exactly when CytePool is given limma's rotations,
except where noted under "Deliberate differences".

Effects and rotations:
    Each gene's contrast effect u_0 and residual effects u_1, ..., u_d are those of
    :mod:`cytepool.stats.gene_set_tests`: coordinates of its data, whitened by its
    precision weights, in the basis of the Householder QR decomposition of the design
    reparametrised so that the contrast is its last coefficient. Under the null
    hypothesis they are independent with variance sigma^2, so one rotation applied to
    the (d + 1)-vectors of every gene leaves the joint distribution of the data, and
    the correlation between genes, unchanged (Wu et al. 2010, section 4.2). A
    rotation is given by its first row r, a point drawn uniformly from the unit
    sphere in d + 1 dimensions (a vector of standard normal draws divided by its
    length): a gene's rotated contrast effect is u* = r'u and its rotated residual
    sum of squares |u|^2 - u*^2. The rotated moderated t-statistic of a gene uses
    the prior variance s_0^2 and prior df d_0 estimated once from the observed
    residual variances of all genes (Wu et al. 2010, section 2.4): t* = u* / s*, with
    s*^2 = (d_0 s_0^2 + |u|^2 - u*^2) / (d_0 + d). The rotations come from
    ``rng.standard_normal``, 4,096 at a time, which bounds the memory they take; numpy
    gives the same stream however its draws are split, so a seeded generator gives the
    same p-values every time. limma draws its own rotations from R's generator, so the
    two programs agree within Monte Carlo error.

roast and mroast:
    Each gene's moderated t becomes a z-score by Bailey's (1980) normalizing
    transformation, limma's default, z = (8 nu + 1) / (8 nu + 9) (nu + 19/12)^(1/2)
    log(1 + t^2 / (nu + 1/12))^(1/2) with the sign of t (limma's ``zscoreT(approx =
    TRUE)`` to 1e-14, found by probing), on nu = min(d_0 + d, 10^4) degrees of freedom,
    limma's limit (found by probing). With an infinite prior df the exact z-score is t
    itself, which the limit lowers by about (t^2 + 1) / (4 10^4) of its value
    (1.2e-4 at t = 2, 2.5e-3 at t = 10); the same transformation applies to the
    observed and the rotated statistics, so the test remains exact. With contribution
    weights a_g (1 unless given; a gene of weight zero leaves the set), A = sum |a_g|
    and m genes in the set, the set statistics of Wu et al. (2010, section 3.3) are, for
    the up, down and mixed alternatives:

    - ``"mean"``: sum a_g z_g / A, its negative, and sum |a_g z_g| / A.
    - ``"floormean"``: sum max(a_g z_g, 0) / A, sum max(-a_g z_g, 0) / A, and
      sum |a_g| max(|z_g|, c) / A, with c = 0.674, the square root of the median of
      the chi-squared distribution on 1 df.
    - ``"mean50"``: the mean of the h largest a_g z_g, of the h largest -a_g z_g, and
      of the h largest |a_g z_g|, with h = ceiling(m / 2), the smallest integer at least
      half the number of genes, as the paper's text and limma have it (its formula,
      ceiling((m + 1) / 2), is larger for even m).
    - ``"msq"``: sum |a_g| z_g^2 / A over the genes with a_g z_g > 0, over those with
      a_g z_g < 0, and over all genes.

    The two-sided directional statistic is the larger of the up and down statistics.
    A rotated statistic at least as large as the observed one counts as at least as
    extreme, and the p-value is (b + 1) / (B + 1) for b such rotations out of B
    (Phipson and Smyth 2010). For the up and down alternatives each rotation r also
    gives -r, which negates every rotated z-score, so b counts 2B rotated statistics
    and the p-value is (b + 1) / (2B + 1), as in limma. A gene is active for the up
    alternative when sign(a_g) z_g > sqrt(2), for the down alternative when
    sign(a_g) z_g < -sqrt(2), and for the mixed alternative when |z_g| > sqrt(2) (Wu
    et al. 2010, section 3.5); the two-sided alternative reports the larger of the up
    and down proportions, as limma does. mroast's direction is "up" when the up
    p-value is smaller than the down p-value and "down" otherwise (found by
    probing), and its adjusted p-values are Benjamini and Hochberg's, applied to the
    mid-p-values p - 1 / (2 (B + 1)) (Routledge 1994) and then raised to at least the
    p-value, as limma does by default. All sets share the same rotations.

romer:
    Every gene's observed moderated t-statistic is ranked among all genes (ties get
    their average rank), and a set's statistic is the mean rank of its genes: the
    ranks of t for the up alternative, of -t for the down alternative and of |t| for
    the mixed alternative (``"mean"``); the same after setting negative values to
    zero, and for the mixed alternative |t| below 1 to 1 (``"floormean"``, the floor
    of 1 found by probing); or the mean of the ceiling(m / 2) largest ranks of the set
    (``"mean50"``). Each rotation ranks the rotated t-statistics of all genes again.
    The p-value of each alternative is (b + 1) / (B + 1), where b counts the rotations
    whose statistic is at least the observed one.

    Before rotating, each gene's contrast effect is shrunk, so that the
    differential expression of the gene does not inflate its rotated statistics
    (limma's ``shrink.resid``; formula found by probing): u_0 becomes
    u_0 / (1 + P_g v_0 / v)^(1/2). P_g is the posterior probability that the gene is
    differentially expressed and v_0 the prior variance of the contrast for such genes,
    from the log-odds (B-statistics) of
    :func:`~cytepool.stats.empirical_bayes.moderated_statistics` with the t-statistics
    on d_0 + d degrees of freedom (their limit for an infinite prior df) and 1 - pi_0
    differentially expressed genes, and v is the unscaled variance of the contrast.
    The proportion of true null hypotheses pi_0 is (2 / (n + 1)) sum min(p_(i), i / n)
    over the n ordered p-values p_(i) of the moderated t-statistics (limma's
    ``propTrueNull(method = "lfdr")``, found by probing). The rotated residual sum of
    squares remains |u|^2 - u*^2 with |u| the length of the gene's unshrunk effects,
    as in limma (found by probing). romer multiplies the residual effects by the sign
    of the last diagonal element of R, as limma's romer does and its roast does not
    (found by probing); a uniformly random rotation does not depend on the basis of the
    residual space, but limma's rotations then mean the same here. The bounds of v_0
    are those of ``moderated_statistics`` (``stdev_coef_lim``), in log-expression
    units, so, as in limma, romer's p-values can change when the data are rescaled.

Deliberate differences from limma:
    - roast and mroast count a rotated statistic equal to the observed one as at least
      as extreme, as Wu et al. (2010) and Phipson and Smyth (2010) define the p-value
      (and as limma's romer does); limma's roast counts only larger ones. The two
      differ only where the observed statistic is the smallest the statistic can
      take, which happens with positive probability for ``"floormean"`` and
      ``"msq"``: a set with no gene on one side of zero has a one-sided p-value of 1
      here and of about 1/2 in limma, and with ``"floormean"`` a set whose |z| are
      all below 0.674 has a mixed p-value of 1 here.
    - ``"msq"`` with contribution weights follows Wu et al. (2010). With weights of
      both signs, limma's up, down and two-sided msq p-values differ from it, and with
      more than 1,000 rotations (limma draws them 1,000 at a time) its msq p-values
      change when every weight is multiplied by the same number (found by probing).
      Its mixed msq p-values, and those of sets whose weights have one sign, agree with
      CytePool's for up to 1,000 rotations.
    - ``"floormean"``'s mixed statistic weighs each gene by |a_g|, as the paper's
      mixed statistics do; limma weighs it by a_g, so a negative weight turns a gene
      with a large |z| into evidence against the set (a one-gene set's mixed p-value
      goes from 0.003 to 0.997 when its weight changes sign).
    - A gene of contribution weight zero leaves the set: it counts neither in
      ``n_genes`` nor among the values that ``"mean50"`` averages. limma keeps it in
      both (and leaves it out of the active proportions, as CytePool does).
    - romer's posterior probability P_g is 1 / (1 + exp(-lods)). limma computes
      exp(lods) / (1 + exp(lods)), which is NaN beyond lods = 709.8 (|t| above about
      38 with an infinite prior df); the gene's shrunk contrast effect is then NaN and
      the p-values of every set change (found by probing).
    - A set without genes, or whose genes all have weight zero, has NaN p-values and
      no direction and is left out of the adjustment of the other sets; limma reports
      "down" and NA, or the smallest possible p-value when all weights are zero. A
      gene listed twice counts once; limma counts it twice.
    - mroast tests every set with the same rotations; limma draws new rotations for
      each set. Each set's p-value is a valid rotation p-value either way.
    - A z-score is computed without overflow for |t| above 1e154, where limma's is
      infinite.

References:
    Wu D, Lim E, Vaillant F, Asselin-Labat ML, Visvader JE, Smyth GK (2010). ROAST:
    rotation gene set tests for complex microarray experiments. Bioinformatics 26,
    2176-2182. doi:10.1093/bioinformatics/btq401

    Majewski IJ, Ritchie ME, Phipson B, Corbin J, Pakusch M, Ebert A, Busslinger M,
    Koseki H, Hu Y, Smyth GK, Alexander WS, Hilton DJ, Blewitt ME (2010). Opposing roles
    of polycomb repressive complexes in hematopoietic stem and progenitor cells. Blood
    116, 731-739. doi:10.1182/blood-2009-12-260760

    Langsrud O (2005). Rotation tests. Statistics and Computing 15, 53-60.
    doi:10.1007/s11222-005-4789-5

    Phipson B, Smyth GK (2010). Permutation p-values should never be zero: calculating
    exact p-values when permutations are randomly drawn. Statistical Applications in
    Genetics and Molecular Biology 9, article 39. doi:10.2202/1544-6115.1585

    Bailey BJR (1980). Accurate normalizing transformations of a Student's t variate.
    Journal of the Royal Statistical Society C 29, 304-306. doi:10.2307/2346906

    Routledge RD (1994). Practicing safe statistics with the mid-p. Canadian Journal of
    Statistics 22, 103-110. doi:10.2307/3315826

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x

    Hoare CAR (1961). Algorithm 65: find. Communications of the ACM 4, 321-322.
    doi:10.1145/366622.366647

    Wirth N (1976). Algorithms + Data Structures = Programs. Prentice-Hall, Englewood
    Cliffs.
"""

import math
import numbers
import operator
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from numba import prange

from cytepool._jit import by_chunks, jit, thread_limit
from cytepool._masked_arrays import reject_masked, unmasked_array
from cytepool._options import checked_flag, positive_integer
from cytepool.stats.distributions import t_sf
from cytepool.stats.empirical_bayes import (
    _coefficient_prior_variance,
    _log_odds,
    squeeze_variances,
)
from cytepool.stats.gene_set_tests import (
    _check_matched,
    _checked_gene_sets,
    _contrast_design,
    _effects,
    _expression,
    _id_positions,
    _positions,
    _set_label,
    _set_positions,
)
from cytepool.stats.linear_models import _observed_means
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.table import Table

__all__ = ["mroast", "roast", "romer"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

_ROAST_STATISTICS = ("mean", "floormean", "mean50", "msq")
_ROMER_STATISTICS = ("mean", "floormean", "mean50")
_MEAN, _FLOORMEAN, _MEAN50, _MSQ = 0, 1, 2, 3
# floormean's floor for the mixed alternative of roast: the square root of the median
# of the chi-squared distribution on 1 df, Phi^-1(3/4) (Wu et al. 2010, section 3.3).
_MIXED_FLOOR = 0.6744897501960817
# romer's floor for |t| in the mixed alternative of floormean (found by probing).
_ROMER_MIXED_FLOOR = 1.0
# A gene is active when its z-score exceeds sqrt(2) in size (Wu et al. 2010, 3.5).
_ACTIVE = math.sqrt(2.0)
# Above this |t|, t^2 would overflow; log(1 + t^2 / b) is then 2 log|t| - log b.
_LARGE_T = 1e150
# roast's z-scores use at most this many degrees of freedom, as limma's do (found by
# probing); romer's log-odds use d_0 + d.
_DF_LIMIT = 1e4
# Rotations drawn and processed at a time.
_ROTATION_BATCH = 4096
# eBayes's limits on the standard deviation of a non-zero coefficient (stdev.coef.lim).
_COEFFICIENT_LIMITS = (0.1, 4.0)


def roast(
    y: Any,
    design: Any,
    contrast: Any,
    gene_set: Any,
    *,
    gene_ids: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    gene_weights: npt.ArrayLike | None = None,
    set_statistic: str = "mean",
    n_rotations: int = 1999,
    trend: bool = False,
    robust: bool = False,
    rng: np.random.Generator | int | None = None,
    n_threads: int | None = None,
) -> Table:
    """Self-contained rotation test of one gene set for one contrast.

    limma's ``roast`` (Wu et al. 2010): tests whether the genes of the set are
    differentially expressed for ``contrast``, for each alternative: down, up, either
    (two-sided) and mixed (in either direction, gene by gene), by comparing a set
    statistic of the genes' moderated z-scores with its values under random rotations
    of the residual space. The module docstring gives the method and the deliberate
    differences from limma. With many sets, :func:`mroast` reports one row per set.

    Args:
        y: ``(n_samples, n_genes)`` log-expression values, finite, or a
            :class:`~cytepool.stats.voom.VoomResult`, whose log-CPM values are tested
            with its weights. All genes are used to estimate the prior of the
            residual variances.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank, or
            anything else that ``fit_linear_model`` accepts, such as a
            :class:`~cytepool.design.Design`. It needs residual degrees of freedom.
        contrast: The contrast to test, as for
            :func:`~cytepool.stats.gene_set_tests.camera`: the name of a named
            contrast of a ``Design``, the name or 0-based position of a design
            column, or weights on the design columns.
        gene_set: The set's genes: gene ids (matched to ``gene_ids``; ids not among
            them are left out), 0-based positions of columns of ``y`` when
            ``gene_ids`` is not given, or a boolean mask over the genes; or a mapping
            of gene (id or position) to its contribution weight. A gene listed twice
            counts once, and a gene of weight zero leaves the set.
        gene_ids: One id per gene (column of ``y``), unique, to name genes by id.
        weights: Precision weights, positive and finite: ``(n_samples,)`` for sample
            weights or ``(n_samples, n_genes)`` for observation weights. Not with a
            ``VoomResult``, which carries its own.
        gene_weights: ``(n_genes,)`` contribution weights, finite, one per gene: a
            negative weight marks a gene expected to change in the opposite
            direction to the set, and larger weights count more in the set
            statistic. Not with a set that gives its own weights.
        set_statistic: ``"mean"`` (limma's default), ``"floormean"``, ``"mean50"``
            or ``"msq"``; see the module docstring. The mean detects sets in which
            most genes change in the same direction; the others detect sets in which
            fewer genes change, msq the fewest.
        n_rotations: Number of rotations B, at least 1. The smallest p-value is
            1 / (B + 1), and 1 / (2B + 1) for the up and down alternatives.
        trend: Whether the prior variance follows the mean log-expression of the
            genes (limma's ``trend``).
        robust: Whether to estimate the prior robustly (limma's ``robust``).
        rng: A ``numpy.random.Generator`` to draw the rotations from, or an integer
            seed for one, or None for a fresh, unseeded generator.
        n_threads: The most threads to use for the rotations; ``None`` uses numba's
            current setting. The result does not depend on it.

    Returns:
        A table with one row per alternative, ``"down"``, ``"up"``, ``"up_or_down"``
        (two-sided) and ``"mixed"``: ``alternative``; ``n_genes``, the number of the
        set's genes among those tested; ``active_proportion``, the proportion of the
        set's genes active for that alternative; and ``p_value``. A set without genes
        has NaN proportions and p-values.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If ``y`` is not finite, the design does not have one row per
            sample, is not of full column rank or leaves no residual degrees of
            freedom, the contrast or the set is invalid, a weight is not positive and
            finite, a contribution weight is not finite, an option is invalid, or no
            gene id of the set is among ``gene_ids``.
    """
    reject_masked(gene_set, "gene_set")
    tests = _roast_tests(
        y,
        design,
        contrast,
        {"gene_set": gene_set},
        gene_ids=gene_ids,
        weights=weights,
        gene_weights=gene_weights,
        set_statistic=set_statistic,
        n_rotations=n_rotations,
        trend=trend,
        robust=robust,
        rng=rng,
        n_threads=n_threads,
        function="roast",
    )
    down, up, mixed = tests.active[0]
    return Table(
        {
            "alternative": ["down", "up", "up_or_down", "mixed"],
            "n_genes": np.full(4, tests.n_genes[0], dtype=np.int64),
            "active_proportion": np.array([down, up, max(down, up), mixed]),
            "p_value": tests.p_values[0],
        }
    )


def mroast(
    y: Any,
    design: Any,
    contrast: Any,
    gene_sets: Mapping[str, Any],
    *,
    gene_ids: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    gene_weights: npt.ArrayLike | None = None,
    set_statistic: str = "mean",
    n_rotations: int = 1999,
    trend: bool = False,
    robust: bool = False,
    mid_p: bool = True,
    rng: np.random.Generator | int | None = None,
    n_threads: int | None = None,
) -> Table:
    """Self-contained rotation tests of many gene sets for one contrast.

    limma's ``mroast``: :func:`roast` for each set, all sets with the same rotations,
    and with their p-values adjusted for testing many sets. The module docstring gives
    the method and the deliberate differences from limma.

    Args:
        y: Log-expression values or a ``VoomResult``, as for :func:`roast`.
        design: The design, as for :func:`roast`.
        contrast: The contrast to test, as for :func:`roast`.
        gene_sets: Mapping of set name to its genes, each as ``gene_set`` of
            :func:`roast`: gene ids, positions or a boolean mask, or a mapping of
            gene to its contribution weight.
        gene_ids: One id per gene, unique, to name genes by id.
        weights: Precision weights, as for :func:`roast`.
        gene_weights: ``(n_genes,)`` contribution weights, as for :func:`roast`; not
            with sets that give their own weights.
        set_statistic: ``"mean"``, ``"floormean"``, ``"mean50"`` or ``"msq"``.
        n_rotations: Number of rotations B, at least 1.
        trend: Whether the prior variance follows the mean log-expression.
        robust: Whether to estimate the prior robustly.
        mid_p: Whether the adjustment uses mid-p-values, p - 1 / (2 (B + 1)), as
            limma does by default (limma's ``midp``), which suit false discovery
            rates; the adjusted p-values are then at least the p-values.
        rng: A ``numpy.random.Generator``, an integer seed, or None.
        n_threads: The most threads to use for the rotations.

    Returns:
        A table with one row per gene set, in the order of ``gene_sets``:
        ``gene_set``; ``n_genes``; ``proportion_down`` and ``proportion_up``, the
        proportions of the set's genes active down and up; ``direction``, ``"up"`` or
        ``"down"`` (None where the p-value is NaN); ``p_value``, the two-sided
        directional p-value; ``adjusted_p_value``, Benjamini-Hochberg over the sets
        (Benjamini and Hochberg 1995), limma's default (its other ``adjust.method``
        choices are not offered);
        ``mixed_p_value``, the p-value of the mixed alternative; and
        ``adjusted_mixed_p_value``. A set without genes has NaN proportions and
        p-values.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: As :func:`roast`, or if ``gene_sets`` is empty or no gene id of
            any set is among ``gene_ids``.
    """
    use_mid_p = checked_flag(mid_p, "mid_p", expected="a bool")
    tests = _roast_tests(
        y,
        design,
        contrast,
        gene_sets,
        gene_ids=gene_ids,
        weights=weights,
        gene_weights=gene_weights,
        set_statistic=set_statistic,
        n_rotations=n_rotations,
        trend=trend,
        robust=robust,
        rng=rng,
        n_threads=n_threads,
        function="mroast",
    )
    down, up, two_sided, mixed = tests.p_values.T
    direction: list[str | None] = [
        None if math.isnan(p) else ("up" if p_up < p_down else "down")
        for p, p_up, p_down in zip(
            two_sided.tolist(), up.tolist(), down.tolist(), strict=True
        )
    ]
    shift = 0.5 / (tests.n_rotations + 1) if use_mid_p else 0.0
    return Table(
        {
            "gene_set": tests.names,
            "n_genes": tests.n_genes,
            "proportion_down": tests.active[:, 0],
            "proportion_up": tests.active[:, 1],
            "direction": direction,
            "p_value": two_sided,
            "adjusted_p_value": _adjusted(two_sided, shift),
            "mixed_p_value": mixed,
            "adjusted_mixed_p_value": _adjusted(mixed, shift),
        }
    )


def romer(
    y: Any,
    design: Any,
    contrast: Any,
    gene_sets: Mapping[str, Any],
    *,
    gene_ids: npt.ArrayLike | None = None,
    weights: npt.ArrayLike | None = None,
    set_statistic: str = "mean",
    n_rotations: int = 9999,
    shrink_contrast_effects: bool = True,
    rng: np.random.Generator | int | None = None,
    n_threads: int | None = None,
) -> Table:
    """Competitive rotation test of gene sets by the ranks of their genes.

    limma's ``romer`` (Majewski et al. 2010): tests whether the genes of each set
    rank higher than the other genes for ``contrast``, by their moderated
    t-statistics, up, down or in either direction gene by gene (mixed), comparing
    the set's mean rank with its values when every gene's rotated statistics are
    ranked again. Like gene set enrichment analysis it pits the sets against one
    another, but it uses rotations instead of permuting samples, so it works for any
    contrast of a linear model with residual degrees of freedom. The module docstring
    gives the method.

    Args:
        y: ``(n_samples, n_genes)`` log-expression values, finite. All genes are
            ranked.
        design: The design, as for :func:`roast`.
        contrast: The contrast to test, as for :func:`roast`.
        gene_sets: Mapping of set name to its genes: gene ids (matched to
            ``gene_ids``; ids not among them are left out), 0-based positions, or a
            boolean mask over the genes. A gene listed twice counts once.
        gene_ids: One id per gene, unique, to name genes by id.
        weights: ``(n_samples,)`` sample weights, positive and finite (limma's
            ``array.weights``). romer has no observation weights.
        set_statistic: ``"mean"`` (limma's default), ``"floormean"`` or ``"mean50"``;
            see the module docstring.
        n_rotations: Number of rotations B, at least 1; limma's default is 9999. The
            smallest p-value is 1 / (B + 1).
        shrink_contrast_effects: Whether each gene's contrast effect is shrunk before
            rotating, as limma does by default (limma's ``shrink.resid``).
        rng: A ``numpy.random.Generator`` to draw the rotations from, or an integer
            seed for one, or None for a fresh, unseeded generator.
        n_threads: The most threads to use for the rotations; ``None`` uses numba's
            current setting. The result does not depend on it.

    Returns:
        A table with one row per gene set, in the order of ``gene_sets``:
        ``gene_set``; ``n_genes``; and ``up_p_value``, ``down_p_value`` and
        ``mixed_p_value``, the p-values of the three alternatives, which, as in
        limma, are not adjusted for testing many sets
        (:func:`~cytepool.stats.multiple_testing.adjust_p_values` adjusts them). A
        set without genes has NaN p-values.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: As :func:`mroast`, or if ``y`` is a ``VoomResult`` or
            ``weights`` are observation weights, which romer does not take.
    """
    statistic = _statistic(set_statistic, _ROMER_STATISTICS)
    n_rotations = _rotation_count(n_rotations)
    shrink = checked_flag(
        shrink_contrast_effects, "shrink_contrast_effects", expected="a bool"
    )
    generator = _generator(rng)
    with thread_limit(n_threads):
        return _romer_sets(
            y,
            design,
            contrast,
            gene_sets,
            gene_ids,
            weights,
            statistic,
            n_rotations,
            shrink,
            generator,
        )


def _romer_sets(
    y: Any,
    design: Any,
    contrast: Any,
    gene_sets: Any,
    gene_ids: Any,
    weights: Any,
    statistic: int,
    n_rotations: int,
    shrink: bool,
    generator: np.random.Generator,
) -> Table:
    """:func:`romer` once its options are checked."""
    values, precision = _expression(y, weights)
    if precision is not None and precision.ndim != 1:
        raise ValueError(
            "romer takes sample weights only, of shape (n_samples,); limma's romer "
            "has no observation weights, so a VoomResult or a matrix of weights "
            "cannot be used"
        )
    index = _checked_gene_sets(gene_sets, gene_ids, values.shape[1])
    names, members = [], []
    matched = 0
    for name, genes in gene_sets.items():
        label = _set_label(name)
        if isinstance(genes, Mapping):
            raise TypeError(
                f"{label} maps genes to contribution weights, which romer does not "
                "take; give the set's genes as a list"
            )
        positions, found = _set_positions(genes, label, index, values.shape[1])
        names.append(name)
        members.append(positions)
        matched += found
    _check_matched(index, matched)
    p_values = _romer_p_values(
        values,
        design,
        contrast,
        precision,
        members,
        statistic,
        n_rotations,
        shrink,
        generator,
    )
    return Table(
        {
            "gene_set": names,
            "n_genes": np.array([rows.size for rows in members], dtype=np.int64),
            "up_p_value": p_values[:, 0],
            "down_p_value": p_values[:, 1],
            "mixed_p_value": p_values[:, 2],
        }
    )


# ---------------------------------------------------------------------------------------
# roast and mroast
# ---------------------------------------------------------------------------------------


class _RoastTests(NamedTuple):
    """Per-set results of roast and mroast."""

    names: list[str]
    n_genes: npt.NDArray[np.int64]
    active: FloatArray  # (n_sets, 3): proportions active down, up and mixed
    p_values: FloatArray  # (n_sets, 4): down, up, two-sided and mixed
    n_rotations: int


def _roast_tests(
    y: Any,
    design: Any,
    contrast: Any,
    gene_sets: Any,
    *,
    gene_ids: Any,
    weights: Any,
    gene_weights: Any,
    set_statistic: Any,
    n_rotations: Any,
    trend: Any,
    robust: Any,
    rng: Any,
    n_threads: Any,
    function: str,
) -> _RoastTests:
    """The active proportions and p-values of each set, for roast and mroast."""
    statistic = _statistic(set_statistic, _ROAST_STATISTICS)
    n_rotations = _rotation_count(n_rotations)
    with_trend = checked_flag(trend, "trend", expected="a bool")
    with_robust = checked_flag(robust, "robust", expected="a bool")
    generator = _generator(rng)
    with thread_limit(n_threads):
        return _roast_sets(
            y,
            design,
            contrast,
            gene_sets,
            gene_ids,
            weights,
            gene_weights,
            statistic,
            n_rotations,
            with_trend,
            with_robust,
            generator,
            function,
        )


def _roast_sets(
    y: Any,
    design: Any,
    contrast: Any,
    gene_sets: Any,
    gene_ids: Any,
    weights: Any,
    gene_weights: Any,
    statistic: int,
    n_rotations: int,
    with_trend: bool,
    with_robust: bool,
    generator: np.random.Generator,
    function: str,
) -> _RoastTests:
    """:func:`_roast_tests` once its options are checked."""
    values, precision = _expression(y, weights)
    n_genes = values.shape[1]
    contribution = _gene_weights(gene_weights, n_genes)
    names, members, set_weights = _weighted_sets(
        gene_sets, gene_ids, n_genes, contribution, single=function == "roast"
    )
    reparametrised, d = _contrast_design(values, design, contrast, function)
    effects = _effects(values, reparametrised, precision)
    variances = np.sum(effects[:, 1:] ** 2, axis=1) / d
    squeezed = squeeze_variances(
        variances,
        d,
        covariate=_observed_means(values) if with_trend else None,
        robust=with_robust,
    )
    prior_variance = np.broadcast_to(squeezed.prior_variance, (n_genes,))
    prior_df = np.broadcast_to(squeezed.prior_df, (n_genes,))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = effects[:, 0] / np.sqrt(squeezed.posterior_variance)
    t = np.where(np.isnan(t), 0.0, t)  # a contrast effect of 0 over a variance of 0
    z = _bailey_z_scores(t, _capped_df(prior_df, d))

    # Only the genes of some set are rotated; members index this union.
    union, inverse = np.unique(
        np.concatenate([*members, np.empty(0, dtype=np.intp)]), return_inverse=True
    )
    sizes = np.array([rows.size for rows in members], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    union_members = inverse.astype(np.int64)
    flat_weights = np.concatenate([*set_weights, np.empty(0)]).astype(np.float64)
    arguments = (
        np.ascontiguousarray(effects[union]),
        np.sum(effects[union] ** 2, axis=1),
        np.ascontiguousarray(prior_variance[union], dtype=np.float64),
        np.ascontiguousarray(prior_df[union], dtype=np.float64),
        float(d),
        starts,
        union_members,
        flat_weights,
        statistic,
    )
    observed = _observed_set_statistics(z[union], *arguments[5:])
    counts = _counts(
        _roast_counts,
        _roast_counts_in_order,
        arguments,
        observed,
        generator,
        n_rotations,
        d + 1,
    )

    one_sided = (counts[:, :2] + 1.0) / (2.0 * n_rotations + 1.0)
    two_sided = (counts[:, 2:] + 1.0) / (n_rotations + 1.0)
    p_values = np.column_stack([one_sided, two_sided])
    active = np.full((len(members), 3), np.nan)
    for i, (rows, a) in enumerate(zip(members, set_weights, strict=True)):
        if rows.size:
            signed = np.sign(a) * z[rows]
            active[i] = [
                np.mean(signed < -_ACTIVE),
                np.mean(signed > _ACTIVE),
                np.mean(np.abs(z[rows]) > _ACTIVE),
            ]
    p_values[sizes == 0] = np.nan
    return _RoastTests(names, sizes, active, p_values, n_rotations)


def _observed_set_statistics(
    z: FloatArray,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    weights: FloatArray,
    statistic: int,
) -> FloatArray:
    """``(n_sets, 3)`` observed up, down and mixed statistics of each set."""
    observed: FloatArray = _set_statistics_of(
        np.ascontiguousarray(z), starts, members, weights, statistic
    )
    return observed


@jit
def _set_statistics_of(
    z: FloatArray,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    weights: FloatArray,
    statistic: int,
) -> FloatArray:
    n_sets = starts.size - 1
    out = np.zeros((n_sets, 3))
    scratch = np.empty(_largest_set(starts))
    for s in range(n_sets):
        if starts[s + 1] > starts[s]:
            up, down, mixed = _set_statistics(
                z, members, weights, starts[s], starts[s + 1], statistic, scratch
            )
            out[s, 0] = up
            out[s, 1] = down
            out[s, 2] = mixed
    return out


@jit
def _largest_set(starts: npt.NDArray[np.int64]) -> int:
    """The size of the largest set, at least 1, for a scratch array."""
    largest = 1
    for s in range(starts.size - 1):
        largest = max(largest, starts[s + 1] - starts[s])
    return largest


@jit
def _sum_of_largest(values: FloatArray, m: int, h: int) -> float:
    """The sum of the h largest of ``values[:m]``, which it reorders.

    Wirth's selection (Wirth 1976), Hoare's partition (Hoare 1961) kept to the side
    that holds the target, puts the (m - h)-th smallest value in its sorted place, with
    no larger value before it, in time proportional to m on average.
    """
    target = m - h
    low = 0
    high = m - 1
    while low < high:
        pivot = values[target]
        i = low
        j = high
        while i <= j:
            while values[i] < pivot:
                i += 1
            while pivot < values[j]:
                j -= 1
            if i <= j:
                values[i], values[j] = values[j], values[i]
                i += 1
                j -= 1
        if j < target:
            low = i
        if target < i:
            high = j
    total = 0.0
    for i in range(target, m):
        total += values[i]
    return total


@jit
def _set_statistics(
    z: FloatArray,
    members: npt.NDArray[np.int64],
    weights: FloatArray,
    start: int,
    stop: int,
    statistic: int,
    scratch: FloatArray,
) -> tuple[float, float, float]:
    """The up, down and mixed statistics of the genes ``members[start:stop]``.

    Larger values are more extreme for every alternative; the down statistic is the
    up statistic of -z.
    """
    m = stop - start
    if statistic == _MEAN50:
        h = (m + 1) // 2  # ceiling(m / 2)
        for i in range(m):
            scratch[i] = weights[start + i] * z[members[start + i]]
        up = _sum_of_largest(scratch, m, h)
        for i in range(m):
            scratch[i] = -weights[start + i] * z[members[start + i]]
        down = _sum_of_largest(scratch, m, h)
        for i in range(m):
            scratch[i] = abs(weights[start + i] * z[members[start + i]])
        mixed = _sum_of_largest(scratch, m, h)
        return up / h, down / h, mixed / h
    total = 0.0
    up = 0.0
    down = 0.0
    mixed = 0.0
    for i in range(start, stop):
        a = weights[i]
        value = z[members[i]]
        x = a * value
        total += abs(a)
        if statistic == _MEAN:
            up += x
            mixed += abs(x)
        elif statistic == _FLOORMEAN:
            if x > 0.0:
                up += x
            else:
                down -= x
            mixed += abs(a) * max(abs(value), _MIXED_FLOOR)
        else:  # _MSQ
            square = abs(a) * value * value
            if x > 0.0:
                up += square
            elif x < 0.0:
                down += square
            mixed += square
    if statistic == _MEAN:
        down = -up
    return up / total, down / total, mixed / total


def _counts(
    parallel: Any,
    serial: Any,
    arguments: tuple[Any, ...],
    observed: FloatArray,
    generator: np.random.Generator,
    n_rotations: int,
    dimension: int,
) -> npt.NDArray[np.int64]:
    """Counts of rotated statistics at least as extreme as the observed ones.

    The rotations are drawn ``_ROTATION_BATCH`` at a time, which bounds the memory
    they take; numpy's generator gives the same stream however its draws are split.
    """
    total: npt.NDArray[np.int64] | None = None
    for first in range(0, n_rotations, _ROTATION_BATCH):
        rotations = _rotation_vectors(
            generator, min(_ROTATION_BATCH, n_rotations - first), dimension
        )
        # The counts are sums of integers, so they do not depend on the chunks.
        counts = by_chunks(
            parallel, serial, (*arguments, observed, rotations), rotations.shape[0]
        )
        total = counts if total is None else total + counts
    assert total is not None  # n_rotations is at least 1
    return total


@jit(parallel=True)
def _roast_counts(
    effects: FloatArray,
    norms: FloatArray,
    prior_variance: FloatArray,
    prior_df: FloatArray,
    d: float,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    weights: FloatArray,
    statistic: int,
    observed: FloatArray,
    rotations: FloatArray,
    n_chunks: int,
) -> npt.NDArray[np.int64]:
    """The counts of :func:`_roast_chunk` over ``n_chunks`` runs of rotations, in
    parallel."""
    n_rotations = rotations.shape[0]
    partial = np.zeros((n_chunks, starts.size - 1, 4), dtype=np.int64)
    for chunk in prange(n_chunks):
        _roast_chunk(
            effects,
            norms,
            prior_variance,
            prior_df,
            d,
            starts,
            members,
            weights,
            statistic,
            observed,
            rotations,
            chunk * n_rotations // n_chunks,
            (chunk + 1) * n_rotations // n_chunks,
            partial[chunk],
        )
    counts: npt.NDArray[np.int64] = partial.sum(axis=0)
    return counts


@jit
def _roast_counts_in_order(
    effects: FloatArray,
    norms: FloatArray,
    prior_variance: FloatArray,
    prior_df: FloatArray,
    d: float,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    weights: FloatArray,
    statistic: int,
    observed: FloatArray,
    rotations: FloatArray,
) -> npt.NDArray[np.int64]:
    """The counts of :func:`_roast_counts` on the calling thread alone."""
    counts = np.zeros((starts.size - 1, 4), dtype=np.int64)
    _roast_chunk(
        effects,
        norms,
        prior_variance,
        prior_df,
        d,
        starts,
        members,
        weights,
        statistic,
        observed,
        rotations,
        0,
        rotations.shape[0],
        counts,
    )
    return counts


@jit
def _roast_chunk(
    effects: FloatArray,
    norms: FloatArray,
    prior_variance: FloatArray,
    prior_df: FloatArray,
    d: float,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    weights: FloatArray,
    statistic: int,
    observed: FloatArray,
    rotations: FloatArray,
    first: int,
    last: int,
    counts: npt.NDArray[np.int64],
) -> None:
    """Add to ``counts`` the rotated statistics of rotations ``first`` to ``last``.

    Args:
        effects: ``(n_genes, d + 1)`` effects of the genes of any set, the contrast
            effect first.
        norms: Each gene's squared length of its effects.
        prior_variance: Each gene's prior variance.
        prior_df: Each gene's prior df, possibly infinite.
        d: Residual degrees of freedom.
        starts: ``(n_sets + 1,)`` start of each set in ``members``.
        members: The genes (rows of ``effects``) of each set, one set after another.
        weights: The contribution weight of each entry of ``members``.
        statistic: The set statistic.
        observed: ``(n_sets, 3)`` observed up, down and mixed statistics.
        rotations: ``(n_rotations, d + 1)`` unit vectors.
        first: First rotation of the chunk.
        last: One past the last rotation of the chunk.
        counts: ``(n_sets, 4)`` counts of rotated statistics at least as extreme as
            the observed ones: down and up out of 2 per rotation (r and -r),
            two-sided and mixed out of 1.
    """
    n_genes, k = effects.shape
    n_sets = starts.size - 1
    z = np.empty(n_genes)
    scratch = np.empty(_largest_set(starts))
    for b in range(first, last):
        for g in range(n_genes):
            u = 0.0
            for j in range(k):
                u += rotations[b, j] * effects[g, j]
            t = _moderated_t(u, norms[g], prior_variance[g], prior_df[g], d)
            z[g] = _bailey_z(t, min(prior_df[g] + d, _DF_LIMIT))
        for s in range(n_sets):
            if starts[s + 1] == starts[s]:
                continue
            up, down, mixed = _set_statistics(
                z, members, weights, starts[s], starts[s + 1], statistic, scratch
            )
            up_observed = observed[s, 0]
            down_observed = observed[s, 1]
            # The rotation -r gives the up statistic of -z, which is down.
            counts[s, 0] += int(down >= down_observed) + int(up >= down_observed)
            counts[s, 1] += int(up >= up_observed) + int(down >= up_observed)
            counts[s, 2] += int(max(up, down) >= max(up_observed, down_observed))
            counts[s, 3] += int(mixed >= observed[s, 2])


@jit
def _moderated_t(
    u: float, norm: float, prior_variance: float, prior_df: float, d: float
) -> float:
    """The moderated t of a rotated contrast effect u of a gene whose effects have
    squared length ``norm``."""
    residual = max(norm - u * u, 0.0) / d
    if math.isinf(prior_df):
        posterior = prior_variance
    else:
        posterior = (prior_df * prior_variance + d * residual) / (prior_df + d)
    if posterior > 0.0:
        return u / math.sqrt(posterior)
    if u == 0.0:
        return 0.0
    return math.copysign(math.inf, u)


@jit
def _bailey_z(t: float, df: float) -> float:
    """Bailey's (1980) normalizing transformation of t on df degrees of freedom."""
    if math.isinf(df) or t == 0.0 or math.isnan(t):
        return t
    a = abs(t)
    shifted = df + 1.0 / 12.0
    if a < _LARGE_T:
        log_term = math.log1p(a * a / shifted)
    else:  # log(1 + e^l) with l = log(t^2 / (df + 1/12)), where t^2 would overflow
        log_ratio = 2.0 * math.log(a) - math.log(shifted)
        if log_ratio > 0.0:
            log_term = log_ratio + math.log1p(math.exp(-log_ratio))
        else:
            log_term = math.log1p(math.exp(log_ratio))
    ratio = 1.0 - 8.0 / (8.0 * df + 9.0)  # (8 df + 1) / (8 df + 9)
    return math.copysign(ratio * math.sqrt((df + 19.0 / 12.0) * log_term), t)


@jit
def _bailey_z_array(t: FloatArray, df: FloatArray) -> FloatArray:
    out = np.empty(t.size)
    for i in range(t.size):
        out[i] = _bailey_z(t[i], df[i])
    return out


def _bailey_z_scores(t: FloatArray, df: FloatArray) -> FloatArray:
    """Bailey's z-scores of moderated t-statistics, element by element."""
    t = np.ascontiguousarray(t, dtype=np.float64)
    df = np.ascontiguousarray(np.broadcast_to(df, t.shape), dtype=np.float64)
    result: FloatArray = _bailey_z_array(t, df)
    return result


def _capped_df(prior_df: FloatArray, d: int) -> FloatArray:
    """min(d_0 + d, 10^4): the degrees of freedom of roast's z-scores."""
    capped: FloatArray = np.minimum(
        np.asarray(prior_df, dtype=np.float64) + d, _DF_LIMIT
    )
    return capped


def _adjusted(p_values: FloatArray, shift: float) -> FloatArray:
    """Benjamini-Hochberg adjustment of p - shift, raised to at least p."""
    adjusted = adjust_p_values(np.maximum(p_values - shift, 0.0), "bh")
    result: FloatArray = np.maximum(adjusted, p_values)
    return result


# ---------------------------------------------------------------------------------------
# romer
# ---------------------------------------------------------------------------------------


def _romer_p_values(
    values: FloatArray,
    design: Any,
    contrast: Any,
    weights: FloatArray | None,
    members: list[IntArray],
    statistic: int,
    n_rotations: int,
    shrink: bool,
    generator: np.random.Generator,
) -> FloatArray:
    """``(n_sets, 3)`` up, down and mixed p-values of romer."""
    reparametrised, d = _contrast_design(values, design, contrast, "romer")
    effects = _effects(values, reparametrised, weights, signed_residuals=True)
    variances = np.sum(effects[:, 1:] ** 2, axis=1) / d
    squeezed = squeeze_variances(variances, d)
    n_genes = values.shape[1]
    prior_variance = np.broadcast_to(squeezed.prior_variance, (n_genes,))
    prior_df = np.broadcast_to(squeezed.prior_df, (n_genes,))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = effects[:, 0] / np.sqrt(squeezed.posterior_variance)
    t = np.where(np.isnan(t), 0.0, t)
    norms = np.sum(effects**2, axis=1)  # of the unshrunk effects, as in limma
    rotated = effects.copy()
    if shrink:
        rotated[:, 0] *= _shrinkage(
            reparametrised, weights, t, prior_df, d, prior_variance
        )
    sizes = np.array([rows.size for rows in members], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    flat = np.concatenate([*members, np.empty(0, dtype=np.intp)]).astype(np.int64)
    arguments = (
        np.ascontiguousarray(rotated),
        norms,
        np.ascontiguousarray(prior_variance, dtype=np.float64),
        np.ascontiguousarray(prior_df, dtype=np.float64),
        float(d),
        starts,
        flat,
        statistic,
    )
    observed = _observed_rank_statistics(
        np.ascontiguousarray(t), starts, flat, statistic
    )
    counts = _counts(
        _romer_counts,
        _romer_counts_in_order,
        arguments,
        observed,
        generator,
        n_rotations,
        d + 1,
    )
    p_values = np.asarray((counts + 1.0) / (n_rotations + 1.0), dtype=np.float64)
    p_values[sizes == 0] = np.nan
    return p_values


def _shrinkage(
    reparametrised: FloatArray,
    weights: FloatArray | None,
    t: FloatArray,
    prior_df: FloatArray,
    d: int,
    prior_variance: FloatArray,
) -> FloatArray:
    """Each gene's factor 1 / (1 + P v_0 / v)^(1/2) for its contrast effect.

    P and v_0 are those of the log-odds of ``moderated_statistics`` (Smyth 2004), with
    the t-statistics on d_0 + d degrees of freedom and the proportion of
    differentially expressed genes 1 - pi_0.
    """
    df = np.asarray(prior_df, dtype=np.float64) + d
    p_values = 2.0 * np.asarray(t_sf(np.abs(t), df), dtype=np.float64)
    null_proportion = _proportion_true_null(p_values)
    if null_proportion >= 1.0:
        return np.ones(t.size)
    # limma passes 1 - pi_0 as the proportion of differentially expressed genes; when
    # every p-value is zero it is 1, which the log-odds cannot take, and the
    # largest proportion below 1 stands for it.
    proportion = min(1.0 - null_proportion, 1.0 - 2.0**-52)
    whitened = reparametrised
    if weights is not None:
        whitened = reparametrised * np.sqrt(weights)[:, np.newaxis]
    v = float(np.linalg.inv(whitened.T @ whitened)[-1, -1])
    unscaled = np.full((t.size, 1), math.sqrt(v))
    v0 = _coefficient_prior_variance(
        t,
        unscaled[:, 0],
        df,
        proportion,
        _COEFFICIENT_LIMITS,
        float(np.nanmedian(prior_variance)),
    )
    lods = _log_odds(
        t[:, np.newaxis], unscaled, df, np.isinf(df), np.array([v0]), proportion
    )[:, 0]
    with np.errstate(over="ignore"):
        posterior = 1.0 / (1.0 + np.exp(-lods))
    with np.errstate(invalid="ignore"):
        factors = 1.0 / np.sqrt(1.0 + posterior * v0 / v)
    # An infinite t-statistic (no residual or prior variance) has no log-odds; its
    # contrast effect is not shrunk.
    result: FloatArray = np.where(np.isfinite(factors), factors, 1.0)
    return result


def _proportion_true_null(p_values: FloatArray) -> float:
    """(2 / (n + 1)) sum min(p_(i), i / n): limma's ``propTrueNull(method = "lfdr")``,
    found by probing."""
    ordered = np.sort(p_values)
    n = ordered.size
    bounded = np.minimum(ordered, np.arange(1, n + 1) / n)
    return 2.0 / (n + 1) * math.fsum(bounded.tolist())


def _observed_rank_statistics(
    t: FloatArray,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    statistic: int,
) -> FloatArray:
    """``(n_sets, 3)`` observed up, down and mixed rank sums of each set."""
    result: FloatArray = _rank_statistics_of(t, starts, members, statistic)
    return result


@jit
def _rank_statistics_of(
    t: FloatArray,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    statistic: int,
) -> FloatArray:
    n_genes = t.size
    up = np.empty(n_genes)
    down = np.empty(n_genes)
    mixed = np.empty(n_genes)
    _alternative_ranks(t, statistic, up, down, mixed)
    n_sets = starts.size - 1
    out = np.zeros((n_sets, 3))
    scratch = np.empty(_largest_set(starts))
    for s in range(n_sets):
        if starts[s + 1] > starts[s]:
            out[s, 0] = _rank_sum(
                up, members, starts[s], starts[s + 1], statistic, scratch
            )
            out[s, 1] = _rank_sum(
                down, members, starts[s], starts[s + 1], statistic, scratch
            )
            out[s, 2] = _rank_sum(
                mixed, members, starts[s], starts[s + 1], statistic, scratch
            )
    return out


@jit
def _alternative_ranks(
    t: FloatArray,
    statistic: int,
    up: FloatArray,
    down: FloatArray,
    mixed: FloatArray,
) -> None:
    """Fill each gene's rank for the up, down and mixed alternatives.

    Ranks run from 1 (smallest) to the number of genes, and ties get their average
    rank. The up ranks are those of t, the down ranks those of -t and the mixed ranks
    those of |t|; for ``"floormean"`` negative values count as zero and, for the mixed
    alternative, |t| below 1 as 1.
    """
    n = t.size
    order = np.argsort(t)
    _average_ranks(t, order, up)
    magnitude = np.abs(t)
    _average_ranks(magnitude, _magnitude_order(t, order), mixed)
    for g in range(n):
        down[g] = n + 1.0 - up[g]  # the rank of -t
    if statistic != _FLOORMEAN:
        return
    not_positive = 0
    not_negative = 0
    small = 0
    for g in range(n):
        not_positive += t[g] <= 0.0
        not_negative += t[g] >= 0.0
        small += magnitude[g] <= _ROMER_MIXED_FLOOR
    for g in range(n):
        if t[g] <= 0.0:
            up[g] = 0.5 * (not_positive + 1.0)
        if t[g] >= 0.0:
            down[g] = 0.5 * (not_negative + 1.0)
        if magnitude[g] <= _ROMER_MIXED_FLOOR:
            mixed[g] = 0.5 * (small + 1.0)


@jit
def _magnitude_order(
    t: FloatArray, order: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """The order of |t| from the order of t: its negative values, the largest first,
    merged with the others, which saves sorting again."""
    n = t.size
    first = 0  # the first value that is not negative, in the order of t
    while first < n and t[order[first]] < 0.0:
        first += 1
    merged = np.empty(n, dtype=np.int64)
    i = first - 1
    j = first
    for k in range(n):
        if j == n or (i >= 0 and -t[order[i]] <= t[order[j]]):
            merged[k] = order[i]
            i -= 1
        else:
            merged[k] = order[j]
            j += 1
    return merged


@jit
def _average_ranks(
    values: FloatArray, order: npt.NDArray[np.int64], ranks: FloatArray
) -> None:
    """Fill ``ranks`` with the average ranks of ``values``, sorted by ``order``."""
    n = values.size
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + 1 + j)  # the mean of ranks i + 1 to j
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j


@jit
def _rank_sum(
    ranks: FloatArray,
    members: npt.NDArray[np.int64],
    start: int,
    stop: int,
    statistic: int,
    scratch: FloatArray,
) -> float:
    """The sum of the set's ranks, or of its ceiling(m / 2) largest for mean50.

    Sums of ranks, which are multiples of 1/2, are exact, so equal means compare
    equal.
    """
    m = stop - start
    if statistic == _MEAN50:
        for i in range(m):
            scratch[i] = ranks[members[start + i]]
        largest: float = _sum_of_largest(scratch, m, (m + 1) // 2)
        return largest
    total = 0.0
    for i in range(start, stop):
        total += ranks[members[i]]
    return total


@jit(parallel=True)
def _romer_counts(
    effects: FloatArray,
    norms: FloatArray,
    prior_variance: FloatArray,
    prior_df: FloatArray,
    d: float,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    statistic: int,
    observed: FloatArray,
    rotations: FloatArray,
    n_chunks: int,
) -> npt.NDArray[np.int64]:
    """The counts of :func:`_romer_chunk` over ``n_chunks`` runs of rotations, in
    parallel."""
    n_rotations = rotations.shape[0]
    partial = np.zeros((n_chunks, starts.size - 1, 3), dtype=np.int64)
    for chunk in prange(n_chunks):
        _romer_chunk(
            effects,
            norms,
            prior_variance,
            prior_df,
            d,
            starts,
            members,
            statistic,
            observed,
            rotations,
            chunk * n_rotations // n_chunks,
            (chunk + 1) * n_rotations // n_chunks,
            partial[chunk],
        )
    counts: npt.NDArray[np.int64] = partial.sum(axis=0)
    return counts


@jit
def _romer_counts_in_order(
    effects: FloatArray,
    norms: FloatArray,
    prior_variance: FloatArray,
    prior_df: FloatArray,
    d: float,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    statistic: int,
    observed: FloatArray,
    rotations: FloatArray,
) -> npt.NDArray[np.int64]:
    """The counts of :func:`_romer_counts` on the calling thread alone."""
    counts = np.zeros((starts.size - 1, 3), dtype=np.int64)
    _romer_chunk(
        effects,
        norms,
        prior_variance,
        prior_df,
        d,
        starts,
        members,
        statistic,
        observed,
        rotations,
        0,
        rotations.shape[0],
        counts,
    )
    return counts


@jit
def _romer_chunk(
    effects: FloatArray,
    norms: FloatArray,
    prior_variance: FloatArray,
    prior_df: FloatArray,
    d: float,
    starts: npt.NDArray[np.int64],
    members: npt.NDArray[np.int64],
    statistic: int,
    observed: FloatArray,
    rotations: FloatArray,
    first: int,
    last: int,
    counts: npt.NDArray[np.int64],
) -> None:
    """Add to ``counts`` the rotated rank statistics of rotations ``first`` to
    ``last``: how many are at least the observed up, down and mixed statistics.

    ``effects`` are every gene's effects with the contrast effect shrunk, and
    ``norms`` the squared lengths of the unshrunk effects.
    """
    n_genes, k = effects.shape
    n_sets = starts.size - 1
    t = np.empty(n_genes)
    up = np.empty(n_genes)
    down = np.empty(n_genes)
    mixed = np.empty(n_genes)
    scratch = np.empty(_largest_set(starts))
    for b in range(first, last):
        for g in range(n_genes):
            u = 0.0
            for j in range(k):
                u += rotations[b, j] * effects[g, j]
            t[g] = _moderated_t(u, norms[g], prior_variance[g], prior_df[g], d)
        _alternative_ranks(t, statistic, up, down, mixed)
        for s in range(n_sets):
            start = starts[s]
            stop = starts[s + 1]
            if stop == start:
                continue
            for a, ranks in enumerate((up, down, mixed)):
                total = _rank_sum(ranks, members, start, stop, statistic, scratch)
                counts[s, a] += int(total >= observed[s, a])


# ---------------------------------------------------------------------------------------
# Rotations, gene sets and arguments
# ---------------------------------------------------------------------------------------


def _rotation_vectors(
    generator: np.random.Generator, n_rotations: int, dimension: int
) -> FloatArray:
    """``(n_rotations, dimension)`` points uniform on the unit sphere."""
    draws = np.asarray(
        generator.standard_normal((n_rotations, dimension)), dtype=np.float64
    )
    lengths = np.sqrt(np.sum(draws * draws, axis=1))
    rotations: FloatArray = np.ascontiguousarray(draws / lengths[:, np.newaxis])
    return rotations


def _weighted_sets(
    gene_sets: Any,
    gene_ids: Any,
    n_genes: int,
    gene_weights: FloatArray | None,
    *,
    single: bool,
) -> tuple[list[str], list[IntArray], list[FloatArray]]:
    """Each set's name, the sorted positions of its genes and their contribution
    weights, with genes of weight zero left out."""
    index = _checked_gene_sets(gene_sets, gene_ids, n_genes)
    names: list[str] = []
    members: list[IntArray] = []
    weights: list[FloatArray] = []
    matched = 0
    for name, genes in gene_sets.items():
        label = "gene_set" if single else _set_label(name)
        if isinstance(genes, Mapping):
            if gene_weights is not None:
                raise ValueError(
                    f"{label} gives its genes' contribution weights, so gene_weights "
                    "cannot be given too; give the weights of every set one way"
                )
            positions, set_weights, found = _mapped_set(genes, label, index, n_genes)
        else:
            positions, found = _set_positions(genes, label, index, n_genes)
            set_weights = (
                np.ones(positions.size)
                if gene_weights is None
                else gene_weights[positions]
            )
        keep = set_weights != 0.0
        names.append(name)
        members.append(positions[keep])
        weights.append(set_weights[keep])
        matched += found
    if index is not None and matched == 0 and single:
        raise ValueError(
            "no gene of gene_set is among gene_ids; the set and gene_ids must name "
            "genes the same way (for example both by symbol, or both by Ensembl id)"
        )
    _check_matched(index, matched)
    return names, members, weights


def _mapped_set(
    genes: Mapping[Any, Any], label: str, index: dict[Any, int] | None, n_genes: int
) -> tuple[IntArray, FloatArray, int]:
    """The positions and weights of a set given as a mapping of gene to weight."""
    reject_masked(genes, label)
    keys = list(genes.keys())
    values = []
    for key in keys:
        weight = genes[key]
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, numbers.Real):
            raise TypeError(
                f"{label} maps genes to contribution weights, which must be real "
                f"numbers; got {type(weight).__name__} for {key!r}"
            )
        if not math.isfinite(float(weight)):
            raise ValueError(
                f"{label} gives {key!r} the weight {float(weight)}; contribution "
                "weights must be finite"
            )
        values.append(float(weight))
    weights = np.array(values, dtype=np.float64)
    found = 0
    if index is None:
        positions = _positions(unmasked_array(keys, label), label, n_genes)
    else:
        present = [key in index for key in _hashable_keys(keys, label)]
        positions = _id_positions(keys, label, index)
        weights = weights[np.array(present, dtype=bool)]
        found = positions.size
    order = np.argsort(positions, kind="stable")
    return positions[order].astype(np.intp), weights[order], found


def _hashable_keys(keys: list[Any], label: str) -> list[Any]:
    try:
        for key in keys:
            hash(key)
    except TypeError:
        raise TypeError(
            f"{label} holds values that cannot be gene ids, such as {key!r}"
        ) from None
    return keys


def _gene_weights(gene_weights: Any, n_genes: int) -> FloatArray | None:
    if gene_weights is None:
        return None
    values = unmasked_array(gene_weights, "gene_weights")
    if values.dtype.kind not in "iuf":
        raise TypeError(
            f"gene_weights must hold real numbers, got dtype {values.dtype}"
        )
    if values.shape != (n_genes,):
        raise ValueError(
            f"gene_weights must have one weight per gene ({n_genes}), got shape "
            f"{values.shape}"
        )
    weights = values.astype(np.float64)
    if not np.all(np.isfinite(weights)):
        raise ValueError("gene_weights must be finite")
    return weights


def _statistic(value: Any, options: tuple[str, ...]) -> int:
    if not isinstance(value, str) or value not in options:
        raise ValueError(
            f"set_statistic must be one of {', '.join(map(repr, options))}, got "
            f"{value!r}"
        )
    return _ROAST_STATISTICS.index(value)


def _rotation_count(value: Any) -> int:
    reject_masked(value, "n_rotations")
    return positive_integer(value, "n_rotations", expected="a positive integer")


def _generator(rng: Any) -> np.random.Generator:
    reject_masked(rng, "rng")
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (bool, np.bool_)) or not isinstance(rng, numbers.Integral):
        raise TypeError(
            "rng must be a numpy.random.Generator, an integer seed or None, got "
            f"{type(rng).__name__}"
        )
    seed = operator.index(rng)
    if seed < 0:
        raise ValueError(f"rng must be a non-negative seed, got {seed}")
    return np.random.default_rng(seed)
