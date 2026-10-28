"""Combined p-values of several studies, and partial conjunction tests of replication.

:func:`combine_p_values` combines, for every feature (gene), the p-values of the
studies that tested it into one p-value of the global null hypothesis that no study
has an effect, by the methods of Fisher (1932), Stouffer et al. (1949), Tippett
(1931), Wilkinson (1951), Pearson (1933), Mudholkar and George (George and Mudholkar
1983) and Simes (1986). :func:`partial_conjunction` gives the p-value that at least u of
the n studies have an effect (Benjamini and Heller 2008). Both return a
:class:`CombinedPValues`.
https://docs.nygen.io/CytePool/statistics/replication#combined-p-values explains the
methods and when to choose each, and
https://docs.nygen.io/CytePool/statistics/replication#partial-conjunction the partial
conjunctions.

One-sided and two-sided p-values:
    The p-values are one-sided, for one direction of effect, unless ``directions``
    gives the signs of the studies' effects (such as their log2 fold changes) for
    two-sided ones. These become one-sided ones, as metap's ``two2one`` makes them:
    for the effects above zero, p / 2 where a study's effect is above zero, 1 - p / 2
    where it is below and 1/2 where it is exactly zero; for the effects below zero,
    their complements. Each direction is combined, into ``up_p_value`` and
    ``down_p_value``, and the p-value is min(1, 2 min(p_up, p_down)), the Bonferroni
    bound over the two directions (Owen 2009); for Stouffer's method it equals
    2 (1 - Phi(|Z|)), in which effects in opposite directions cancel. The one-sided
    p-value near 0 is kept exactly: Stouffer's z_i is the normal quantile of p / 2 with
    the sign of the effect, and each method reads log(1 - p_i) from the complement
    p / 2 where p_i is 1 - p / 2.

Studies that did not test a feature:
    NaN marks them. Each feature is combined over the k studies that tested it,
    which ``n_studies`` gives; a feature that no study tested has NaN. A single
    study's p-value is its own combined p-value, by every method but the logit,
    whose t distribution approximates the distribution of a sum of logistic
    variables, also for one.

p-values of exactly 0 and 1:
    They are limits that every method reaches: a p-value of 0 makes Fisher's,
    Tippett's, Simes's, Stouffer's and the logit p-values 0, and a p-value of 1
    makes Pearson's 1, and Stouffer's and the logit 1 unless another study's p-value
    is 0, where the limit does not exist and the combined p-value is NaN.

Matching scipy and metap:
    The combined p-values match scipy 1.18.1's ``combine_pvalues`` and metap 1.1, and
    the partial conjunctions their definition computed with metap; running metap on
    chosen inputs settled what its documentation leaves open.

Differences from scipy and metap:
    https://docs.nygen.io/CytePool/statistics/replication#replication-limits gives
    those from metap. scipy's ``combine_pvalues`` gives NaN for a feature with a
    missing p-value unless asked to omit it (``nan_policy="omit"``), as this module
    always does.

References:
    Fisher RA (1932). Statistical Methods for Research Workers, 4th edition. Oliver
    and Boyd, Edinburgh.

    Stouffer SA, Suchman EA, DeVinney LC, Star SA, Williams RM (1949). The American
    Soldier, volume 1: Adjustment during Army Life. Princeton University Press,
    Princeton.

    Lipták T (1958). On the combination of independent tests. Magyar Tudományos
    Akadémia Matematikai Kutató Intézetének Közleményei 3, 171-197.

    Tippett LHC (1931). The Methods of Statistics. Williams and Norgate, London.

    Wilkinson B (1951). A statistical consideration in psychological research.
    Psychological Bulletin 48, 156-158. doi:10.1037/h0059111

    Pearson K (1933). On a method of determining whether a sample of size n supposed
    to have been drawn from a parent population having a known probability integral
    has probably been drawn at random. Biometrika 25, 379-410.
    doi:10.1093/biomet/25.3-4.379

    George EO, Mudholkar GS (1983). On the convolution of logistic random variables.
    Metrika 30, 1-13. doi:10.1007/BF02056895

    Simes RJ (1986). An improved Bonferroni procedure for multiple tests of
    significance. Biometrika 73, 751-754. doi:10.1093/biomet/73.3.751

    Zaykin DV (2011). Optimally weighted Z-test is a powerful method for combining
    probabilities in meta-analysis. Journal of Evolutionary Biology 24, 1836-1841.
    doi:10.1111/j.1420-9101.2011.02297.x

    Owen AB (2009). Karl Pearson's meta-analysis revisited. Annals of Statistics 37,
    3867-3892. doi:10.1214/09-AOS697

    Benjamini Y, Heller R (2008). Screening for partial conjunction hypotheses.
    Biometrics 64, 1215-1222. doi:10.1111/j.1541-0420.2007.00984.x
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._masked_arrays import read_only, unmasked_array
from cytepool._options import checked_choice, integral_number
from cytepool.stats.distributions import (
    beta_cdf,
    chi2_cdf,
    chi2_sf,
    norm_isf,
    norm_sf,
    t_sf,
)

__all__ = ["CombinedPValues", "combine_p_values", "partial_conjunction"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_METHODS = (
    "fisher",
    "stouffer",
    "tippett",
    "wilkinson",
    "pearson",
    "mudholkar_george",
    "simes",
)
# The combinations of the largest p-values that Benjamini and Heller (2008) give.
_CONJUNCTION_METHODS = ("simes", "fisher", "stouffer")


@dataclass(frozen=True, eq=False, repr=False)
class CombinedPValues:
    """Combined p-values of many features, from :func:`combine_p_values` or
    :func:`partial_conjunction`.

    The per-feature arrays have the shape of the p-values without their first (study)
    axis: one value per feature, or 0-d arrays for the p-values of one feature given
    as a vector. They are read-only, and NaN where fewer than ``at_least`` studies
    tested a feature.

    Attributes:
        p_value: The combined p-value: with ``directions``, min(1, 2 min(p_up,
            p_down)), two-sided.
        statistic: The method's statistic: Fisher's and Pearson's X, Stouffer's Z, the
            smallest or the r-th smallest p-value (Tippett, Wilkinson), the logit
            method's t, or the Simes p-value; of the n - u + 1 largest p-values for a
            partial conjunction; with ``directions``, that of the effects above zero,
            so that Stouffer's Z is positive for effects mostly up.
        up_p_value: With ``directions``, the combined one-sided p-value for effects
            above zero; None without.
        down_p_value: Likewise, for effects below zero.
        sign: With ``directions``, the sign of the combined effect: 1 where
            ``up_p_value`` is the smaller, -1 where ``down_p_value`` is, and 0 where
            they are equal; None without.
        n_studies: The number of studies that tested each feature (whose p-value is
            not NaN).
        method: The method, such as ``"fisher"``.
        at_least: u, the fewest studies with an effect that the alternative
            hypothesis claims: 1 for :func:`combine_p_values`.
        rank: r of Wilkinson's method; None for the others.
    """

    p_value: FloatArray
    statistic: FloatArray
    up_p_value: FloatArray | None
    down_p_value: FloatArray | None
    sign: FloatArray | None
    n_studies: IntArray
    method: str
    at_least: int
    rank: int | None

    def __repr__(self) -> str:
        studies = self.n_studies[self.n_studies > 0]
        span = f"{studies.min()} to {studies.max()}" if studies.size else "none"
        signed = ", with directions" if self.sign is not None else ""
        return (
            f"CombinedPValues(n_features={self.p_value.size}, method={self.method!r}, "
            f"at_least={self.at_least}{signed}, studies per feature: {span})"
        )


def combine_p_values(
    p_values: npt.ArrayLike,
    *,
    method: str = "fisher",
    weights: npt.ArrayLike | None = None,
    directions: npt.ArrayLike | None = None,
    rank: int | None = None,
) -> CombinedPValues:
    """Combine the p-values of several studies, feature by feature.

    Every feature's p-values, over the studies that tested it, are combined into one
    p-value of the hypothesis that no study has an effect, by the methods that
    scipy's ``combine_pvalues`` and metap offer. With ``directions``, two-sided
    p-values become one-sided ones for each direction, each direction is combined,
    and the p-value is min(1, 2 min(p_up, p_down)). NaN marks a study that did not
    test a feature, and p-values of 0 and 1 are combined as the limits they are.
    https://docs.nygen.io/CytePool/statistics/replication#combined-p-values explains
    when to choose each method.

    The methods:
        For the p-values p_1, ..., p_k of the k studies that tested a feature, in order
        p_(1) <= ... <= p_(k):

        - ``"fisher"``: X = -2 sum log p_i against the chi-squared distribution with 2k
          degrees of freedom (metap's ``sumlog``).
        - ``"stouffer"``: Z = sum w_i z_i / sqrt(sum w_i^2), with z_i = Phi^-1(1 - p_i),
          against the standard normal distribution (Stouffer et al. 1949; Lipták 1958;
          metap's ``sumz``). The weights w_i are equal by default; the square roots of
          the studies' sample sizes are the usual choice (Zaykin 2011).
        - ``"tippett"``: the smallest p-value, with P(p_(1) <= s) = 1 - (1 - s)^k
          (metap's ``minimump``).
        - ``"wilkinson"``: the r-th smallest p-value, against the beta distribution with
          parameters r and k - r + 1 (metap's ``wilkinsonp``; r = 1 is Tippett's method
          and r = k metap's ``maximump``, p_(k)^k). It tests the global null hypothesis,
          also with r = k: that every study has an effect is the alternative of
          :func:`partial_conjunction` with ``at_least`` = k.
        - ``"pearson"``: X = -2 sum log(1 - p_i), whose chi-squared distribution function
          with 2k degrees of freedom is the p-value: small when every p-value is small,
          and near 1 when any p-value is near 1 (scipy's ``"pearson"``, whose statistic
          is -X).
        - ``"mudholkar_george"``: the logit statistic T = sum log((1 - p_i) / p_i), times
          sqrt(3 / k) / pi sqrt((5k + 4) / (5k + 2)), against Student's t distribution
          with 5k + 4 degrees of freedom (metap's ``logitp``, whose statistic this is;
          scipy reports T).
        - ``"simes"``: min over j of k p_(j) / j, a p-value of the global null hypothesis
          that is valid for independent and positively dependent p-values (Simes 1986).

        The statistics are monotone in each p-value, and each p-value is exact for
        independent studies whose p-values are uniform under the null hypothesis.

    With ``directions``, a two-sided p-value below the smallest normal double
    (2.2e-308) loses digits when halved, and one of 5e-324 halves to 0.

    Args:
        p_values: ``(n_studies, n_features)`` p-values in [0, 1], or ``(n_studies,)``
            for one feature; NaN where a study did not test a feature. One-sided
            p-values for the same direction of effect, or two-sided p-values, which
            ``directions`` turns into one-sided ones.
        method: ``"fisher"``, ``"stouffer"``, ``"tippett"``, ``"wilkinson"``,
            ``"pearson"``, ``"mudholkar_george"`` or ``"simes"``.
        weights: For Stouffer's method only: the weight of every study,
            ``(n_studies,)``, or of every study and feature, of the shape of
            ``p_values``; positive and finite where a p-value is given, such as the
            square roots of the studies' sample sizes. Equal by default.
        directions: The signs of the studies' effects, such as their log2 fold
            changes, of the shape of ``p_values``, for two-sided p-values: each
            direction is combined from one-sided p-values, and the p-value is
            two-sided. Not NaN where a p-value is given; zero for an effect of exactly
            zero.
        rank: r, for Wilkinson's method only: the r-th smallest p-value is
            combined, at least 1. A feature that fewer than r studies tested has NaN.

    Returns:
        The combined p-value and statistic of every feature, with the number of
        studies that tested it, and with ``directions`` the one-sided p-values and
        the direction of the combined effect.

    Raises:
        TypeError: If an argument does not hold real numbers, is a numpy masked
            array, or an option has the wrong type.
        ValueError: If a p-value lies outside [0, 1], the shapes do not match,
            there is no study, ``method`` is unknown, ``weights`` are given for a
            method other than Stouffer's or are not positive and finite where a
            p-value is given, a direction is NaN where a p-value is given, or
            ``rank`` is missing for Wilkinson's method, given for another, or less
            than 1.

    Example:
        >>> result = combine_p_values([0.01, 0.2, 0.5])
        >>> round(float(result.statistic), 4), round(float(result.p_value), 6)
        (13.8155, 0.031766)

        Two studies (rows) of two genes (columns), with two-sided p-values: the
        effects of the first gene agree, those of the second do not.

        >>> two_sided = [[0.01, 0.01], [0.04, 0.04]]
        >>> signed = combine_p_values(
        ...     two_sided, method="stouffer", directions=[[1.0, 1.0], [1.0, -1.0]]
        ... )
        >>> signed.sign.tolist(), signed.p_value.round(4).tolist()
        ([1.0, 1.0], [0.0011, 0.712])
    """
    chosen = checked_choice(method, "method", _METHODS)
    p = _p_values(p_values)
    order = _wilkinson_rank(rank, chosen)
    weight = _weights(weights, p, chosen)
    signs = _directions(directions, p)
    return _combination(p, chosen, 1, weight, signs, order)


def partial_conjunction(
    p_values: npt.ArrayLike,
    at_least: int,
    *,
    method: str = "simes",
    directions: npt.ArrayLike | None = None,
) -> CombinedPValues:
    """The p-value that at least u of the studies have an effect, feature by feature.

    Benjamini and Heller's (2008) partial conjunction test: the hypothesis that fewer
    than u = ``at_least`` of the n studies that tested a feature have an effect is
    tested by combining its n - u + 1 largest p-values by Simes's, Fisher's or
    Stouffer's method. With ``at_least=2`` a small p-value says that the effect is
    replicated: one study's effect alone does not make it small. Simes's p-value is
    the minimum over i = u, ..., n of (n - u + 1) p_(i) / (i - u + 1), for the
    p-values in order p_(1) <= ... <= p_(n); Fisher's and Stouffer's (without
    weights, as the paper defines it) combine the n - u + 1 largest p-values. For
    u = 1 it is the global null hypothesis, and for u = n every method gives p_(n).
    With ``directions``, the effects must share their direction: the one-sided
    p-values of each direction are combined, and the p-value is min(1, 2 min(p_up,
    p_down)). Adjust the p-values over the features with
    ``adjust_p_values(result.p_value, "bh")`` to control the false discovery rate of
    the replication claims, for independent features (Benjamini and Heller 2008).
    https://docs.nygen.io/CytePool/statistics/replication#partial-conjunction explains
    the test.

    Args:
        p_values: ``(n_studies, n_features)`` p-values in [0, 1], or ``(n_studies,)``
            for one feature; NaN where a study did not test a feature.
        at_least: u, the fewest studies with an effect, at least 1. A feature that
            fewer studies tested has NaN.
        method: ``"simes"``, ``"fisher"`` or ``"stouffer"``.
        directions: The signs of the studies' effects, for two-sided p-values, as for
            :func:`combine_p_values`.

    Returns:
        The partial conjunction p-value of every feature, with the number of studies
        that tested it, and with ``directions`` the one-sided p-values and the
        direction of the effect.

    Raises:
        TypeError: If an argument does not hold real numbers, is a numpy masked
            array, or an option has the wrong type.
        ValueError: If a p-value lies outside [0, 1], the shapes do not match,
            there is no study, ``method`` is unknown, ``at_least`` is less than 1,
            or a direction is NaN where a p-value is given.

    Example:
        >>> found_once = [[1e-8], [0.6], [0.9]]
        >>> partial_conjunction(found_once, 1).p_value.round(12).tolist()
        [3e-08]
        >>> partial_conjunction(found_once, 2).p_value.tolist()
        [0.9]
    """
    chosen = checked_choice(method, "method", _CONJUNCTION_METHODS)
    u = integral_number(at_least, "at_least", expected="a whole number of studies")
    if u < 1:
        raise ValueError(f"at_least must be at least 1, got {u}")
    p = _p_values(p_values)
    signs = _directions(directions, p)
    return _combination(p, chosen, u, None, signs, None)


# Checks of the arguments.


def _p_values(values: Any) -> FloatArray:
    array = unmasked_array(values, "p_values", missing="NaN")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"p_values must hold real numbers, got dtype {array.dtype}")
    p: FloatArray = np.array(array, dtype=np.float64)
    if p.ndim not in (1, 2):
        raise ValueError(
            "p_values must be (n_studies, n_features), or (n_studies,) for one "
            f"feature, got {p.ndim} dimensions"
        )
    if p.shape[0] == 0:
        raise ValueError("there are no studies: p_values has no rows")
    outside = ~np.isnan(p) & ~((p >= 0.0) & (p <= 1.0))
    if outside.any():
        position = tuple(int(i) for i in np.argwhere(outside)[0])
        raise ValueError(
            "p_values must lie in [0, 1], or be NaN where a study did not test a "
            f"feature; found {np.count_nonzero(outside)} outside, such as "
            f"{p[position]} at {position}"
        )
    return p


def _wilkinson_rank(rank: Any, method: str) -> int | None:
    if method != "wilkinson":
        if rank is not None:
            raise ValueError(
                f"rank is Wilkinson's r, for method='wilkinson'; method is {method!r}"
            )
        return None
    if rank is None:
        raise ValueError(
            "method='wilkinson' needs rank, r: the r-th smallest p-value is combined"
        )
    r = integral_number(rank, "rank", expected="a whole number")
    if r < 1:
        raise ValueError(f"rank must be at least 1, got {r}")
    return r


def _weights(weights: Any, p: FloatArray, method: str) -> FloatArray | None:
    if weights is None:
        return None
    if method != "stouffer":
        raise ValueError(
            f"weights apply to Stouffer's method only; method is {method!r}"
        )
    array = unmasked_array(weights, "weights", missing="NaN")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"weights must hold real numbers, got dtype {array.dtype}")
    w = np.array(array, dtype=np.float64)
    if w.shape == (p.shape[0],) and p.ndim == 2:
        w = np.broadcast_to(w[:, np.newaxis], p.shape)
    if w.shape != p.shape:
        raise ValueError(
            f"weights must be (n_studies,) = ({p.shape[0]},) or of the shape of "
            f"p_values, {p.shape}; got {w.shape}"
        )
    given = ~np.isnan(p)
    invalid = given & ~((w > 0.0) & np.isfinite(w))
    if invalid.any():
        position = tuple(int(i) for i in np.argwhere(invalid)[0])
        raise ValueError(
            "weights must be positive and finite where a p-value is given; found "
            f"{w[position]} at {position}"
        )
    return np.where(given, w, np.nan)


def _directions(directions: Any, p: FloatArray) -> FloatArray | None:
    if directions is None:
        return None
    array = unmasked_array(directions, "directions", missing="NaN")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"directions must hold real numbers, got dtype {array.dtype}")
    d = np.array(array, dtype=np.float64)
    if d.shape != p.shape:
        raise ValueError(
            f"directions must have the shape of p_values, {p.shape}; got {d.shape}"
        )
    missing = ~np.isnan(p) & np.isnan(d)
    if missing.any():
        position = tuple(int(i) for i in np.argwhere(missing)[0])
        raise ValueError(
            "directions must be given wherever a p-value is: the sign of the "
            f"study's effect; found NaN at {position}"
        )
    return np.sign(d)


# The combinations.


def _combination(
    p: FloatArray,
    method: str,
    at_least: int,
    weights: FloatArray | None,
    signs: FloatArray | None,
    rank: int | None,
) -> CombinedPValues:
    shape = p.shape[1:]
    n_studies = p.shape[0]
    q = p.reshape(n_studies, -1)
    tested = np.count_nonzero(~np.isnan(q), axis=0).astype(np.int64)
    w = None if weights is None else weights.reshape(n_studies, -1)
    if signs is None:
        statistic, p_value = _combined(q, 1.0 - q, method, at_least, w, rank, tested)
        up = down = direction = None
    else:
        s = signs.reshape(n_studies, -1)
        # The one-sided p-values for effects above zero and their complements, the
        # one-sided p-values for effects below; the one near 0 is p / 2 exactly.
        half = 0.5 * q
        upper = np.where(s > 0.0, half, np.where(s < 0.0, 1.0 - half, 0.5))
        lower = np.where(s > 0.0, 1.0 - half, np.where(s < 0.0, half, 0.5))
        upper[np.isnan(q)] = np.nan
        lower[np.isnan(q)] = np.nan
        statistic, up = _combined(upper, lower, method, at_least, w, rank, tested)
        _, down = _combined(lower, upper, method, at_least, w, rank, tested)
        with np.errstate(invalid="ignore"):
            p_value = np.minimum(1.0, 2.0 * np.minimum(up, down))
            direction = np.sign(down - up)

    def shaped(values: FloatArray) -> FloatArray:
        view: FloatArray = read_only(values.reshape(shape))
        return view

    return CombinedPValues(
        p_value=shaped(p_value),
        statistic=shaped(statistic),
        up_p_value=None if up is None else shaped(up),
        down_p_value=None if down is None else shaped(down),
        sign=None if direction is None else shaped(direction),
        n_studies=read_only(tested.reshape(shape)),
        method=method,
        at_least=at_least,
        rank=rank,
    )


def _combined(
    q: FloatArray,
    qc: FloatArray,
    method: str,
    at_least: int,
    weights: FloatArray | None,
    rank: int | None,
    tested: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """The statistic and p-value of every column of one-sided p-values ``q`` (studies
    x features, NaN where missing), with their complements ``qc``, from the
    ``tested - at_least + 1`` largest of each column."""
    order = np.argsort(q, axis=0, kind="stable")  # NaN sort last
    q = np.take_along_axis(q, order, axis=0)
    qc = np.take_along_axis(qc, order, axis=0)
    position = np.arange(q.shape[0])[:, np.newaxis]
    used = (position >= at_least - 1) & (position < tested)
    m = (tested - at_least + 1).astype(np.float64)  # the p-values combined
    valid = m >= 1.0
    m = np.where(valid, m, np.nan)
    # Every entry that is not combined, NaN ones included, adds nothing.
    q = np.where(used, q, 1.0)
    qc = np.where(used, qc, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        # log q and log(1 - q), each from whichever of q and 1 - q is near 0, which
        # is exact; the other is read through log1p.
        small = q <= 0.5
        log_q = np.where(small, np.log(q), np.log1p(-qc))
        log_qc = np.where(small, np.log1p(-q), np.log(qc))
        if method == "fisher":
            statistic = -2.0 * np.sum(np.where(used, log_q, 0.0), axis=0)
            p_value = chi2_sf(statistic, 2.0 * m)
        elif method == "pearson":
            statistic = -2.0 * np.sum(np.where(used, log_qc, 0.0), axis=0)
            p_value = chi2_cdf(statistic, 2.0 * m)
        elif method == "stouffer":
            # The normal quantile of whichever of q and 1 - q is near 0.
            z = np.where(small, norm_isf(q), -norm_isf(qc))
            w = np.ones_like(q) if weights is None else weights
            w = np.where(used, np.take_along_axis(w, order, axis=0), 0.0)
            statistic = np.sum(np.where(used, w * z, 0.0), axis=0) / np.sqrt(
                np.sum(w * w, axis=0)
            )
            p_value = norm_sf(statistic)
        elif method == "mudholkar_george":
            logit = np.where(used, log_qc - log_q, 0.0)
            scale = (
                np.sqrt(3.0 / m) / math.pi * np.sqrt((5.0 * m + 4.0) / (5.0 * m + 2.0))
            )
            statistic = np.sum(logit, axis=0) * scale
            p_value = t_sf(statistic, 5.0 * m + 4.0)
        elif method == "tippett":
            first = min(at_least - 1, q.shape[0] - 1)
            statistic = q[first]
            p_value = -np.expm1(m * log_qc[first])
        elif method == "wilkinson":
            assert rank is not None
            row = at_least - 1 + rank - 1
            if row < q.shape[0]:
                statistic = np.where(m >= rank, q[row], np.nan)
            else:
                statistic = np.full(q.shape[1], np.nan)
            p_value = beta_cdf(statistic, float(rank), m - rank + 1.0)
        else:  # simes
            j = (position - (at_least - 1) + 1).astype(np.float64)
            # m / j first, as p.adjust orders it: the term of the largest is exact.
            ratios = np.where(used, m / np.where(used, j, 1.0) * q, np.inf)
            statistic = np.minimum(np.min(ratios, axis=0), 1.0)
            p_value = statistic.copy()
    statistic = np.where(valid, np.asarray(statistic, dtype=np.float64), np.nan)
    p_value = np.where(valid, np.asarray(p_value, dtype=np.float64), np.nan)
    return statistic, p_value
