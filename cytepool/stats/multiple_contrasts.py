"""Multiple contrast tests: which treatments or doses differ from the control, with one
error rate for the contrasts of each feature.

A dose series asks, for each gene, which doses differ from the vehicle. Testing each
contrast on its own and correcting by Bonferroni ignores that the contrasts share the
vehicle, which makes their t-statistics correlated, and loses power; an F-test says
whether any dose differs, not which. :func:`multiple_contrast_test` tests every
contrast of every feature and adjusts each p-value for the contrasts of its feature
by the single-step max-t method (Hothorn, Bretz and Westfall 2008): the adjusted
p-value of contrast j is the chance that the largest of the feature's correlated
t-statistics is at least as extreme as t_j, from the multivariate t distribution
(:func:`~cytepool.stats.multivariate_t.multivariate_t_cdf`). It takes estimates and
their covariance, such as the coefficients of a fit, as multcomp's ``glht`` does with
``parm()``. :func:`dunnett_contrasts` builds Dunnett's contrasts, each treatment
against the control (Dunnett 1955), and :func:`williams_contrasts` the Williams-type
contrasts, the control against the size-weighted mean of the highest one, two or
more doses, for means that change in dose order (Bretz 2006), as multcomp's
``contrMat``.
https://docs.nygen.io/CytePool/statistics/multiple-testing#correlated-tests explains
the tests.

Differences from multcomp:
    - ``df`` has no default and need not be an integer, as moderated degrees of
      freedom are not; multcomp's default of 0 gives z-tests, here ``np.inf``.
    - Williams-type contrasts are named by the doses they pool, such as
      "mid to high - vehicle", where ``contrMat`` names them "C 1", "C 2", ...; with
      two groups they are Dunnett's single contrast, which ``contrMat`` refuses.
    - Each adjusted p-value carries a standard error of at most ``relative_error`` of
      it (1% by default) at every magnitude, where mvtnorm's default allows an
      absolute error of 1e-3.

References:
    Dunnett CW (1955). A multiple comparison procedure for comparing several
    treatments with a control. Journal of the American Statistical Association 50,
    1096-1121. doi:10.1080/01621459.1955.10501294

    Bretz F (2006). An extension of the Williams trend test to general unbalanced
    linear models. Computational Statistics and Data Analysis 50, 1735-1748.
    doi:10.1016/j.csda.2005.02.005

    Hothorn T, Bretz F, Westfall P (2008). Simultaneous inference in general
    parametric models. Biometrical Journal 50, 346-363. doi:10.1002/bimj.200810425
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._columns import text_column
from cytepool._jit import thread_limit
from cytepool._masked_arrays import read_only, reject_masked, unmasked_array
from cytepool._messages import close_matches, listing, quoted
from cytepool._options import checked_choice, integral_number
from cytepool.design._contrast_weights import (
    checked_names,
    contrast_matrix,
    is_named_contrasts,
    is_pandas,
)
from cytepool.design.contrasts import Contrasts
from cytepool.stats._result_tables import coefficient_position, feature_id_column
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.stats.multivariate_t import multivariate_t_cdf
from cytepool.table import Table

__all__ = [
    "MultipleContrastTest",
    "dunnett_contrasts",
    "multiple_contrast_test",
    "williams_contrasts",
]

FloatArray = npt.NDArray[np.float64]

_ALTERNATIVES = ("two-sided", "greater", "less")
_EPS = float(np.finfo(np.float64).eps)
# A contrast whose correlation matrix has an eigenvalue below this times the largest
# is not positive semidefinite.
_EIGENVALUE_TOLERANCE = 1e-10


# ---------------------------------------------------------------------------------------
# Contrast matrices
# ---------------------------------------------------------------------------------------


def _group_names(groups: Any, name: str) -> list[str]:
    reject_masked(groups, name)
    if isinstance(groups, str):
        raise TypeError(
            f"{name} must be a sequence of group names, not a single string"
        )
    names = list(groups.keys() if isinstance(groups, Mapping) else groups)
    if len(names) < 2:
        raise ValueError(f"{name} must hold at least two groups, got {len(names)}")
    for position, group in enumerate(names):
        if not isinstance(group, str):
            raise TypeError(
                f"{name} must name its groups with text, got {type(group).__name__} "
                f"{group!r} at position {position}"
            )
        if not group:
            raise ValueError(f"{name} has an empty group name at position {position}")
    repeated = sorted({group for group in names if names.count(group) > 1})
    if repeated:
        raise ValueError(
            f"{name} must name each group once; repeated: {quoted(repeated)}"
        )
    return names


def dunnett_contrasts(
    groups: Sequence[str] | Mapping[str, Any], *, control: str | int = 0
) -> Contrasts:
    """Dunnett's contrasts: each treatment or dose against the control.

    Contrast "{group} - {control}" weighs the group 1 and the control -1, as
    multcomp's ``contrMat(n, type = "Dunnett")`` names and weighs them; the contrasts
    follow the order of the groups. Dunnett's test (Dunnett 1955) is
    :func:`multiple_contrast_test` of these contrasts.
    https://docs.nygen.io/CytePool/statistics/multiple-testing#dunnett-and-williams-contrasts
    explains them.

    Args:
        groups: The group names, in order, as a sequence, or the keys of a mapping
            such as the group sizes that :func:`williams_contrasts` takes (Dunnett's
            weights do not depend on the sizes).
        control: The control group, by name or 0-based position; the first group by
            default.

    Returns:
        The contrasts, with the groups as the columns they weigh.

    Raises:
        ValueError: If there are fewer than two groups, a name is empty or
            repeated, or ``control`` is not one of the groups.
        TypeError: If ``groups`` is a single string or names a group with something
            other than text, or an argument is a numpy masked array.

    Example:
        >>> dunnett_contrasts(["vehicle", "low", "high"]).names
        ('low - vehicle', 'high - vehicle')
    """
    names = _group_names(groups, "groups")
    reject_masked(control, "control")
    if isinstance(control, str):
        if control not in names:
            raise ValueError(
                f"control {control!r} is not one of the groups: {quoted(names)}"
            )
        reference = names.index(control)
    else:
        position = integral_number(
            control, "control", expected="a group name or a position"
        )
        if not -len(names) <= position < len(names):
            raise ValueError(
                f"control position {position} is out of range for {len(names)} groups"
            )
        reference = position % len(names)
    treated = [j for j in range(len(names)) if j != reference]
    matrix = np.zeros((len(names), len(treated)))
    for column, j in enumerate(treated):
        matrix[j, column] = 1.0
        matrix[reference, column] = -1.0
    labels = tuple(f"{names[j]} - {names[reference]}" for j in treated)
    return Contrasts(matrix, labels, tuple(names))


def williams_contrasts(sizes: Sequence[int] | Mapping[str, int]) -> Contrasts:
    """Williams-type contrasts: the control against the size-weighted mean of the
    highest one, two or more doses.

    The first group is the control and the doses follow in increasing order.
    Contrast j = 1, ..., K - 1 weighs the control -1 and each of the j highest doses
    n_i / (the sum of their sizes), as multcomp's ``contrMat(n, type = "Williams")``
    (Bretz 2006). Tested one-sided with :func:`multiple_contrast_test`, they detect a
    change that grows in dose order, and their largest t-statistic picks how many of
    the top doses have changed. They differ from Williams' (1971) trend test, which
    compares each dose with the control after fitting means that change in dose
    order. https://docs.nygen.io/CytePool/statistics/multiple-testing#dunnett-and-williams-contrasts
    explains them.

    Args:
        sizes: The replicates of each group, in dose order with the control first,
            as a mapping of group name to count, or a sequence of counts, whose
            groups are then named "1", "2", ..., as multcomp names unnamed groups.

    Returns:
        The contrasts, named by the doses they pool, such as "high - vehicle",
        "mid to high - vehicle" and "low to high - vehicle", with the groups as the
        columns they weigh. With two groups, the one contrast is Dunnett's.

    Raises:
        ValueError: If there are fewer than two groups, a count is not a positive
            integer, or a name is empty or repeated.
        TypeError: If ``sizes`` is a single string or a count is not a number, or an
            argument is a numpy masked array.

    Example:
        >>> williams_contrasts({"vehicle": 4, "low": 3, "high": 3}).names
        ('high - vehicle', 'low to high - vehicle')
    """
    reject_masked(sizes, "sizes")
    if isinstance(sizes, str):
        raise TypeError("sizes must give one count per group, not a single string")
    if isinstance(sizes, Mapping):
        names = _group_names(sizes, "sizes")
        values = list(sizes.values())
    else:
        values = list(sizes)
        names = [str(j + 1) for j in range(len(values))]
        if len(values) < 2:
            raise ValueError(f"sizes must hold at least two groups, got {len(values)}")
    counts = []
    for group, value in zip(names, values):
        count = integral_number(
            value, f"the size of group {group!r}", expected="an integer"
        )
        if count < 1:
            raise ValueError(
                f"the size of group {group!r} must be a positive integer, got {count}"
            )
        counts.append(count)
    k = len(names)
    matrix = np.zeros((k, k - 1))
    labels = []
    for j in range(1, k):
        pooled = range(k - j, k)
        total = sum(counts[i] for i in pooled)
        matrix[0, j - 1] = -1.0
        for i in pooled:
            matrix[i, j - 1] = counts[i] / total
        top, bottom = names[-1], names[k - j]
        doses = top if j == 1 else f"{bottom} to {top}"
        labels.append(f"{doses} - {names[0]}")
    return Contrasts(matrix, tuple(labels), tuple(names))


# ---------------------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False, repr=False)
class MultipleContrastTest:
    """Multiple contrast tests of every feature, adjusted for the feature's contrasts.

    For each feature and contrast c: the estimate c'b, its standard error
    sqrt(c' V c), t, and the single-step adjusted p-value, the chance that the
    largest of the feature's t-statistics on its ``df`` is at least as extreme as t
    (Hothorn, Bretz and Westfall 2008). The arrays are read-only.

    Attributes:
        names: The contrast names.
        estimate_names: The names of the estimates, which the rows of ``contrasts``
            weigh.
        contrasts: ``(n_estimates, n_contrasts)`` weights of each contrast.
        estimate: ``(n_features, n_contrasts)`` estimates of the contrasts.
        standard_error: ``(n_features, n_contrasts)`` their standard errors.
        t: ``(n_features, n_contrasts)`` t-statistics.
        p_value: ``(n_features, n_contrasts)`` p-values adjusted for the contrasts of
            the feature (single-step): P(max_l \\|T_l\\| >= \\|t\\|) two-sided,
            P(max_l T_l >= t) for ``"greater"`` and P(min_l T_l <= t) for
            ``"less"``. Within a feature they never decrease as t becomes less
            extreme.
        p_value_error: ``(n_features, n_contrasts)`` the standard error of each
            p-value (the quadrature's error bound with two contrasts, 0 with one).
        df: ``(n_features,)`` degrees of freedom of the t-statistics.
        alternative: ``"two-sided"``, ``"greater"`` or ``"less"``.
        relative_error: The requested standard error of the smaller of each p-value
            and 1 minus it, as a fraction of it.
    """

    names: tuple[str, ...]
    estimate_names: tuple[str, ...]
    contrasts: FloatArray
    estimate: FloatArray
    standard_error: FloatArray
    t: FloatArray
    p_value: FloatArray
    p_value_error: FloatArray
    df: FloatArray
    alternative: str
    relative_error: float

    def table(
        self, contrast: str | int | None = None, *, feature_ids: Any = None
    ) -> Table:
        """Return the tests as a table, one row per feature.

        Without ``contrast``, the test of each feature as a whole, the max-t test of
        its contrasts: ``feature_id``; ``contrast``, the contrast with the most
        extreme t, whose adjusted p-value is the feature's; its ``t``, ``p_value``
        and ``p_value_error``; and ``adjusted_p_value``, Benjamini and Hochberg's
        adjustment of ``p_value`` over the features. With ``contrast``: its
        ``estimate``, ``standard_error``, ``t``, ``p_value`` and ``p_value_error``.
        This ``p_value`` is adjusted for the feature's contrasts, unlike the
        p-values of other CytePool tables; across features, call the features from
        the table without ``contrast``.
        https://docs.nygen.io/CytePool/statistics/multiple-testing#correlated-tests
        explains the two levels.

        Args:
            contrast: A contrast, by name or position; None for the test of each
                feature as a whole.
            feature_ids: One id per feature, as text; by default ``feature_0``,
                ``feature_1``, ...

        Raises:
            KeyError: If no contrast has the name.
            IndexError: If the position is out of range.
            ValueError: If ``feature_ids`` does not give one id per feature.
            TypeError: If an id is not text, ``contrast`` is neither a name nor a
                position, or an argument is or holds a numpy masked array.
        """
        n_features = self.t.shape[0]
        ids = feature_id_column(feature_ids, n_features)
        if contrast is None:
            extremeness = _extremeness(self.t, self.alternative)
            usable = ~np.isnan(extremeness).all(axis=1)
            filled = np.where(np.isnan(extremeness), -np.inf, extremeness)
            chosen = np.argmax(filled, axis=1)
            rows = np.arange(n_features)
            p_value = np.where(usable, self.p_value[rows, chosen], np.nan)
            return Table(
                {
                    "feature_id": ids,
                    "contrast": text_column(
                        [self.names[j] if ok else None for j, ok in zip(chosen, usable)]
                    ),
                    "t": np.where(usable, self.t[rows, chosen], np.nan),
                    "p_value": p_value,
                    "p_value_error": np.where(
                        usable, self.p_value_error[rows, chosen], np.nan
                    ),
                    "adjusted_p_value": np.asarray(
                        adjust_p_values(p_value, "bh"), dtype=np.float64
                    ),
                }
            )
        j = coefficient_position(contrast, self.names)
        return Table(
            {
                "feature_id": ids,
                "estimate": self.estimate[:, j],
                "standard_error": self.standard_error[:, j],
                "t": self.t[:, j],
                "p_value": self.p_value[:, j],
                "p_value_error": self.p_value_error[:, j],
            }
        )

    def __repr__(self) -> str:
        overall = self.table()["adjusted_p_value"]
        called = int(np.count_nonzero(overall < 0.05))
        # 1 - p of a p-value near 1 is known only to the rounding of p.
        smaller = np.minimum(self.p_value, 1.0 - self.p_value)
        allowed = self.relative_error * smaller + _EPS * self.p_value
        short = int(np.count_nonzero(self.p_value_error > allowed))
        return (
            f"MultipleContrastTest(n_features={self.t.shape[0]}, names={self.names!r}, "
            f"alternative={self.alternative!r}, adjusted_p_value_below_0.05={called}, "
            f"p_values_short_of_relative_error={short})"
        )


def _extremeness(t: FloatArray, alternative: str) -> FloatArray:
    """How extreme each t-statistic is for the alternative: larger is more extreme."""
    if alternative == "two-sided":
        return np.abs(t)
    return t if alternative == "greater" else -t


def _named_rows(weights: Any) -> list[str] | None:
    """The estimate names that weights name their rows by, or None for weights read
    in order."""
    if is_named_contrasts(weights):
        return [str(name) for name in weights.columns]
    if is_pandas(weights):
        index = list(weights.index)
        if index and all(isinstance(name, str) for name in index):
            return index
    return None


def _contrast_weights(
    contrasts: Any, estimate_names: Any, n_estimates: int
) -> tuple[FloatArray, tuple[str, ...], tuple[str, ...]]:
    """The contrast matrix, rows in the order of the estimates, its names and the
    estimate names."""
    if estimate_names is None:
        names = tuple(f"estimate{j + 1}" for j in range(n_estimates))
    else:
        reject_masked(estimate_names, "estimate_names")
        names = checked_names(estimate_names, n_estimates, "estimate_names")
    if contrasts is None:
        return np.eye(n_estimates), names, names
    reject_masked(contrasts, "contrasts")
    if isinstance(contrasts, Mapping) and not is_named_contrasts(contrasts):
        parts = list(contrasts.values())
    else:
        parts = [contrasts]
    for part in parts:
        rows = _named_rows(part)
        if rows is None:
            continue
        if estimate_names is None:
            raise ValueError(
                "the contrasts name the estimates they weigh "
                f"({quoted(rows)}), so estimate_names must name the estimates, in the "
                "order of their columns, for the contrasts to be matched by name"
            )
        if sorted(rows) != sorted(names):
            missing = [name for name in names if name not in rows]
            unexpected = [name for name in rows if name not in names]
            problems = []
            if missing:
                problems.append(f"they do not weigh {quoted(missing)}")
            if unexpected:
                hinted = [
                    repr(name) + close_matches(name, missing) for name in unexpected
                ]
                which = "which is" if len(unexpected) == 1 else "which are"
                problems.append(
                    f"they weigh {listing(hinted)}, {which} not among estimate_names"
                )
            raise ValueError(
                f"the contrasts must weigh exactly the estimates of estimate_names "
                f"({quoted(list(names))}), in any order, but {'; '.join(problems)}"
            )
    matrix, contrast_names = contrast_matrix(contrasts, None, names)
    zero = np.flatnonzero(~matrix.any(axis=0))
    if zero.size:
        raise ValueError(
            f"contrast {contrast_names[zero[0]]!r} has only zero weights, so it tests "
            "nothing"
        )
    return matrix, contrast_names, names


def _feature_df(df: Any, n_features: int) -> FloatArray:
    if isinstance(df, bool | np.bool_):
        raise TypeError("df must be a number or one number per feature, got bool")
    values = unmasked_array(df, "df", missing="NaN")
    if values.dtype.kind not in "iuf":
        raise TypeError(f"df must hold numbers, got an array of dtype {values.dtype}")
    values = np.asarray(values, dtype=np.float64)
    if values.ndim > 1:
        raise ValueError(
            "df must be one number or one per feature: the multivariate t distribution "
            f"takes one df per feature, got shape {values.shape}; where each contrast "
            "has its own df, as in dream's tests, the smallest of a feature's is "
            "conservative"
        )
    if values.ndim == 1 and values.size != n_features:
        raise ValueError(
            f"df must be one number or one per feature ({n_features}), got "
            f"{values.size}"
        )
    values = np.broadcast_to(values, (n_features,)).astype(np.float64)
    bad = np.flatnonzero(values <= 0.0)
    if bad.size:
        raise ValueError(
            f"df must be positive, or np.inf for z-tests; feature {bad[0]} has "
            f"{values[bad[0]]}"
        )
    return values


def multiple_contrast_test(
    estimates: npt.ArrayLike,
    covariances: npt.ArrayLike,
    contrasts: Any = None,
    *,
    df: npt.ArrayLike,
    estimate_names: Sequence[str] | None = None,
    alternative: str = "two-sided",
    relative_error: float = 0.01,
    rng: Any = None,
    n_threads: int | None = None,
) -> MultipleContrastTest:
    """Test several contrasts of every feature, with one error rate per feature.

    For each feature, the contrasts C' b of its estimates b, their standard errors
    from C' V C for its covariance V, and their t-statistics, which are correlated
    through V; each p-value is adjusted for the feature's contrasts by the
    single-step max-t method (Hothorn, Bretz and Westfall 2008), from the
    multivariate t distribution on the feature's ``df`` with the correlation of
    C' V C (:func:`~cytepool.stats.multivariate_t.multivariate_t_cdf`), as multcomp's
    ``summary(glht(parm(b, V, df), linfct = t(C)))``. With :func:`dunnett_contrasts`
    this is Dunnett's test of each dose against the vehicle; with
    :func:`williams_contrasts` and ``alternative="greater"`` (or ``"less"``), the
    Williams-type test of a change that grows in dose order.
    https://docs.nygen.io/CytePool/statistics/multiple-testing#correlated-tests
    explains the tests.

    Contrasts that name the estimates they weigh, such as those of
    :func:`dunnett_contrasts` and :func:`williams_contrasts`, are matched to
    ``estimate_names`` by name, never by position; other weights are taken in the
    order of the estimates. A feature with a missing or infinite estimate or
    covariance, or a contrast of zero variance, gets NaN for every contrast.

    Args:
        estimates: ``(n_features, n_estimates)`` estimates, such as group means or
            the coefficients of a fit, or ``(n_estimates,)`` for one feature.
        covariances: Their covariance matrices, ``(n_features, n_estimates,
            n_estimates)``, or one ``(n_estimates, n_estimates)`` matrix for every
            feature; positive semidefinite.
        contrasts: The contrasts to test: a :class:`~cytepool.design.Contrasts`, a
            mapping of name to weights, or a ``(n_estimates, n_contrasts)`` matrix;
            None tests the estimates themselves.
        df: Degrees of freedom of the t-statistics: one number, or one per feature,
            positive; ``np.inf`` for z-tests. Need not be integers, as moderated
            degrees of freedom are not.
        estimate_names: The names of the estimates, in order, which named contrasts
            are matched to; required by them.
        alternative: ``"two-sided"``, ``"greater"`` (the contrasts are larger than
            0) or ``"less"``.
        relative_error: The largest standard error of the smaller of each adjusted
            p-value and 1 minus it, as a fraction of it, between 0 and 1.
        rng: A numpy random generator, an integer seed, or None; a seed gives the
            same p-values whatever the number of threads. Two or fewer contrasts use
            no random numbers.
        n_threads: The most threads to use; None uses numba's setting. The results
            do not depend on it.

    Returns:
        The tests, with one table per contrast and one of each feature as a whole.

    Raises:
        ValueError: If the shapes disagree, named contrasts do not weigh exactly the
            estimates of ``estimate_names`` or come without them, a contrast is all
            zero, ``df`` is not positive or is not one number per feature, the
            contrasts of a feature have a covariance that is not positive
            semidefinite, ``alternative`` is unknown, or ``relative_error`` is not
            between 0 and 1.
        TypeError: If an argument has the wrong type or is a numpy masked array.

    Example:
        >>> import numpy as np
        >>> sizes = {"vehicle": 4, "low": 4, "high": 4}
        >>> means = np.array([[5.0, 5.2, 6.1]])
        >>> covariance = np.diag([0.05, 0.05, 0.05])
        >>> test = multiple_contrast_test(
        ...     means, covariance, dunnett_contrasts(sizes), df=9.5,
        ...     estimate_names=list(sizes))
        >>> test.p_value[0].round(4).tolist()  # low and high against the vehicle
        [0.7584, 0.0117]
    """
    b = unmasked_array(estimates, "estimates", missing="NaN")
    if b.dtype.kind not in "biuf":
        raise TypeError(f"estimates must hold numbers, got an array of dtype {b.dtype}")
    b = np.asarray(b, dtype=np.float64)
    if b.ndim == 1:
        b = b[np.newaxis, :]
    if b.ndim != 2 or b.shape[1] == 0:
        raise ValueError(
            "estimates must have shape (n_features, n_estimates) or (n_estimates,), "
            f"got {np.shape(estimates)}"
        )
    n_features, n_estimates = b.shape
    v = unmasked_array(covariances, "covariances", missing="NaN")
    if v.dtype.kind not in "biuf":
        raise TypeError(
            f"covariances must hold numbers, got an array of dtype {v.dtype}"
        )
    v = np.asarray(v, dtype=np.float64)
    if v.shape not in (
        (n_estimates, n_estimates),
        (n_features, n_estimates, n_estimates),
    ):
        raise ValueError(
            f"covariances must have shape ({n_estimates}, {n_estimates}) or "
            f"({n_features}, {n_estimates}, {n_estimates}) for estimates of shape "
            f"{b.shape}, got {v.shape}"
        )
    matrix, names, estimate_labels = _contrast_weights(
        contrasts, estimate_names, n_estimates
    )
    side = checked_choice(alternative, "alternative", _ALTERNATIVES)
    feature_df = _feature_df(df, n_features)
    estimate = b @ matrix
    covariance = np.einsum("ia,...ij,jb->...ab", matrix, v, matrix)
    covariance = np.broadcast_to(covariance, (n_features, *covariance.shape[-2:]))
    variance = np.diagonal(covariance, axis1=1, axis2=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        standard_error = np.sqrt(variance)
        t = estimate / standard_error
        correlation = covariance / (
            standard_error[:, :, None] * standard_error[:, None, :]
        )
    if v.ndim == 2:
        finite_covariance = np.full(n_features, np.isfinite(v).all())
    else:
        finite_covariance = np.isfinite(v.reshape(n_features, -1)).all(axis=1)
    complete = (
        np.isfinite(b).all(axis=1)
        & finite_covariance
        & ~np.isnan(feature_df)
        & (variance > 0.0).all(axis=1)
    )
    _check_positive_semidefinite(correlation, complete)
    nan = ~complete
    estimate[nan] = np.nan
    standard_error = np.where(nan[:, None], np.nan, standard_error)
    t = np.where(nan[:, None], np.nan, t)
    correlation = np.where(nan[:, None, None], np.nan, correlation)
    m = matrix.shape[1]
    if side == "two-sided":
        thresholds, two_sided = np.abs(t), True
    else:
        thresholds, two_sided = (t if side == "greater" else -t), False
    upper = np.repeat(thresholds[:, :, None], m, axis=2)
    with thread_limit(n_threads):
        probabilities = multivariate_t_cdf(
            upper,
            correlation[:, None, :, :],
            feature_df[:, None],
            two_sided=two_sided,
            relative_error=relative_error,
            rng=rng,
        )
    p_value, p_value_error = _monotone(
        np.asarray(probabilities.complement).reshape(n_features, m),
        np.asarray(probabilities.error).reshape(n_features, m),
        _extremeness(t, side),
    )
    return MultipleContrastTest(
        names=names,
        estimate_names=estimate_labels,
        contrasts=read_only(np.ascontiguousarray(matrix)),
        estimate=read_only(np.ascontiguousarray(estimate)),
        standard_error=read_only(np.ascontiguousarray(standard_error)),
        t=read_only(np.ascontiguousarray(t)),
        p_value=read_only(p_value),
        p_value_error=read_only(p_value_error),
        df=read_only(feature_df),
        alternative=side,
        relative_error=float(relative_error),
    )


def _check_positive_semidefinite(correlation: FloatArray, complete: np.ndarray) -> None:
    """Refuse a feature whose contrasts' correlation matrix is not positive
    semidefinite."""
    rows = np.flatnonzero(complete)
    if rows.size == 0:
        return
    stack = correlation[rows]
    eigenvalues = np.linalg.eigvalsh(0.5 * (stack + stack.transpose(0, 2, 1)))
    smallest = eigenvalues[:, 0] / eigenvalues[:, -1]
    bad = np.flatnonzero(smallest < -_EIGENVALUE_TOLERANCE)
    if bad.size:
        feature = int(rows[bad[0]])
        raise ValueError(
            f"the covariance of feature {feature} is not positive semidefinite: the "
            f"correlation of its contrasts has the eigenvalue "
            f"{eigenvalues[bad[0], 0]:.3g} against a largest of "
            f"{eigenvalues[bad[0], -1]:.3g}"
        )


def _monotone(
    p_value: FloatArray, error: FloatArray, extremeness: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """P-values that never decrease as t becomes less extreme within a feature, by a
    running maximum in decreasing extremeness; a raised p-value takes the value and
    the error of the more extreme contrast whose estimate it takes."""
    order = np.argsort(
        -np.where(np.isnan(extremeness), -np.inf, extremeness), axis=1, kind="stable"
    )
    sorted_p = np.take_along_axis(p_value, order, axis=1)
    running = np.maximum.accumulate(sorted_p, axis=1)
    positions = np.arange(p_value.shape[1])
    source = np.maximum.accumulate(np.where(sorted_p >= running, positions, 0), axis=1)
    sorted_error = np.take_along_axis(
        np.take_along_axis(error, order, axis=1), source, axis=1
    )
    out_p = np.empty_like(p_value)
    out_error = np.empty_like(error)
    np.put_along_axis(out_p, order, running, axis=1)
    np.put_along_axis(out_error, order, sorted_error, axis=1)
    return out_p, out_error
