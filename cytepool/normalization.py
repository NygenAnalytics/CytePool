"""Library-size normalization and expression filtering for count data.

Counts are samples x features (rows are samples, columns are genes): finite,
non-negative, integer or not. They are converted to float64, which holds every
whole number up to 2**53 (about 9.0e15) exactly; a larger integer count is
rounded to the nearest float64, a relative change below 1.2e-16, and never wraps
around. R, whose integers have 32 bits, holds counts that large as doubles too.
A library size, a total of a feature's counts or a count per million beyond the
float64 range (about 1.8e308) raises OverflowError instead of becoming infinite
or NaN. The functions reproduce edgeR and DESeq2:

- normalization factors as edgeR's ``normLibSizes`` (``calcNormFactors``)
  computes them: :func:`tmm_factors`, :func:`upper_quartile_factors` and
  :func:`rle_factors`. They multiply the library sizes into effective library
  sizes, ``library_sizes * norm_factors``, and have geometric mean one;
- DESeq2's size factors, which take the place of the library sizes:
  :func:`median_ratio_size_factors`;
- counts per million and log counts per million as edgeR's ``cpm`` computes
  them: :func:`cpm` and :func:`log_cpm`;
- edgeR's ``filterByExpr``: :func:`filter_by_expression`.

The routines repeat R's floating-point operations wherever an operation
decides a discrete outcome: the rank-based trimming and the reference sample
of TMM, and the comparisons of ``filterByExpr``. Sums and means that R
accumulates in extended precision (``sum``, ``mean``, ``colSums``,
``rowMeans``) are accumulated with error compensation and rounded once.
Logarithms and exponentials come from the C math library, as in R. With these,
the results agree with edgeR and DESeq2 to the last bit on the validation
inputs (numba set up to use Intel's SVML library may change the last bit).
Conventions that the documentation of edgeR, DESeq2 or R leaves open were
found by running them as black boxes; comments mark them "found by probing".

References:
    Robinson MD, Oshlack A (2010). Genome Biology 11:R25 (TMM).
    Bullard JH et al. (2010). BMC Bioinformatics 11:94 (upper quartile).
    Anders S, Huber W (2010). Genome Biology 11:R106 (median of ratios).
    Chen Y, Lun ATL, Smyth GK (2016). F1000Research 5:1438 (filterByExpr).
"""

import math
import numbers
import warnings
from collections import Counter

import numpy as np
import numpy.typing as npt

from ._jit import jit
from ._summation import accurate_mean, accurate_sums, rounded_quotient, two_sum
from ._linear_algebra import column_space
from ._masked_arrays import unmasked_array

__all__ = [
    "cpm",
    "filter_by_expression",
    "library_sizes",
    "log_cpm",
    "median_ratio_size_factors",
    "rle_factors",
    "tmm_factors",
    "upper_quartile_factors",
]

FloatArray = npt.NDArray[np.float64]

# TMM gives factor 1 when every log ratio is smaller than this in absolute
# value (found by probing: edgeR treats the two samples as identical).
_SAME_PROFILE = 1e-6
# filterByExpr compares the number of samples and the total count with this
# slack (found by probing); the comparison of CPM with the cutoff has none.
_FILTER_SLACK = 1e-14
# Below this median upper quartile, edgeR picks the TMM reference sample by
# its square-root counts instead (found by probing).
_TINY_UPPER_QUARTILE = 1e-20


def library_sizes(counts: npt.ArrayLike) -> FloatArray:
    """Return the library size of each sample: its total count over all features.

    The sums are accumulated with error compensation and rounded once, like R's
    ``colSums``, so they are exact for whole-number counts whose total is at most
    2**53.

    Args:
        counts: Counts, samples x features.

    Returns:
        Float64 array of shape ``(n_samples,)``.

    Raises:
        TypeError: If counts are not numeric, or are a numpy masked array.
        ValueError: If counts are not a non-empty 2-D array of finite,
            non-negative values.
        OverflowError: If a library size exceeds the float64 range.
    """
    return _sums_in_range(_as_counts(counts), 1)


def tmm_factors(
    counts: npt.ArrayLike,
    *,
    library_sizes: npt.ArrayLike | None = None,
    reference: int | None = None,
    log_ratio_trim: float = 0.3,
    sum_trim: float = 0.05,
    weighted: bool = True,
    a_cutoff: float = -1e10,
) -> FloatArray:
    """Return TMM normalization factors, as edgeR's ``calcNormFactors`` computes them.

    The trimmed mean of M values (Robinson and Oshlack 2010) compares each sample
    k with a reference sample r over the features with positive counts in both.
    With proportions ``p = y / N``, where N is the library size, a feature has
    log ratio ``M = log2(p_k / p_r)`` and average log proportion
    ``A = (log2 p_k + log2 p_r) / 2``. Features with ``A <= a_cutoff`` are
    dropped. Of the n features left, those whose M lies in the lowest or
    highest ``log_ratio_trim`` fraction, or whose A lies in the lowest or
    highest ``sum_trim`` fraction, are trimmed: a feature is kept when its
    average rank lies within ``floor(n * trim) + 1`` and ``n - floor(n * trim)``
    for both M and A, so tied values are kept or trimmed together. The factor
    of sample k is 2 to the mean of the remaining M values, weighted by
    ``1 / v`` with the delta-method variance
    ``v = (N_k - y_k) / (N_k y_k) + (N_r - y_r) / (N_r y_r)``.
    A sample with no feature left, or whose log ratios are all smaller than
    1e-6 in absolute value, gets factor 1. Finally the factors are scaled to
    geometric mean one.

    Features with zero counts in every sample are left out, as edgeR removes
    them; this matters only for the upper quartiles that pick the default
    reference.

    Args:
        counts: Counts, samples x features.
        library_sizes: Library size of each sample; defaults to the total
            counts. Must be positive.
        reference: Index of the reference sample. The default is edgeR's: the
            sample whose upper quartile of counts divided by its library size
            is closest to the mean of that ratio over samples (the first one
            when several are equally close). When the median of the ratios is
            below 1e-20, as with very sparse counts, it is the sample with the
            largest sum of square-root counts instead (found by probing).
        log_ratio_trim: Fraction of features trimmed from each end of the M
            values, in [0, 0.5).
        sum_trim: Fraction of features trimmed from each end of the A values,
            in [0, 0.5).
        weighted: Whether to weight the M values by their inverse variances.
        a_cutoff: Features with A at or below this value are not used.

    Returns:
        Float64 array of shape ``(n_samples,)`` with geometric mean one.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: If an argument is out of range, or a sample has library
            size zero.
        OverflowError: If a library size exceeds the float64 range.
    """
    y = _as_counts(counts)
    sizes = _library_sizes_of(y, library_sizes)
    if reference is not None:
        reference = _sample_index(reference, y.shape[0])
    trim_m = _fraction(log_ratio_trim, "log_ratio_trim")
    trim_a = _fraction(sum_trim, "sum_trim")
    if not isinstance(weighted, bool | np.bool_):
        raise TypeError(f"weighted must be a bool, got {type(weighted).__name__}")
    cutoff = _real(a_cutoff, "a_cutoff", finite=False)
    expressed = _expressed(y)
    if expressed.shape[1] == 0:
        return np.ones(y.shape[0])
    if reference is None:
        reference = _tmm_reference(expressed, sizes)
    log2_factors = _tmm_log2_factors(
        expressed, sizes, reference, trim_m, trim_a, bool(weighted), cutoff
    )
    # 2.0 ** t is C's pow(2, t), as R computes 2^t.
    return _to_geometric_mean_one(np.array([2.0**t for t in log2_factors]))


def upper_quartile_factors(
    counts: npt.ArrayLike,
    *,
    library_sizes: npt.ArrayLike | None = None,
    p: float = 0.75,
) -> FloatArray:
    """Return upper-quartile normalization factors, as edgeR computes them.

    The factor of a sample is the ``p`` quantile of its counts (R's default
    quantile, type 7) over the features that are not zero in every sample,
    divided by its library size (Bullard et al. 2010). The factors are scaled
    to geometric mean one.

    Args:
        counts: Counts, samples x features.
        library_sizes: Library size of each sample; defaults to the total
            counts. Must be positive.
        p: Quantile to use, in [0, 1].

    Returns:
        Float64 array of shape ``(n_samples,)`` with geometric mean one.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: If an argument is out of range, a sample has library size
            zero, or the quantile is zero in some sample (edgeR then returns
            NaN factors).
        OverflowError: If a library size exceeds the float64 range.
    """
    y = _as_counts(counts)
    sizes = _library_sizes_of(y, library_sizes)
    p = _real(p, "p")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    expressed = _expressed(y)
    if expressed.shape[1] == 0:
        raise ValueError("every count is zero, so every quantile is zero")
    quantiles = _row_quantiles(expressed, p)
    zero = np.flatnonzero(quantiles == 0)
    if zero.size:
        raise ValueError(
            f"the {p} quantile of the counts is zero in sample(s) {zero.tolist()}; "
            "use a larger p or TMM factors"
        )
    return _to_geometric_mean_one(quantiles / sizes)


def rle_factors(
    counts: npt.ArrayLike, *, library_sizes: npt.ArrayLike | None = None
) -> FloatArray:
    """Return relative log expression (RLE) factors, as edgeR computes them.

    This is the median-ratio method of Anders and Huber (2010) in edgeR's form.
    Over the features with positive counts in every sample, a sample's counts
    are divided by the features' geometric means across samples; the factor is
    the median of these ratios divided by the library size. edgeR takes the
    median of the ratios, where DESeq2 takes the median of their logarithms
    (found by probing); the two differ when the number of features is even.
    The factors are scaled to geometric mean one.

    Args:
        counts: Counts, samples x features.
        library_sizes: Library size of each sample; defaults to the total
            counts. Must be positive.

    Returns:
        Float64 array of shape ``(n_samples,)`` with geometric mean one.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: If a sample has library size zero, or no feature has
            positive counts in every sample (edgeR then returns NaN factors).
        OverflowError: If a library size exceeds the float64 range.
    """
    y = _as_counts(counts)
    sizes = _library_sizes_of(y, library_sizes)
    medians = _rle_median_ratios(y)
    if np.isnan(medians[0]):
        raise ValueError(
            "no feature has positive counts in every sample, so RLE factors are "
            "undefined; use TMM factors"
        )
    return _to_geometric_mean_one(medians / sizes)


def median_ratio_size_factors(
    counts: npt.ArrayLike, *, method: str = "ratio"
) -> FloatArray:
    """Return DESeq2's median-of-ratios size factors.

    With ``method="ratio"`` (Anders and Huber 2010, equation 5), the size factor
    of a sample is the median, over the features with positive counts in every
    sample, of its counts divided by the features' geometric means across
    samples. As in DESeq2, the median is taken of the log ratios (found by
    probing), and the factors are not rescaled.

    With ``method="poscounts"``, a feature's geometric mean is the n-th root of
    the product of its positive counts, n being the number of samples, so
    features with zeros take part. A sample's median runs over the features
    where its count is positive. The factors are then scaled to geometric mean
    one, as DESeq2's ``estimateSizeFactors()``, and so ``DESeq()``, returns them
    (found by probing). The lower-level ``estimateSizeFactorsForMatrix()``
    returns them before this scaling.

    Args:
        counts: Counts, samples x features.
        method: ``"ratio"`` or ``"poscounts"``.

    Returns:
        Float64 array of shape ``(n_samples,)``.

    Raises:
        TypeError: If counts are not numeric, or are a numpy masked array.
        ValueError: If method is unknown; with ``"ratio"``, if no feature has
            positive counts in every sample; with ``"poscounts"``, if a sample
            has no positive count.
    """
    y = _as_counts(counts)
    if method not in ("ratio", "poscounts"):
        raise ValueError(f'method must be "ratio" or "poscounts", got {method!r}')
    positive_only = method == "poscounts"
    medians = _median_log_ratios(y, positive_only)
    missing = np.flatnonzero(np.isnan(medians))
    if missing.size and not positive_only:
        raise ValueError(
            "every feature has a zero count in some sample, so every geometric "
            'mean is zero; use method="poscounts"'
        )
    if missing.size:
        raise ValueError(f"sample(s) {missing.tolist()} have no positive count")
    factors = np.array([math.exp(m) for m in medians])
    return _to_geometric_mean_one(factors) if positive_only else factors


def cpm(
    counts: npt.ArrayLike,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
) -> FloatArray:
    """Return counts per million, as edgeR's ``cpm`` computes them.

    The value is ``1e6 * y / (N f)`` for library size N and normalization
    factor f.

    Args:
        counts: Counts, samples x features.
        library_sizes: Library size of each sample; defaults to the total
            counts. Must be positive.
        norm_factors: Normalization factor of each sample, multiplied into the
            library sizes; defaults to ones. Must be positive.

    Returns:
        Float64 array of shape ``(n_samples, n_features)``.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: If an argument is out of range, or a sample has library
            size zero.
        OverflowError: If a library size, or a count per million as edgeR forms
            it, exceeds the float64 range.
    """
    y = _as_counts(counts)
    sizes = _effective_library_sizes(y, library_sizes, norm_factors)
    try:
        with np.errstate(over="raise"):
            return _cpm(y, sizes)
    except FloatingPointError:
        with np.errstate(over="ignore"):
            sample, feature = np.argwhere(~np.isfinite(_cpm(y, sizes)))[0]
        raise OverflowError(
            f"the count per million of sample {sample}, feature {feature} exceeds the "
            "float64 range when formed as edgeR forms it, 1e6 * count / library size"
        ) from None


def log_cpm(
    counts: npt.ArrayLike,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    prior_count: float = 2.0,
) -> FloatArray:
    """Return log2 counts per million, as edgeR's ``cpm(log=TRUE)`` computes them.

    The prior count is scaled by library size, ``c_j = prior_count * N_j / mean(N)``
    with N the effective library sizes. It is added to every count of sample j
    and twice to its library size (edgeR's ``addPriorCount``), giving
    ``log2((y + c_j) / (N_j + 2 c_j) * 1e6)``.

    Args:
        counts: Counts, samples x features.
        library_sizes: Library size of each sample; defaults to the total
            counts. Must be positive.
        norm_factors: Normalization factor of each sample, multiplied into the
            library sizes; defaults to ones. Must be positive.
        prior_count: Average count added to each observation, at least 0. With
            0, zero counts give minus infinity.

    Returns:
        Float64 array of shape ``(n_samples, n_features)``.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: If an argument is out of range, or a sample has library
            size zero.
        OverflowError: If a library size, or the sum of the library sizes,
            exceeds the float64 range.
    """
    y = _as_counts(counts)
    sizes = _effective_library_sizes(y, library_sizes, norm_factors)
    prior = _non_negative(prior_count, "prior_count")
    result: FloatArray = _log2_cpm(y, sizes, prior, math.log(1e6), math.log(2.0))
    return result


def filter_by_expression(
    counts: npt.ArrayLike,
    *,
    group: npt.ArrayLike | None = None,
    design: object | None = None,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    min_count: float = 10,
    min_total_count: float = 15,
    large_n: float = 10,
    min_prop: float = 0.7,
) -> npt.NDArray[np.bool_]:
    """Return which features to keep, by edgeR's ``filterByExpr`` rule.

    A feature is kept when its CPM reaches ``min_count / median(N) * 1e6`` in at
    least n samples and its total count over all samples is at least
    ``min_total_count``, where N are the effective library sizes (Chen, Lun and
    Smyth 2016). n is the smallest group size or, with a design, the smallest
    inverse leverage ``1 / max(h)``, where h is the diagonal of the hat matrix
    of the design with an intercept column added, as R's ``hat()`` adds it
    (found by probing); the two agree for a one-way design. When n exceeds
    ``large_n`` it is relaxed to ``large_n + (n - large_n) * min_prop``.
    Without group and design all samples form one group, as in edgeR, with a
    warning.

    Args:
        counts: Counts, samples x features.
        group: Group label of each sample; labels must not be missing.
        design: Design matrix, samples x coefficients, or any object with a
            ``matrix`` attribute holding one, such as a ``Design``. With the
            intercept column added, its rank is decided as R's ``qr()`` decides
            it. Only the fixed effects set n: the random effects of a ``Design``
            are ignored, as edgeR defines n for the model matrix of the fixed
            effects. Leaving them out can only raise n, and so keep fewer
            features. When a random effect separates groups whose expression may
            differ, such as the tissues of a variance partition, pass those
            groups as ``group`` instead.
        library_sizes: Library size of each sample; defaults to the total
            counts. Must be positive.
        norm_factors: Normalization factor of each sample, multiplied into the
            library sizes; defaults to ones. Must be positive.
        min_count: Count needed, at the median library size, in at least n
            samples. At least 0.
        min_total_count: Total count needed over all samples. At least 0.
        large_n: Number of samples per group considered large. At least 0.
        min_prop: Fraction of the samples beyond ``large_n`` in which a feature
            must reach the cutoff, in [0, 1].

    Returns:
        Boolean array of shape ``(n_features,)``, True for features to keep.

    Raises:
        TypeError: If an argument has the wrong type, or is a numpy masked array.
        ValueError: If an argument is out of range, both group and design are
            given (edgeR silently ignores the design then), or a sample has
            library size zero.
        OverflowError: If a library size, or the total count of a feature,
            exceeds the float64 range.
    """
    y = _as_counts(counts)
    n_samples = y.shape[0]
    sizes = _effective_library_sizes(y, library_sizes, norm_factors)
    min_count = _non_negative(min_count, "min_count")
    min_total_count = _non_negative(min_total_count, "min_total_count")
    large_n = _non_negative(large_n, "large_n")
    min_prop = _real(min_prop, "min_prop")
    if not 0.0 <= min_prop <= 1.0:
        raise ValueError(f"min_prop must be in [0, 1], got {min_prop}")
    if group is not None and design is not None:
        raise ValueError("pass group or design, not both")
    if group is not None:
        min_samples = float(_smallest_group_size(group, n_samples))
    elif design is not None:
        matrix = _design_matrix(design, n_samples)
        leverages = _leverages(np.column_stack([np.ones(n_samples), matrix]))
        min_samples = 1.0 / float(np.max(leverages))
    else:
        warnings.warn(
            "no group or design given, so all samples are treated as one group",
            UserWarning,
            stacklevel=2,
        )
        min_samples = float(n_samples)
    if min_samples > large_n:
        min_samples = large_n + (min_samples - large_n) * min_prop
    # The cutoff and the CPM are formed as edgeR forms them: a count exactly at
    # the cutoff is kept or dropped by the last bit (found by probing).
    cutoff = min_count / float(np.median(sizes)) * 1e6
    with np.errstate(over="ignore"):  # an infinite CPM still reaches the cutoff
        n_expressed = np.count_nonzero(_cpm(y, sizes) >= cutoff, axis=0)
    totals = _sums_in_range(y, 0)
    keep: npt.NDArray[np.bool_] = n_expressed >= min_samples - _FILTER_SLACK
    return keep & (totals >= min_total_count - _FILTER_SLACK)


# Argument checks


def _as_counts(counts: npt.ArrayLike) -> FloatArray:
    array = unmasked_array(counts, "counts")
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"counts must be a dense integer or float array, got dtype {array.dtype}"
        )
    if array.ndim != 2:
        raise ValueError(
            "counts must be a 2-D array of samples x features, "
            f"got {array.ndim} dimension(s)"
        )
    if 0 in array.shape:
        raise ValueError(
            "counts must have at least one sample and one feature, "
            f"got shape {array.shape}"
        )
    # Writeable, aligned C order: the compiled routines then compile once.
    values = np.require(array, np.float64, ["C", "A", "W"])
    bad = np.argwhere(~(np.isfinite(values) & (values >= 0)))
    if bad.size:
        i, j = bad[0]
        raise ValueError(
            "counts must be finite and non-negative; "
            f"sample {i}, feature {j} is {values[i, j]}"
        )
    return values


def _positive_vector(values: npt.ArrayLike, name: str, n_samples: int) -> FloatArray:
    array = unmasked_array(values, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    if array.ndim != 1 or array.shape[0] != n_samples:
        raise ValueError(
            f"{name} must have one value per sample ({n_samples}), "
            f"got shape {array.shape}"
        )
    vector = np.require(array, np.float64, ["C", "A", "W"])
    bad = np.flatnonzero(~(np.isfinite(vector) & (vector > 0)))
    if bad.size:
        raise ValueError(
            f"{name} must be finite and positive; sample {bad[0]} has {vector[bad[0]]}"
        )
    return vector


def _library_sizes_of(
    counts: FloatArray, library_sizes: npt.ArrayLike | None
) -> FloatArray:
    if library_sizes is not None:
        return _positive_vector(library_sizes, "library_sizes", counts.shape[0])
    sizes = _sums_in_range(counts, 1)
    empty = np.flatnonzero(sizes == 0)
    if empty.size:
        raise ValueError(
            f"sample(s) {empty.tolist()} have no counts, so their library size is "
            "zero; remove them"
        )
    return sizes


def _effective_library_sizes(
    counts: FloatArray,
    library_sizes: npt.ArrayLike | None,
    norm_factors: npt.ArrayLike | None,
) -> FloatArray:
    sizes = _library_sizes_of(counts, library_sizes)
    if norm_factors is None:
        return sizes
    factors = _positive_vector(norm_factors, "norm_factors", counts.shape[0])
    with np.errstate(over="ignore", under="ignore"):
        sizes = sizes * factors
    if not np.all(np.isfinite(sizes) & (sizes > 0)):
        raise ValueError("library_sizes * norm_factors must be finite and positive")
    return sizes


def _real(value: object, name: str, *, finite: bool = True) -> float:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if math.isnan(number) or (finite and math.isinf(number)):
        expected = "finite" if finite else "a number"
        raise ValueError(f"{name} must be {expected}, got {number}")
    return number


def _non_negative(value: object, name: str) -> float:
    number = _real(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number


def _fraction(value: object, name: str) -> float:
    number = _real(value, name)
    if not 0.0 <= number < 0.5:
        raise ValueError(f"{name} must be in [0, 0.5), got {number}")
    return number


def _sample_index(value: object, n_samples: int) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Integral):
        raise TypeError(
            f"reference must be an integer sample index, got {type(value).__name__}"
        )
    index = int(value)
    if not 0 <= index < n_samples:
        raise ValueError(f"reference must be in [0, {n_samples}), got {index}")
    return index


def _smallest_group_size(group: npt.ArrayLike, n_samples: int) -> int:
    labels = unmasked_array(group, "group")
    if labels.ndim != 1 or labels.shape[0] != n_samples:
        raise ValueError(
            f"group must have one label per sample ({n_samples}), "
            f"got shape {labels.shape}"
        )
    values = labels.tolist()
    for i, value in enumerate(values):
        if value is None or value != value:
            raise ValueError(
                f"group labels must not be missing; sample {i} has {value}"
            )
    return min(Counter(values).values())


def _design_matrix(design: object, n_samples: int) -> FloatArray:
    if hasattr(design, "matrix"):
        matrix = unmasked_array(design.matrix, "design.matrix")
    else:
        matrix = unmasked_array(design, "design")
    if matrix.dtype.kind not in "iufb":
        raise TypeError(f"design must be a numeric matrix, got dtype {matrix.dtype}")
    if matrix.ndim != 2 or matrix.shape[0] != n_samples or matrix.shape[1] == 0:
        raise ValueError(
            f"design must have one row per sample ({n_samples}) and at least one "
            f"column, got shape {matrix.shape}"
        )
    values = matrix.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("design must be finite")
    return values


# Shared pieces


def _sums_in_range(counts: FloatArray, axis: int) -> FloatArray:
    """Return ``accurate_sums(counts, axis)``; a sum beyond float64 raises OverflowError."""
    sums: FloatArray = accurate_sums(counts, axis)
    beyond = np.flatnonzero(~np.isfinite(sums))
    if beyond.size:
        unit = "sample" if axis == 1 else "feature"
        raise OverflowError(
            f"the counts of {unit} {beyond[0]} sum beyond the float64 range (about "
            "1.8e308)"
        )
    return sums


def _expressed(counts: FloatArray) -> FloatArray:
    """The features with a positive count in some sample, in C order."""
    return np.ascontiguousarray(counts[:, (counts > 0).any(axis=0)])


def _cpm(counts: FloatArray, sizes: FloatArray) -> FloatArray:
    # edgeR multiplies by 1e6 first, then divides (found by probing).
    result: FloatArray = (1e6 * counts) / sizes[:, None]
    return result


def _to_geometric_mean_one(factors: FloatArray) -> FloatArray:
    logs = np.array([math.log(f) for f in factors])
    result: FloatArray = factors / math.exp(accurate_mean(logs))
    return result


def _row_quantiles(matrix: FloatArray, p: float) -> FloatArray:
    """R's type 7 quantile of each row, with R's floating-point operations.

    R places the quantile at ``1 + (n - 1) p``, returns the lower order
    statistic when the two neighbours are equal, and otherwise
    ``(1 - h) x[lo] + h x[hi]`` (found by probing; the algebraically equal
    forms of the documentation and of numpy differ in the last bit).
    """
    n = matrix.shape[1]
    position = 1 + (n - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    h = position - lo
    part = np.partition(matrix, sorted({lo - 1, hi - 1}), axis=1)
    below = part[:, lo - 1]
    above = part[:, hi - 1]
    result: FloatArray = np.where(below == above, below, (1 - h) * below + h * above)
    return result


def _tmm_reference(counts: FloatArray, sizes: FloatArray) -> int:
    # The documented rule, with the upper quartile of the counts divided by the
    # library size and R's mean (found by probing: these operations decide
    # between samples at equal distance from the mean).
    quartiles = _row_quantiles(counts, 0.75) / sizes
    if np.median(quartiles) < _TINY_UPPER_QUARTILE:
        return int(np.argmax(accurate_sums(np.sqrt(counts), 1)))
    return int(np.argmin(np.abs(quartiles - accurate_mean(quartiles))))


def _leverages(design: FloatArray) -> FloatArray:
    """Diagonal of the hat matrix, with the rank decided as R's ``qr()`` does."""
    return column_space(design).leverages()


# Compiled routines


@jit
def _average_ranks(values: FloatArray, n: int, ranks: FloatArray) -> None:
    """Write the average ranks of values[:n] (from 1; ties share their mean rank)."""
    order = np.argsort(values[:n], kind="mergesort")
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1


@jit(error_model="numpy")
def _tmm_log2_factors(
    counts: FloatArray,
    sizes: FloatArray,
    reference: int,
    log_ratio_trim: float,
    sum_trim: float,
    weighted: bool,
    a_cutoff: float,
) -> FloatArray:
    """log2 TMM factor of each sample against the reference, before scaling.

    M and A are formed as edgeR forms them, which reproduces its ties, and so
    its trimming, bit for bit; algebraically equal forms such as
    log2(p_k) - log2(p_r) do not (found by probing). With IEEE division, a
    degenerate weight gives infinities and NaN as in R, and a factor that is
    not finite becomes 1, as in edgeR.
    """
    n_samples, n_features = counts.shape
    m = np.empty(n_features)
    a = np.empty(n_features)
    v = np.empty(n_features)
    rank_m = np.empty(n_features)
    rank_a = np.empty(n_features)
    kept_m = np.empty(n_features)
    size_r = sizes[reference]
    out = np.zeros(n_samples)
    for k in range(n_samples):
        size_k = sizes[k]
        n = 0
        largest = 0.0
        for g in range(n_features):
            y_k = counts[k, g]
            y_r = counts[reference, g]
            if y_k <= 0.0 or y_r <= 0.0:
                continue
            p_k = y_k / size_k
            p_r = y_r / size_r
            log_ratio = np.log2(p_k / p_r)
            average = (np.log2(p_k) + np.log2(p_r)) / 2.0
            # The A cutoff is strict (found by probing).
            if not (np.isfinite(log_ratio) and np.isfinite(average)):
                continue
            if not average > a_cutoff:
                continue
            m[n] = log_ratio
            a[n] = average
            v[n] = (size_k - y_k) / size_k / y_k + (size_r - y_r) / size_r / y_r
            largest = max(largest, abs(log_ratio))
            n += 1
        # Identical profiles are judged after the A cutoff (found by probing).
        if n == 0 or largest < _SAME_PROFILE:
            continue
        low_m = np.floor(n * log_ratio_trim) + 1.0
        high_m = n + 1.0 - low_m
        low_a = np.floor(n * sum_trim) + 1.0
        high_a = n + 1.0 - low_a
        _average_ranks(m, n, rank_m)
        _average_ranks(a, n, rank_a)
        top_high = 0.0
        top_low = 0.0
        bottom_high = 0.0
        bottom_low = 0.0
        kept = 0
        for i in range(n):
            if rank_m[i] < low_m or rank_m[i] > high_m:
                continue
            if rank_a[i] < low_a or rank_a[i] > high_a:
                continue
            kept_m[kept] = m[i]
            kept += 1
            if weighted:
                # sum(M / v) / sum(1 / v), each sum rounded once, reproduces
                # edgeR's last bit (found by probing).
                s, error = two_sum(top_high, m[i] / v[i])
                top_high = s
                top_low += error
                s, error = two_sum(bottom_high, 1.0 / v[i])
                bottom_high = s
                bottom_low += error
        if kept == 0:
            continue
        if weighted:
            value = (top_high + top_low) / (bottom_high + bottom_low)
        else:
            value = accurate_mean(kept_m[:kept])
        if np.isfinite(value):
            out[k] = value
    return out


@jit
def _median_sorted(values: FloatArray, n: int) -> float:
    """Median of values[:n] as R computes it; sorts values[:n] in place."""
    values[:n].sort()
    half = n // 2
    if n % 2 == 1:
        return float(values[half])
    return float((values[half - 1] + values[half]) / 2.0)


@jit
def _log_count_sums(
    counts: FloatArray,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int64]]:
    """Per feature, the sum of the logs of the positive counts, over samples.

    Returned as rounded sums and remainders, each accumulated in sample order,
    with the number of positive counts.
    """
    n_samples, n_features = counts.shape
    high = np.zeros(n_features)
    low = np.zeros(n_features)
    n_positive = np.zeros(n_features, dtype=np.int64)
    for k in range(n_samples):
        for g in range(n_features):
            y = counts[k, g]
            if y > 0.0:
                s, error = two_sum(high[g], np.log(y))
                high[g] = s
                low[g] += error
                n_positive[g] += 1
    for g in range(n_features):
        s, error = two_sum(high[g], low[g])
        high[g] = s
        low[g] = error
    return high, low, n_positive


@jit
def _rle_median_ratios(counts: FloatArray) -> FloatArray:
    """Per sample, the median over complete features of y / geometric mean.

    A feature is complete when its counts are positive in every sample. All
    values are NaN when no feature is complete.
    """
    n_samples, n_features = counts.shape
    high, low, n_positive = _log_count_sums(counts)
    geometric_means = np.empty(n_features)
    n_complete = 0
    for g in range(n_features):
        if n_positive[g] == n_samples:
            geometric_means[g] = np.exp(
                rounded_quotient(high[g], low[g], float(n_samples))
            )
            n_complete += 1
    out = np.full(n_samples, np.nan)
    if n_complete == 0:
        return out
    ratios = np.empty(n_complete)
    for k in range(n_samples):
        i = 0
        for g in range(n_features):
            if n_positive[g] == n_samples:
                ratios[i] = counts[k, g] / geometric_means[g]
                i += 1
        out[k] = _median_sorted(ratios, n_complete)
    return out


@jit
def _median_log_ratios(counts: FloatArray, positive_only: bool) -> FloatArray:
    """Per sample, DESeq2's median log ratio of counts to geometric means.

    ``ratio``: features with a zero count are skipped, and the log geometric
    mean is R's mean of the logs. ``poscounts``: the log geometric mean is the
    sum of the logs of the positive counts divided by the number of samples,
    and a sample's log ratios, over its positive counts, are taken against the
    log of its exponential, which reproduces DESeq2's last bit (found by
    probing). A sample without any usable feature gets NaN.
    """
    n_samples, n_features = counts.shape
    high, low, n_positive = _log_count_sums(counts)
    log_means = np.full(n_features, -np.inf)
    for g in range(n_features):
        if positive_only and n_positive[g] > 0:
            log_means[g] = np.log(np.exp(high[g] / n_samples))
        elif n_positive[g] == n_samples:
            log_means[g] = rounded_quotient(high[g], low[g], float(n_samples))
    out = np.full(n_samples, np.nan)
    values = np.empty(n_features)
    for k in range(n_samples):
        n = 0
        for g in range(n_features):
            y = counts[k, g]
            if y > 0.0 and np.isfinite(log_means[g]):
                values[n] = np.log(y) - log_means[g]
                n += 1
        if n > 0:
            out[k] = _median_sorted(values, n)
    return out


@jit
def _log2_cpm(
    counts: FloatArray,
    sizes: FloatArray,
    prior_count: float,
    log_million: float,
    log_two: float,
) -> FloatArray:
    """edgeR's log2 CPM with a prior count scaled by library size.

    The prior uses the plain mean of the library sizes, and the result is
    formed from natural logarithms; both reproduce edgeR's last bit (found by
    probing).
    """
    n_samples, n_features = counts.shape
    total = 0.0
    for k in range(n_samples):
        total += sizes[k]
    mean_size = total / n_samples
    if not np.isfinite(mean_size):
        raise OverflowError("the library sizes sum beyond the float64 range")
    out = np.empty((n_samples, n_features))
    for k in range(n_samples):
        prior = prior_count * sizes[k] / mean_size
        offset = np.log(sizes[k] + 2.0 * prior)
        for g in range(n_features):
            out[k, g] = (np.log(counts[k, g] + prior) - offset + log_million) / log_two
    return out
