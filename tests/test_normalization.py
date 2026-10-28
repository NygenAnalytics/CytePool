"""Tests for cytepool.normalization.

Reference values come from edgeR 4.4.0 and DESeq2 1.46.0 run as black boxes by
validation/normalization.py --write-fixtures. The other tests check values
derived by hand and invariants that any correct implementation keeps.
"""

import json
import math
import platform
from pathlib import Path

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cytepool.normalization as nz
from cytepool._summation import accurate_sums

DATA = Path(__file__).parent / "data" / "normalization"
# R's values are reproduced bit for bit with the GNU C library. The tolerance
# leaves room for other math libraries, while a different trimmed set or
# reference sample moves a factor by far more (1e-6 or more on these inputs).
RTOL = 1e-12
# The fixtures were computed by R with the GNU C library on x86-64; there the
# logarithms and exponentials are the same, and so are the results, bit for bit,
# unless numba uses Intel's SVML library for them.
SAME_MATH_LIBRARY = (
    platform.system() == "Linux"
    and platform.machine() == "x86_64"
    and platform.libc_ver()[0] == "glibc"
    and not numba.config.USING_SVML
)


def _load_fixtures() -> tuple[dict, list[dict]]:
    index = json.loads((DATA / "calls.json").read_text())
    with np.load(DATA / "normalization.npz") as arrays:
        inputs, outputs = arrays["inputs"], arrays["outputs"]

    def take(flat: np.ndarray, entry: list) -> np.ndarray:
        start, shape = entry
        return flat[start : start + math.prod(shape)].reshape(shape)

    cases = {
        name: {v: take(inputs, entry) for v, entry in variables.items()}
        for name, variables in index["cases"].items()
    }
    calls = [
        dict(call, expected=take(outputs, call["output"])) for call in index["calls"]
    ]
    return cases, calls


CASES, CALLS = _load_fixtures()
# calls.json names the function log2_cpm and the argument sample_groups of
# filter_by_expression as they were named when validation/normalization.py wrote it.
RENAMED_FUNCTIONS = {"log_cpm": "log2_cpm"}
RENAMED_ARGUMENTS = {"group": "sample_groups"}
CHECKS = sorted({call["check"] for call in CALLS})


def _run(call: dict) -> np.ndarray:
    case = CASES[call["case"]]
    kwargs = {}
    for key, value in call["kwargs"].items():
        if isinstance(value, str) and value.startswith("$"):
            value = case[value[1:]]
            if key in ("a_cutoff", "min_prop"):
                value = float(value[0])
        kwargs[RENAMED_ARGUMENTS.get(key, key)] = value
    function = RENAMED_FUNCTIONS.get(call["function"], call["function"])
    return np.asarray(getattr(nz, function)(case["Y"], **kwargs))


def test_fixtures_cover_every_function_and_rule() -> None:
    functions = {RENAMED_FUNCTIONS.get(c["function"], c["function"]) for c in CALLS}
    assert functions == set(nz.__all__) - {"library_sizes"}
    assert sum(check.startswith("edge case") for check in CHECKS) == 18
    assert sum(call["case"].startswith("small") for call in CALLS) == 177


@pytest.mark.filterwarnings("ignore:no sample_groups or design given")
@pytest.mark.parametrize("check", CHECKS)
def test_matches_edger_and_deseq2(check: str) -> None:
    failures = []
    for call in (c for c in CALLS if c["check"] == check):
        got, expected = _run(call), call["expected"]
        if call["kind"] == "filter":
            ok = np.array_equal(got, expected.astype(bool))
        elif call["kind"] == "logcpm":
            ok = np.allclose(got[:, : expected.shape[1]], expected, rtol=0, atol=RTOL)
        elif call["kind"] == "cpm":
            ok = np.allclose(got[:, : expected.shape[1]], expected, rtol=RTOL, atol=0)
        else:
            ok = np.allclose(got, expected, rtol=RTOL, atol=0)
        if not ok:
            failures.append(f"{call['case']}: {call['r_call']}")
    assert not failures, failures[:5]


@pytest.mark.filterwarnings("ignore:no sample_groups or design given")
@pytest.mark.skipif(
    not SAME_MATH_LIBRARY, reason="R's results came from glibc on x86-64"
)
def test_matches_edger_and_deseq2_to_the_last_bit() -> None:
    differ = []
    for call in CALLS:
        got, expected = _run(call).astype(np.float64), call["expected"]
        if call["kind"] in ("cpm", "logcpm"):
            got = got[:, : expected.shape[1]]
        if not np.array_equal(got, expected):
            differ.append(f"{call['case']}: {call['r_call']}")
    assert not differ, differ[:5]


# Values derived by hand


def test_tmm_keeps_the_ranks_inside_the_trimming_window() -> None:
    # Equal library sizes make M = log2(y_0 / y_1) exactly: -3, 1, 2, 5. With
    # n = 4 and a 30% trim the window of ranks is [2, 3], so M = 1 and M = 2
    # remain. Their variances 1/12 + 1/6 - 2/1000 and 1/40 + 1/10 - 2/1000 give
    # weights 125/31 and 1000/123, a weighted mean of 77375/46375 and, after
    # scaling to geometric mean one, factors 2^(+-77375/92750).
    counts = np.array([[1, 12, 40, 32], [8, 6, 10, 1]], dtype=np.float64)
    sizes = np.array([1000.0, 1000.0])
    half = 77375 / 92750
    assert_allclose(
        nz.tmm_factors(counts, library_sizes=sizes, reference=1, sum_trim=0.0),
        [2.0**half, 2.0**-half],
        rtol=1e-14,
    )
    # Unweighted, the mean of 1 and 2 is 1.5.
    assert_allclose(
        nz.tmm_factors(
            counts, library_sizes=sizes, reference=1, sum_trim=0.0, weighted=False
        ),
        [2.0**0.75, 2.0**-0.75],
        rtol=1e-14,
    )


def test_tmm_with_a_single_kept_feature_is_its_ratio() -> None:
    # M = -2, -1, 2, 3, 4; trimming 45% from each end of five keeps rank 3,
    # whose proportions differ fourfold: factors 4 and 1, scaled to 2 and 1/2.
    counts = np.array([[10, 20, 160, 320, 640], [40, 40, 40, 40, 40]], dtype=np.float64)
    got = nz.tmm_factors(
        counts,
        library_sizes=[1000, 1000],
        reference=1,
        log_ratio_trim=0.45,
        sum_trim=0.0,
    )
    assert_allclose(got, [2.0, 0.5], rtol=1e-15)


def test_tmm_gives_one_when_nothing_is_left_to_compare() -> None:
    # M ties (-1, -1, log2 1.5, log2 1.5) have average ranks 1.5 and 3.5,
    # outside the window [2, 3].
    ties = np.array([[1, 2, 3, 6], [1, 2, 1, 2]], dtype=np.float64)
    assert_array_equal(nz.tmm_factors(ties, reference=1), [1.0, 1.0])
    disjoint = np.array([[5, 0, 3, 0], [0, 7, 0, 2]], dtype=np.float64)
    assert_array_equal(nz.tmm_factors(disjoint), [1.0, 1.0])
    assert_array_equal(nz.tmm_factors([[5.0, 3.0, 0.0]]), [1.0])
    # Every log ratio is log2(1000 / 1000.0005), below 1e-6 in absolute value.
    same = np.array([[100, 200, 300, 400], [100, 200, 300, 400]], dtype=np.float64)
    assert_array_equal(
        nz.tmm_factors(same, library_sizes=[1000.0005, 1000]), [1.0, 1.0]
    )
    # No feature passes an infinite A cutoff.
    assert not np.all(nz.tmm_factors(GOOD) == 1.0)
    assert_array_equal(nz.tmm_factors(GOOD, a_cutoff=math.inf), [1.0, 1.0])


def test_tmm_reference_defaults_to_the_upper_quartile_closest_to_the_mean() -> None:
    rng = np.random.default_rng(3)
    counts = rng.poisson(40.0, (3, 50)).astype(np.float64)
    # Library sizes that put the upper quartile over the library size at 0.1,
    # 0.2 and 0.4; the mean is 0.233, closest to sample 1.
    quartiles = np.quantile(counts, 0.75, axis=1)
    sizes = quartiles / np.array([0.1, 0.2, 0.4])
    default = nz.tmm_factors(counts, library_sizes=sizes)
    assert_array_equal(
        default, nz.tmm_factors(counts, library_sizes=sizes, reference=1)
    )
    for other in (0, 2):
        alternative = nz.tmm_factors(counts, library_sizes=sizes, reference=other)
        assert np.max(np.abs(alternative / default - 1)) > 1e-6


def test_tmm_reference_of_sparse_counts_has_most_square_root_counts() -> None:
    # Two of three upper quartiles are zero, so their median is zero and edgeR
    # takes the sample with the largest sum of square-root counts: 9.10, 8.01
    # and 20.70 here, so sample 2.
    counts = np.array(
        [
            [0, 0, 0, 16, 0, 0, 0, 26, 0],
            [22, 0, 0, 0, 11, 0, 0, 0, 0],
            [0, 12, 8, 21, 0, 26, 11, 0, 2],
        ],
        dtype=np.float64,
    )
    default = nz.tmm_factors(counts)
    assert_array_equal(default, nz.tmm_factors(counts, reference=2))
    assert np.max(np.abs(nz.tmm_factors(counts, reference=0) / default - 1)) > 0.1


def test_tmm_a_cutoff_drops_features_at_low_abundance() -> None:
    rng = np.random.default_rng(4)
    counts = rng.poisson(rng.gamma(0.5, 60.0, 300), (2, 300)).astype(np.float64) + 1
    sizes = counts.sum(axis=1)
    a = (np.log2(counts[0] / sizes[0]) + np.log2(counts[1] / sizes[1])) / 2
    cutoff = float(np.median(a))
    above = a > cutoff
    assert 0 < above.sum() < above.size
    # Features below the cutoff count for nothing, as if they were removed.
    assert_allclose(
        nz.tmm_factors(counts, library_sizes=sizes, reference=0, a_cutoff=cutoff),
        nz.tmm_factors(counts[:, above], library_sizes=sizes, reference=0),
        rtol=1e-15,
    )


def test_upper_quartile_factors_by_hand() -> None:
    counts = np.array([[1, 2, 3, 0], [2, 4, 8, 0]], dtype=np.float64)
    # The all-zero feature is removed; medians 2 and 4 over library sizes 6 and 14.
    raw = np.array([2 / 6, 4 / 14])
    assert_allclose(
        nz.upper_quartile_factors(counts, p=0.5), raw / np.sqrt(raw.prod()), rtol=1e-15
    )


def test_rle_takes_the_median_of_ratios_and_deseq2_of_log_ratios() -> None:
    # Geometric means are 2 for both features. The ratios are (1/2, 1/2),
    # (1, 4) and (2, 1/2). edgeR's medians 1/2, 5/2 and 5/4 over library sizes
    # 2, 10 and 5 are equal, so the RLE factors are one. DESeq2's medians of
    # the log ratios give 1/2, 2 and 1.
    counts = np.array([[1, 1], [2, 8], [4, 1]], dtype=np.float64)
    assert_allclose(nz.rle_factors(counts), [1.0, 1.0, 1.0], rtol=1e-14)
    assert_allclose(nz.median_ratio_size_factors(counts), [0.5, 2.0, 1.0], rtol=1e-14)


def test_poscounts_by_hand() -> None:
    # Geometric means over both samples, zeros left out of the product: 2, 3, 2.
    # Sample 0's log ratios over its positive counts are -log 2 and log 2
    # (median 0); sample 1's are log 2, log 3 and -log 2 (median log 2). Size
    # factors 1 and 2, scaled to geometric mean one.
    counts = np.array([[1, 0, 4], [4, 9, 1]], dtype=np.float64)
    assert_allclose(
        nz.median_ratio_size_factors(counts, method="poscounts"),
        [2**-0.5, 2**0.5],
        rtol=1e-14,
    )


def test_median_ratio_of_proportional_samples() -> None:
    base = np.array([5, 10, 20, 7, 3], dtype=np.float64)
    counts = np.array([1.0, 2.0, 4.0])[:, None] * base
    for method in ("ratio", "poscounts"):
        got = nz.median_ratio_size_factors(counts, method=method)
        assert_allclose(got, [0.5, 1.0, 2.0], rtol=1e-14)
    for factors in (nz.tmm_factors, nz.upper_quartile_factors, nz.rle_factors):
        assert_allclose(factors(counts), [1.0, 1.0, 1.0], rtol=1e-14)


def test_cpm_and_log2_cpm_by_hand() -> None:
    counts = np.array([[1, 3], [2, 2]], dtype=np.float64)
    assert_array_equal(nz.cpm(counts), [[250000, 750000], [500000, 500000]])
    assert_array_equal(
        nz.cpm(counts, norm_factors=[2.0, 0.5]), [[125000, 375000], [1e6, 1e6]]
    )
    # Library sizes 4 and 4: the scaled prior is 2, added to each count and
    # twice to the library size.
    swapped = np.array([[0, 4], [4, 0]], dtype=np.float64)
    expected = np.log2([[2 / 8 * 1e6, 6 / 8 * 1e6], [6 / 8 * 1e6, 2 / 8 * 1e6]])
    assert_allclose(nz.log2_cpm(swapped), expected, rtol=1e-14)
    # Library sizes 4 and 12 scale the prior to 1 and 3.
    unequal = np.array([[0, 4], [6, 6]], dtype=np.float64)
    expected = np.log2([[1 / 6 * 1e6, 5 / 6 * 1e6], [9 / 18 * 1e6, 9 / 18 * 1e6]])
    assert_allclose(nz.log2_cpm(unequal, prior_count=2), expected, rtol=1e-14)
    # Without a prior, zero counts give minus infinity.
    million = math.log2(1e6)
    assert_allclose(
        nz.log2_cpm(swapped, prior_count=0),
        [[-math.inf, million], [million, -math.inf]],
        rtol=1e-14,
    )


def _equal_depth(rows: list[list[float]]) -> np.ndarray:
    """Samples x features with every library of exactly one million reads."""
    genes = np.array(rows, dtype=np.float64).T
    return np.column_stack([genes, 1e6 - genes.sum(axis=1)])


def test_filter_by_expression_follows_the_documented_rule() -> None:
    # Library sizes of 1e6 make CPM equal to the count and the cutoff 10.
    counts = _equal_depth(
        [
            [10, 10, 10, 0, 0, 0],  # 3 samples at the cutoff: kept
            [10, 10, 0, 0, 0, 0],  # 2 samples: dropped
            [9, 9, 9, 9, 9, 9],  # below the cutoff everywhere: dropped
            [10, 0, 0, 10, 10, 0],  # 3 samples, across groups: kept
            [0, 0, 0, 0, 0, 0],  # all zero: dropped
        ]
    )
    balanced = ["a", "a", "a", "b", "b", "b"]
    expected = [True, False, False, True, False, True]
    assert_array_equal(
        nz.filter_by_expression(counts, sample_groups=balanced), expected
    )
    one_way = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    assert_array_equal(nz.filter_by_expression(counts, design=one_way), expected)
    # A group of two needs only two samples.
    unbalanced = ["a", "a", "b", "b", "b", "b"]
    assert_array_equal(
        nz.filter_by_expression(counts, sample_groups=unbalanced),
        [True, True, False, True, False, True],
    )


def test_filter_by_expression_total_count_is_inclusive() -> None:
    counts = _equal_depth([[1, 1, 1, 0, 0, 0]])
    group = [0, 0, 0, 1, 1, 1]
    for total, keep in ((3, True), (3.5, False)):
        got = nz.filter_by_expression(
            counts, sample_groups=group, min_count=1, min_total_count=total
        )
        assert_array_equal(got, [keep, True])


def test_filter_by_expression_relaxes_large_groups() -> None:
    # Groups of 15: 10 + (15 - 10) * 0.7 = 13.5 samples, so 14 are needed.
    counts = _equal_depth([[10] * 14 + [0] * 16, [10] * 13 + [0] * 17])
    group = [0] * 15 + [1] * 15
    assert_array_equal(
        nz.filter_by_expression(counts, sample_groups=group), [True, False, True]
    )
    design = np.column_stack([np.ones(30), np.repeat([0.0, 1.0], 15)])
    assert_array_equal(
        nz.filter_by_expression(counts, design=design), [True, False, True]
    )
    assert_array_equal(
        nz.filter_by_expression(counts, sample_groups=group, min_prop=1.0),
        [False, False, True],
    )
    assert_array_equal(
        nz.filter_by_expression(counts, sample_groups=group, large_n=15),
        [False, False, True],
    )


def test_filter_by_expression_design_rank_and_matrix_attribute() -> None:
    counts = _equal_depth([[10, 10, 10, 0, 0, 0], [10, 10, 0, 0, 0, 0]])
    one_way = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    # A copy of a column adds nothing to the column space, as in R's qr().
    repeated = np.column_stack([one_way, one_way[:, 1]])

    class Design:
        matrix = repeated

    expected = nz.filter_by_expression(counts, design=one_way)
    assert_array_equal(nz.filter_by_expression(counts, design=repeated), expected)
    assert_array_equal(nz.filter_by_expression(counts, design=Design()), expected)


def test_filter_by_expression_reads_only_the_fixed_effects_of_a_design() -> None:
    # As documented, only the fixed effects set n: a Design with a random donor
    # intercept filters as edgeR's filterByExpr does with the model matrix of its
    # fixed effects. Read as fixed, the donor columns would give every sample
    # leverage 1, and so n = 1, a laxer filter.
    from cytepool import Design

    case = CASES["ties"]
    (call,) = [
        c for c in CALLS if c["case"] == "ties" and c["kwargs"] == {"design": "$X"}
    ]
    labels = np.array(["a", "b", "c"])[case["g"].astype(int)]
    samples = {"g": labels, "x": case["X"][:, 3], "donor": ["d1", "d2", "d3"] * 2}
    design = Design(samples, "~ g + x", random="(1 | donor)")
    assert_array_equal(design.matrix, case["X"])
    kept = nz.filter_by_expression(case["Y"], design=design)
    assert_array_equal(kept, call["expected"].astype(bool))
    as_fixed = Design(samples, "~ g + x + donor")
    assert not np.array_equal(nz.filter_by_expression(case["Y"], design=as_fixed), kept)


def test_filter_by_expression_adds_an_intercept_to_the_design() -> None:
    # With the intercept that edgeR adds, x = (1, 1, 1, 1, 1, 2) gives the last
    # sample leverage 1/6 + (5/6)^2 / (5/6) = 1, so one sample suffices. Without
    # it the leverages x^2 / 9 would require 9/4, so three samples.
    counts = _equal_depth([[0, 0, 0, 0, 0, 20], [20, 20, 0, 0, 0, 0]])
    x = np.array([[1, 1, 1, 1, 1, 2.0]]).T
    assert_array_equal(nz.filter_by_expression(counts, design=x), [True, True, True])
    # An all-zero design leaves the intercept alone: all six samples.
    one_group = _equal_depth([[20, 20, 20, 20, 20, 0], [20] * 6])
    assert_array_equal(
        nz.filter_by_expression(one_group, design=np.zeros((6, 1))), [False, True, True]
    )


def test_filter_by_expression_without_group_or_design_uses_one_group() -> None:
    counts = _equal_depth([[10, 10, 10, 10, 10, 0], [10] * 6])
    with pytest.warns(UserWarning, match="one group"):
        got = nz.filter_by_expression(counts)
    assert_array_equal(got, nz.filter_by_expression(counts, sample_groups=[0] * 6))
    assert_array_equal(got, [False, True, True])


# Invariants


@pytest.fixture(scope="module")
def counts() -> np.ndarray:
    rng = np.random.default_rng(11)
    means = rng.gamma(0.8, 50.0, 300) * rng.uniform(0.5, 2.0, (6, 1))
    return rng.poisson(means).astype(np.float64)


def test_library_sizes_are_rounded_once(counts: np.ndarray) -> None:
    assert_array_equal(nz.library_sizes(counts), [sum(map(int, row)) for row in counts])
    # A large value followed by many small ones, which a plain sum loses.
    row = np.array([1e16] + [1.0] * 10 + [0.1] * 7)
    rng = np.random.default_rng(5)
    noisy = rng.gamma(0.5, 30.0, (3, 1000))
    for values in (row[None, :], noisy):
        assert_array_equal(nz.library_sizes(values), [math.fsum(v) for v in values])


def test_scaling_a_sample_keeps_its_normalized_counts(counts: np.ndarray) -> None:
    positive = counts + 1
    scaled = positive.copy()
    scaled[2] *= 3
    assert_array_equal(nz.cpm(scaled), nz.cpm(positive))
    # The effective library size of the scaled sample triples with its
    # counts, so its factor and every other factor stay the same.
    for factors in (
        lambda y: nz.tmm_factors(y, weighted=False),
        nz.upper_quartile_factors,
        nz.rle_factors,
    ):
        assert_allclose(factors(scaled), factors(positive), rtol=1e-14)
    # Precision weights depend on depth, so weighted TMM keeps its factors
    # when every sample is scaled.
    assert_allclose(nz.tmm_factors(positive * 7), nz.tmm_factors(positive), rtol=1e-14)
    # DESeq2's size factors replace the library size: the scaled sample's
    # factor triples relative to the others.
    for method in ("ratio", "poscounts"):
        ratio = nz.median_ratio_size_factors(
            scaled, method=method
        ) / nz.median_ratio_size_factors(positive, method=method)
        assert_allclose(ratio / ratio[0], [1, 1, 3, 1, 1, 1], rtol=1e-14)


def test_factors_follow_the_order_of_samples(counts: np.ndarray) -> None:
    order = np.array([3, 0, 5, 1, 4, 2])
    group = np.array([0, 0, 0, 1, 1, 1])
    for factors in (
        nz.tmm_factors,
        lambda y: nz.tmm_factors(y, weighted=False),
        nz.upper_quartile_factors,
        lambda y: nz.rle_factors(y + 1),
        lambda y: nz.median_ratio_size_factors(y + 1),
        lambda y: nz.median_ratio_size_factors(y, method="poscounts"),
    ):
        assert_allclose(factors(counts[order]), factors(counts)[order], rtol=1e-14)
    assert_array_equal(
        nz.filter_by_expression(counts[order], sample_groups=group[order]),
        nz.filter_by_expression(counts, sample_groups=group),
    )


def test_edger_factors_have_geometric_mean_one(counts: np.ndarray) -> None:
    for factors in (
        nz.tmm_factors(counts),
        nz.upper_quartile_factors(counts),
        nz.rle_factors(counts + 1),
        nz.median_ratio_size_factors(counts, method="poscounts"),
    ):
        assert math.isclose(math.exp(np.mean(np.log(factors))), 1.0, rel_tol=1e-15)


def test_all_zero_features_change_nothing(counts: np.ndarray) -> None:
    padded = np.column_stack([np.zeros((6, 3)), counts, np.zeros((6, 2))])
    for factors in (
        nz.tmm_factors,
        nz.upper_quartile_factors,
        lambda y: nz.rle_factors(y + (y.sum(axis=0) > 0)),
        nz.median_ratio_size_factors,
        lambda y: nz.median_ratio_size_factors(y, method="poscounts"),
    ):
        assert_array_equal(factors(padded), factors(counts))
    keep = nz.filter_by_expression(padded, sample_groups=[0, 0, 0, 1, 1, 1])
    assert not keep[:3].any() and not keep[-2:].any()
    assert_array_equal(
        keep[3:-2], nz.filter_by_expression(counts, sample_groups=[0, 0, 0, 1, 1, 1])
    )


def test_inputs_are_not_modified(counts: np.ndarray) -> None:
    before = counts.copy()
    nz.tmm_factors(counts)
    nz.rle_factors(counts + 1)
    nz.median_ratio_size_factors(counts, method="poscounts")
    assert_array_equal(counts, before)


def test_memory_layout_changes_nothing_and_compiles_nothing(counts: np.ndarray) -> None:
    expected = nz.tmm_factors(counts), nz.log2_cpm(counts), nz.rle_factors(counts + 1)
    kernels = (nz._tmm_log2_factors, nz._log2_cpm, nz._rle_median_ratios, accurate_sums)
    compiled = [len(kernel.signatures) for kernel in kernels]
    read_only = counts.copy()
    read_only.flags.writeable = False
    for layout in (
        read_only,
        np.asfortranarray(counts),
        np.repeat(counts, 2, 1)[:, ::2],
    ):
        assert_array_equal(nz.tmm_factors(layout), expected[0])
        assert_array_equal(nz.log2_cpm(layout), expected[1])
        assert_array_equal(nz.rle_factors(layout + 1), expected[2])
    assert [len(kernel.signatures) for kernel in kernels] == compiled


def test_integer_and_float_counts_agree(counts: np.ndarray) -> None:
    as_int = counts.astype(np.int32)
    assert_array_equal(nz.tmm_factors(as_int), nz.tmm_factors(counts))
    assert_array_equal(nz.cpm(as_int), nz.cpm(counts))


def test_integers_beyond_int64_become_float64_without_wrapping_around() -> None:
    # 2**64 - 2048, 2048 and their total 2**64 are all float64 values. Through
    # int64, 2**64 - 2048 would wrap around to -2048.
    counts = np.array([[2**64 - 2048, 2048], [1, 3]], dtype=np.uint64)
    assert nz.library_sizes(counts).tolist() == [2.0**64, 4.0]
    assert nz.cpm(counts)[0].tolist() == [1e6 * (1 - 2.0**-53), 1e6 * 2.0**-53]
    # Above 2**53, float64 rounds to the nearest whole number it holds.
    rounded = np.array([[2**53 + 1]], dtype=np.int64)
    assert nz.library_sizes(rounded).tolist() == [2.0**53]


def test_sums_beyond_the_float64_range_raise() -> None:
    largest = np.finfo(np.float64).max
    # Two halves of the largest float64 sum to it exactly.
    halves = np.array([[largest / 2, largest / 2]])
    assert nz.library_sizes(halves).tolist() == [largest]
    # Sample 0 sums to 2e308; its library size would be NaN, and its factors too.
    huge = np.array([[1e308, 1e308, 1.0], [1.0, 2.0, 3.0]])
    for compute in (
        nz.library_sizes,
        nz.tmm_factors,
        nz.upper_quartile_factors,
        nz.rle_factors,
        nz.cpm,
        nz.log2_cpm,
        lambda y: nz.filter_by_expression(y, sample_groups=[0, 1]),
    ):
        with pytest.raises(OverflowError, match="counts of sample 0 sum beyond"):
            compute(huge)
    # Each sample sums to 1e308 at most, but feature 0 sums to 2e308 over them.
    tall = np.array([[1e308, 1.0], [1e308, 1.0], [1.0, 1.0]])
    with pytest.raises(OverflowError, match="the counts of feature 0 sum beyond"):
        nz.filter_by_expression(tall, sample_groups=[0, 0, 1])
    # edgeR forms 1e6 * count before dividing by the library size: 1e309 here.
    with pytest.raises(
        OverflowError, match="count per million of sample 0, feature 0 exceeds"
    ):
        nz.cpm(np.array([[1e303, 1.0], [1.0, 1.0]]))
    # log-CPM scales the prior count by the mean library size, here 2e308 / 2.
    with pytest.raises(OverflowError, match="the library sizes sum beyond the float64"):
        nz.log2_cpm(np.ones((2, 2)), library_sizes=[1e308, 1e308])


# Errors


GOOD = np.array([[5.0, 3.0, 8.0], [2.0, 6.0, 4.0]])


@pytest.mark.parametrize(
    ("counts", "error", "match"),
    [
        (np.array([["a", "b"]]), TypeError, "dense integer or float array"),
        (np.array([[True, False]]), TypeError, "dense integer or float array"),
        (np.array([1.0, 2.0]), ValueError, "2-D array of samples x features"),
        (np.zeros((2, 0)), ValueError, "at least one sample and one feature"),
        (np.array([[1.0, np.nan]]), ValueError, "sample 0, feature 1 is nan"),
        (
            np.array([[1.0, 2.0], [np.inf, 1.0]]),
            ValueError,
            "sample 1, feature 0 is inf",
        ),
        (np.array([[1.0, -2.0]]), ValueError, "finite and non-negative"),
    ],
)
def test_counts_are_checked(counts: np.ndarray, error: type, match: str) -> None:
    for function in (
        nz.library_sizes,
        nz.tmm_factors,
        nz.upper_quartile_factors,
        nz.rle_factors,
        nz.median_ratio_size_factors,
        nz.cpm,
        nz.log2_cpm,
        lambda y: nz.filter_by_expression(y, sample_groups=[0] * y.shape[0]),
    ):
        with pytest.raises(error, match=match):
            function(counts)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (dict(library_sizes=[1.0]), ValueError, r"one value per sample \(2\)"),
        (dict(library_sizes=[1.0, 0.0]), ValueError, "sample 1 has 0.0"),
        (dict(library_sizes=[np.nan, 1.0]), ValueError, "finite and positive"),
        (dict(library_sizes=["a", "b"]), TypeError, "library_sizes must be numeric"),
        (
            dict(norm_factors=[1.0, -1.0]),
            ValueError,
            "norm_factors must be finite and positive",
        ),
        (
            dict(norm_factors=[1e308, 1.0], library_sizes=[1e308, 1.0]),
            ValueError,
            r"library_sizes \* norm_factors must be finite",
        ),
    ],
)
def test_library_sizes_and_factors_are_checked(
    kwargs: dict, error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        nz.cpm(GOOD, **kwargs)


def test_empty_samples_are_rejected() -> None:
    empty = np.array([[5.0, 3.0], [0.0, 0.0]])
    for function in (nz.tmm_factors, nz.upper_quartile_factors, nz.rle_factors, nz.cpm):
        with pytest.raises(ValueError, match=r"sample\(s\) \[1\] have no counts"):
            function(empty)
    with pytest.raises(ValueError, match="every geometric mean is zero"):
        nz.median_ratio_size_factors(empty)
    with pytest.raises(ValueError, match=r"sample\(s\) \[1\] have no positive count"):
        nz.median_ratio_size_factors(empty, method="poscounts")
    # Library sizes given for an empty sample are taken as they are.
    assert_allclose(nz.cpm(empty, library_sizes=[8.0, 4.0])[1], [0.0, 0.0])


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (dict(reference=2), ValueError, r"reference must be in \[0, 2\), got 2"),
        (dict(reference=-1), ValueError, "reference must be in"),
        (dict(reference=1.0), TypeError, "integer sample index"),
        (dict(reference=True), TypeError, "integer sample index"),
        (dict(log_ratio_trim=0.5), ValueError, r"log_ratio_trim must be in \[0, 0.5\)"),
        (dict(sum_trim=-0.1), ValueError, r"sum_trim must be in \[0, 0.5\)"),
        (dict(sum_trim=np.nan), ValueError, "sum_trim must be finite"),
        (dict(log_ratio_trim="0.3"), TypeError, "log_ratio_trim must be a real number"),
        (dict(weighted=1), TypeError, "weighted must be True or False"),
        (dict(a_cutoff=np.nan), ValueError, "a_cutoff must be a number"),
        (dict(a_cutoff=None), TypeError, "a_cutoff must be a real number"),
    ],
)
def test_tmm_arguments_are_checked(kwargs: dict, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        nz.tmm_factors(GOOD, **kwargs)


def test_undefined_factors_raise() -> None:
    mostly_zero = np.array([[5.0, 3.0, 8.0, 2.0, 6.0], [0.0, 0.0, 0.0, 0.0, 7.0]])
    with pytest.raises(
        ValueError, match=r"0.75 quantile of the counts is zero in sample\(s\) \[1\]"
    ):
        nz.upper_quartile_factors(mostly_zero)
    with pytest.raises(ValueError, match="every count is zero"):
        nz.upper_quartile_factors(np.zeros((2, 3)), library_sizes=[1.0, 1.0])
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        nz.upper_quartile_factors(GOOD, p=1.5)
    with pytest.raises(TypeError, match="p must be a real number"):
        nz.upper_quartile_factors(GOOD, p="0.75")
    no_complete = np.array([[5.0, 3.0, 0.0], [0.0, 0.0, 7.0]])
    with pytest.raises(
        ValueError, match="no feature has positive counts in every sample"
    ):
        nz.rle_factors(no_complete)
    with pytest.raises(ValueError, match='use method="poscounts"'):
        nz.median_ratio_size_factors(no_complete)
    with pytest.raises(ValueError, match='method must be "ratio" or "poscounts"'):
        nz.median_ratio_size_factors(GOOD, method="iterate")
    with pytest.raises(ValueError, match="prior_count must be non-negative"):
        nz.log2_cpm(GOOD, prior_count=-1)
    with pytest.raises(ValueError, match="prior_count must be finite"):
        nz.log2_cpm(GOOD, prior_count=np.inf)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            dict(sample_groups=[0, 1], design=np.ones((2, 1))),
            ValueError,
            "sample_groups or design, not both",
        ),
        (dict(sample_groups=[0, 1, 2]), ValueError, r"one label per sample \(2\)"),
        (dict(sample_groups=[0, None]), ValueError, "sample 1 has None"),
        (
            dict(sample_groups=[0.0, np.nan]),
            ValueError,
            "the labels of sample_groups must not be missing",
        ),
        (dict(design=np.ones((3, 1))), ValueError, r"one row per sample \(2\)"),
        (dict(design=np.array([[1.0], [np.inf]])), ValueError, "design must be finite"),
        (
            dict(design=np.array([["a"], ["b"]])),
            TypeError,
            "design must be a numeric matrix",
        ),
        (
            dict(sample_groups=[0, 1], min_prop=1.5),
            ValueError,
            r"min_prop must be in \[0, 1\]",
        ),
        (
            dict(sample_groups=[0, 1], min_count=-1),
            ValueError,
            "min_count must be non-negative",
        ),
        (
            dict(sample_groups=[0, 1], large_n=-1),
            ValueError,
            "large_n must be non-negative",
        ),
        (
            dict(sample_groups=[0, 1], min_total_count=np.nan),
            ValueError,
            "min_total_count must be finite",
        ),
    ],
)
def test_filter_arguments_are_checked(kwargs: dict, error: type, match: str) -> None:
    with pytest.raises(error, match=match):
        nz.filter_by_expression(GOOD, **kwargs)
