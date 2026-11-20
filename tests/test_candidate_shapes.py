"""Tests for candidate shapes of a dose response.

Reference values in ``tests/data/dose_contrasts/dosefinding.json`` come from
DoseFinding 1.4-2 (``getResp(Mods(...))``; ``guesst`` for the exponential root, which
it finds by a search, and for the beta model's closed form), from the placement rule
computed in R from each ladder, and from mpmath's roots at 40 digits, written by
``validation/dose_contrasts.py``. Other checks use values derived by hand: the series
of a nearly linear exponential shape, a beta shape's value at its peak and halfway,
and invariance to the dose's unit.
"""

import doctest
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import CandidateShapes, DoseAxis, Table, candidate_shapes

DATA = Path(__file__).parent / "data" / "dose_contrasts"
HALF_LOG = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")


@pytest.fixture(scope="module")
def dosefinding() -> dict[str, Any]:
    with open(DATA / "dosefinding.json") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


def numbers(values: Any) -> np.ndarray:
    return np.array(
        [np.nan if v is None else v for v in np.ravel(np.array(values, dtype=object))],
        dtype=np.float64,
    ).reshape(np.shape(np.array(values, dtype=object)))


def axis_of(record: dict[str, Any]) -> DoseAxis:
    return DoseAxis(
        "dose",
        unit="ug/kg",
        metric="administered",
        scale=record["scale"],
        vehicle=record["vehicle"],
        offset=record["offset"],
    )


def spread_case(case: dict[str, Any]) -> CandidateShapes:
    counts = case["counts"]
    return CandidateShapes.spread(
        axis_of(case["axis"]),
        {"dose": case["doses"]},
        n_emax=counts["emax"],
        n_sigmoid_emax=counts["sigmoid_emax"],
        n_exponential=counts["exponential"],
        n_beta=counts["beta"],
    )


def test_means_match_dosefinding(dosefinding: dict[str, Any]) -> None:
    for case in dosefinding["means"]:
        shapes = CandidateShapes(**case["shapes"])
        reference = numbers(case["reference"])
        ours = shapes.means(case["doses"])
        assert ours.shape == reference.shape
        scale = np.where(reference == 0.0, 1.0, np.abs(reference))
        # DoseFinding's exp(d / delta) - 1 loses digits where d / delta is small:
        # 4e-11 at delta = 1000 D and the lowest dose.
        assert np.max(np.abs(ours - reference) / scale) < 1e-10, case["label"]


def test_spread_matches_the_rule_and_guesst(dosefinding: dict[str, Any]) -> None:
    roots = iter(dosefinding["exponential_roots"])
    for case in dosefinding["spread"]:
        shapes = spread_case(case)
        reference = case["reference"]
        counts = case["counts"]
        assert len(shapes.emax) == counts["emax"]
        assert len(shapes.sigmoid_emax) == counts["sigmoid_emax"]
        assert len(shapes.exponential) == counts["exponential"]
        assert len(shapes.beta) == counts["beta"]
        assert_allclose(shapes.emax, numbers(reference["emax"]).ravel(), rtol=1e-12)
        assert_allclose(
            [ed50 for ed50, _ in shapes.sigmoid_emax],
            numbers(reference["sigmoid_ed50"]).ravel(),
            rtol=1e-12,
        )
        assert all(slope == 3.0 for _, slope in shapes.sigmoid_emax)
        exact = [next(roots) for _ in range(counts["exponential"])]
        assert_allclose(shapes.exponential, exact, rtol=1e-13)
        # guesst finds the root by a search, to about 1e-6.
        assert_allclose(
            shapes.exponential, numbers(reference["exponential"]).ravel(), rtol=1e-5
        )
        if counts["beta"]:
            assert_allclose(
                np.ravel(shapes.beta),
                np.ravel([numbers(pair) for pair in reference["beta"]]),
                rtol=1e-12,
            )
            assert shapes.beta_scale == pytest.approx(
                reference["beta_scale"], rel=1e-15
            )
        else:
            assert shapes.beta_scale is None


def test_spread_by_hand() -> None:
    """The half-log ladder of the TCDD liver series, 0.01 to 30 ug/kg."""
    shapes = CandidateShapes.spread(AXIS, {"dose": HALF_LOG})
    low, high = math.log10(0.01), math.log10(30.0)
    assert_allclose(
        shapes.emax,
        [10 ** (low + (high - low) / 3), 10 ** (low + 2 * (high - low) / 3)],
        rtol=1e-14,
    )
    assert_allclose(shapes.emax, [0.144225, 2.080084], rtol=1e-6)
    ((ed50, slope),) = shapes.sigmoid_emax
    assert ed50 == pytest.approx(math.sqrt(0.3), rel=1e-14)
    assert slope == 3.0
    # The exponential shape reaches half its change at the top dose at 0.75 D.
    (delta,) = shapes.exponential
    assert delta == pytest.approx(12.30763447, rel=1e-9)
    assert shapes.means([0.0, 22.5, 30.0])[3].tolist() == pytest.approx(
        [0.0, 0.5, 1.0], abs=1e-14
    )
    # The beta shape peaks at sqrt(0.3) and is half its peak halfway to 0.01 on log10.
    (delta1, delta2), scale = shapes.beta[0], shapes.beta_scale
    assert scale == pytest.approx(36.0, rel=1e-15)
    peak = scale * delta1 / (delta1 + delta2)
    assert peak == pytest.approx(math.sqrt(0.3), rel=1e-13)
    half = 10 ** ((low + math.log10(peak)) / 2)
    assert half == pytest.approx(0.07400828, rel=1e-7)

    def beta(d: float) -> float:
        return (d / scale) ** delta1 * (1 - d / scale) ** delta2

    assert beta(half) / beta(peak) == pytest.approx(0.5, rel=1e-13)
    assert shapes.means([0.0, half, peak, 30.0])[4][1:3].tolist() == pytest.approx(
        [0.5, 1.0], rel=1e-13
    )


def test_spread_follows_the_dose_s_unit() -> None:
    """Doses in ng/kg instead of ug/kg: every dose parameter x 1000, the beta model's
    exponents and every mean unchanged."""
    shapes = CandidateShapes.spread(AXIS, {"dose": HALF_LOG})
    scaled = CandidateShapes.spread(AXIS, {"dose": [1000 * d for d in HALF_LOG]})
    assert_allclose(scaled.emax, np.multiply(shapes.emax, 1000), rtol=1e-13)
    assert_allclose(
        scaled.exponential, np.multiply(shapes.exponential, 1000), rtol=1e-13
    )
    assert_allclose(scaled.beta, shapes.beta, rtol=1e-12)
    assert scaled.beta_scale == pytest.approx(1000 * shapes.beta_scale, rel=1e-15)
    assert_allclose(
        scaled.means([1000 * d for d in HALF_LOG]),
        shapes.means(HALF_LOG),
        rtol=1e-11,
        atol=1e-300,
    )


@pytest.mark.parametrize(
    ("placement", "expected"),
    [
        # log10(d + offset) from 0.01 + 0.001 to 30 + 0.001
        (
            DoseAxis(
                "dose",
                unit="ug/kg",
                metric="administered",
                vehicle="on_axis",
                offset=0.001,
            ),
            [
                10
                ** (
                    math.log10(0.011) + j * (math.log10(30.001) - math.log10(0.011)) / 3
                )
                - 0.001
                for j in (1, 2)
            ],
        ),
        # The dose itself, from 0.01 to 30.
        (
            DoseAxis("dose", unit="ug/kg", metric="administered", scale="linear"),
            [0.01 + j * (30.0 - 0.01) / 3 for j in (1, 2)],
        ),
    ],
)
def test_spread_places_shapes_along_the_axis(
    placement: DoseAxis, expected: list
) -> None:
    shapes = CandidateShapes.spread(placement, {"dose": HALF_LOG})
    assert_allclose(shapes.emax, expected, rtol=1e-13)


def test_a_nearly_linear_exponential_keeps_its_digits() -> None:
    """delta = 1e6 D: (exp(a x) - 1) / (exp(x) - 1) with x = D / delta and a = d / D,
    whose series a (1 + (a - 1) x / 2 + (2a - 1)(a - 1) x^2 / 12) is exact to 1e-19
    here, where exp(x) - 1 loses half the digits."""
    top = 30.0
    doses = np.array([0.0, 1e-3, 0.3, 7.5, 15.0, 22.5, 30.0])
    x = 1e-6
    a = doses / top
    series = a * (1 + (a - 1) * x / 2 + (2 * a - 1) * (a - 1) * x**2 / 12)
    means = CandidateShapes(exponential=[top / x]).means(doses)[0]
    assert_allclose(means, series, rtol=2e-16, atol=0)


def test_a_steep_exponential_neither_overflows_nor_warns() -> None:
    """delta = D / 1000: exp(D / delta) = exp(1000) overflows a double, and
    DoseFinding's means are NaN at the top dose; CytePool's are the limit."""
    doses = [0.0, 0.05, 0.2, 2.0, 3.0, 49.5, 50.0]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        means = CandidateShapes(exponential=[0.05]).means(doses)[0]
    assert means[-1] == 1.0
    assert means[0] == 0.0
    # exp((d - D) / delta) (1 - exp(-d / delta)) / (1 - exp(-D / delta)) at d = 49.5.
    assert means[-2] == pytest.approx(math.exp(-10.0) * -math.expm1(-990.0), rel=1e-14)
    assert np.all(means[1:5] < 1e-300)


def test_beta_is_scaled_at_its_peak_as_dosefinding_scales_it() -> None:
    """A beta shape peaking at 90.9, above the top dose of 10: scaled to 1 at its peak,
    so its largest mean at the doses is 0.138, as DoseFinding printed (build probe)."""
    shapes = CandidateShapes(beta=[(1.0, 0.1)], beta_scale=100.0)
    means = shapes.means([0.0, 0.01, 0.1, 1.0, 10.0])[0]
    dosefinding = [
        0.0,
        0.00013980657953042118,
        0.00139793990580001763,
        0.01396675365459004725,
        0.13834268635841220907,
    ]
    assert_allclose(means, dosefinding, rtol=1e-13)
    peak = 100 / 1.1
    by_hand = 0.1 * 0.9**0.1 / (peak / 100 * (1 - peak / 100) ** 0.1)
    assert means[-1] == pytest.approx(by_hand, rel=1e-13)


def test_a_beta_shape_reaches_zero_at_its_scale() -> None:
    """DoseFinding accepts a scale equal to the top dose, where the shape is 0."""
    means = CandidateShapes(beta=[(1.0, 1.0)], beta_scale=10.0).means([0.0, 1.0, 10.0])
    assert_allclose(means[0], [0.0, 0.36, 0.0], rtol=1e-14, atol=0)


def test_names_and_the_record() -> None:
    shapes = CandidateShapes(
        emax=[0.1, 2.0],
        sigmoid_emax=[(0.5, 3.0)],
        exponential=np.array([4.0, 9.0]),
        beta=[[1.0, 2.0]],
        beta_scale=12,
    )
    assert shapes.names == (
        "emax_1",
        "emax_2",
        "sigmoid_emax",
        "exponential_1",
        "exponential_2",
        "beta",
    )
    assert shapes.emax == (0.1, 2.0)
    assert shapes.beta == ((1.0, 2.0),)
    assert isinstance(shapes.beta_scale, float)
    record = json.loads(json.dumps(shapes.to_dict()))
    assert CandidateShapes(**record) == shapes
    assert pickle.loads(pickle.dumps(shapes)) == shapes
    assert shapes.means([0.0, 1.0, 10.0]).shape == (6, 3)
    with pytest.raises(AttributeError):
        shapes.emax = (1.0,)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({}, ValueError, "no candidate shapes"),
        ({"emax": []}, ValueError, "no candidate shapes"),
        ({"emax": 2.0}, TypeError, r"emax=\[0.5\] for one shape"),
        ({"emax": "2"}, TypeError, "one entry per shape"),
        (
            {"sigmoid_emax": (0.5, 3.0)},
            TypeError,
            r"sigmoid_emax=\[\(ed50, hill_slope\)\]",
        ),
        ({"sigmoid_emax": [(0.5, 3.0, 1.0)]}, TypeError, r"\(ED50, Hill slope\) pairs"),
        (
            {"beta": [(1.0,)], "beta_scale": 3.0},
            TypeError,
            r"beta=\[\(delta1, delta2\)\]",
        ),
        (
            {"emax": [0.0]},
            ValueError,
            "ED50 of emax shape 1 must be finite and above 0",
        ),
        ({"emax": [1.0, -2.0]}, ValueError, "emax shape 2"),
        ({"exponential": [np.inf]}, ValueError, "delta of exponential shape 1"),
        ({"sigmoid_emax": [(1.0, np.nan)]}, ValueError, "Hill slope of sigmoid_emax"),
        ({"emax": [True]}, TypeError, "real number"),
        ({"beta": [(1.0, 2.0)]}, ValueError, "need beta_scale.*of at least the top"),
        ({"emax": [1.0], "beta_scale": 3.0}, ValueError, "there are none"),
        ({"beta": [(1.0, 2.0)], "beta_scale": 0.0}, ValueError, "beta_scale"),
        ({"beta": [(1.0, 2.0)], "beta_scale": "3"}, TypeError, "beta_scale"),
    ],
)
def test_shapes_are_refused_with_a_fix(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        CandidateShapes(**arguments)


def test_means_refuse_doses_they_cannot_place() -> None:
    shapes = CandidateShapes(emax=[1.0], beta=[(1.0, 2.0)], beta_scale=12.0)
    with pytest.raises(ValueError, match="at least the top dose"):
        shapes.means([0.0, 1.0, 13.0])
    with pytest.raises(ValueError, match="a dose above 0"):
        shapes.means([0.0, 0.0])
    with pytest.raises(ValueError, match="finite and at least 0"):
        shapes.means([0.0, -1.0, 2.0])
    with pytest.raises(ValueError, match="finite and at least 0"):
        shapes.means([0.0, np.nan, 2.0])
    with pytest.raises(TypeError, match="real numbers"):
        shapes.means(["0", "1"])


def test_spread_refuses_what_it_cannot_place() -> None:
    with pytest.raises(ValueError, match="fewer than two doses above 0"):
        CandidateShapes.spread(AXIS, {"dose": [0.0, 0.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="all 0"):
        CandidateShapes.spread(
            AXIS,
            {"dose": HALF_LOG},
            n_emax=0,
            n_sigmoid_emax=0,
            n_exponential=0,
            n_beta=0,
        )
    with pytest.raises(ValueError, match="n_beta must be 0 or more"):
        CandidateShapes.spread(AXIS, {"dose": HALF_LOG}, n_beta=-1)
    with pytest.raises(TypeError, match="n_emax must be an integer"):
        CandidateShapes.spread(AXIS, {"dose": HALF_LOG}, n_emax=2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="axis must be a DoseAxis"):
        CandidateShapes.spread("dose", {"dose": HALF_LOG})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dose column 'dose' is not a column"):
        CandidateShapes.spread(AXIS, {"concentration": HALF_LOG})
    with pytest.raises(ValueError, match="differ by less than 1e-8"):
        CandidateShapes.spread(AXIS, {"dose": [0.0, 0.1, 0.1 * (1 + 1e-12), 1.0]})


def test_spread_reads_any_table() -> None:
    shapes = CandidateShapes.spread(AXIS, {"dose": HALF_LOG})
    table = Table({"sample_id": [f"s{i}" for i in range(9)], "dose": HALF_LOG})
    assert CandidateShapes.spread(AXIS, table) == shapes
    assert CandidateShapes.spread(AXIS, {"dose": np.repeat(HALF_LOG, 3)}) == shapes


def test_the_examples_in_the_docstrings_run() -> None:
    assert doctest.testmod(candidate_shapes, raise_on_error=False).failed == 0


def test_no_shape_is_identical() -> None:
    """The defaults give five distinct directions on a ladder of eight treated doses."""
    shapes = CandidateShapes.spread(AXIS, {"dose": HALF_LOG})
    means = shapes.means(HALF_LOG)
    assert np.linalg.matrix_rank(means - means.mean(axis=1, keepdims=True)) == 5
    assert_array_equal(means[:, 0], 0.0)


def test_many_exponential_shapes_place_the_steepest_near_the_top() -> None:
    """1,000 exponential shapes: the steepest reaches half its change at the top dose
    at 0.9995 D, where exp(D / delta) exceeds a double during the search."""
    shapes = CandidateShapes.spread(
        AXIS,
        {"dose": [0.0, 1.0, 10.0]},
        n_emax=0,
        n_sigmoid_emax=0,
        n_exponential=1000,
        n_beta=0,
    )
    steepest = CandidateShapes(exponential=[shapes.exponential[-1]])
    half = (0.5 + 1000 / 2002) * 10.0
    assert steepest.means([0.0, half, 10.0])[0].tolist() == pytest.approx(
        [0.0, 0.5, 1.0], abs=1e-12
    )
    assert np.all(np.diff(shapes.exponential) < 0)


def test_spread_refuses_a_cube() -> None:
    from cytepool import simulate_dose_series

    cube, _ = simulate_dose_series([0, 1, 10], replicates=2, n_genes=20, rng=1)
    with pytest.raises(TypeError, match="pass cube.samples"):
        CandidateShapes.spread(AXIS, cube)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="samples must be a Table"):
        CandidateShapes.spread(AXIS, [0.0, 1.0, 10.0])  # type: ignore[arg-type]
