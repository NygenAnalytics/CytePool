"""BenchmarkResponse: the two kinds of benchmark response, their checks, labels and
round trips."""

import math

import numpy as np
import pytest

from cytepool import BenchmarkResponse


def test_the_constructors_give_the_two_kinds() -> None:
    sd = BenchmarkResponse.sd(1.349)
    absolute = BenchmarkResponse.absolute(0.5)
    assert (sd.kind, sd.value) == ("sd", 1.349)
    assert (absolute.kind, absolute.value) == ("absolute", 0.5)
    assert BenchmarkResponse("sd", 1.349) == sd
    assert sd != BenchmarkResponse.absolute(1.349)


@pytest.mark.parametrize(
    ("bmr", "label"),
    [
        (BenchmarkResponse.sd(1.0), "1 SD"),
        (BenchmarkResponse.sd(1.349), "1.349 SD"),
        (BenchmarkResponse.absolute(0.5), "change of 0.5"),
        (BenchmarkResponse.absolute(2.0), "change of 2"),
        # Six significant digits: a change computed from a fold change prints as a
        # value typed in with those digits.
        (BenchmarkResponse.absolute(math.log2(1.5)), "change of 0.584963"),
        (BenchmarkResponse.absolute(0.584963), "change of 0.584963"),
        (BenchmarkResponse.sd(1e-7), "1e-07 SD"),
        (BenchmarkResponse.absolute(123456.7), "change of 123457"),
    ],
)
def test_labels_have_six_significant_digits(bmr: BenchmarkResponse, label: str) -> None:
    assert bmr.label == label
    number = label.removeprefix("change of ").removesuffix(" SD")
    assert float(number) == pytest.approx(bmr.value, rel=5e-6)


def test_round_trip_through_plain_values() -> None:
    for bmr in (BenchmarkResponse.sd(1.349), BenchmarkResponse.absolute(np.log2(1.1))):
        stored = bmr.to_dict()
        assert stored == {"kind": bmr.kind, "value": bmr.value}
        assert type(stored["value"]) is float
        assert BenchmarkResponse(**stored) == bmr


def test_numpy_numbers_are_kept_as_floats() -> None:
    bmr = BenchmarkResponse.absolute(np.float32(0.5))
    assert type(bmr.value) is float and bmr.value == 0.5
    assert BenchmarkResponse.sd(np.int64(2)).value == 2.0


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_values_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="value must be positive and finite"):
        BenchmarkResponse.sd(value)
    with pytest.raises(ValueError, match="value must be positive and finite"):
        BenchmarkResponse.absolute(value)


@pytest.mark.parametrize(
    "value", [True, "1", None, [1.0], np.array([1.0]), np.ma.masked_array(1.0)]
)
def test_values_must_be_real_numbers(value: object) -> None:
    with pytest.raises(TypeError, match="value must be a real number"):
        BenchmarkResponse.sd(value)  # type: ignore[arg-type]


def test_kinds_are_checked() -> None:
    with pytest.raises(ValueError, match="kind must be one of 'sd', 'absolute'"):
        BenchmarkResponse("relative", 0.1)
    with pytest.raises(TypeError, match="kind must be one of"):
        BenchmarkResponse(1, 0.1)  # type: ignore[arg-type]


def test_a_response_is_immutable_and_hashable() -> None:
    bmr = BenchmarkResponse.sd(1.0)
    with pytest.raises(AttributeError):
        bmr.value = 2.0  # type: ignore[misc]
    assert len({bmr, BenchmarkResponse.sd(1.0), BenchmarkResponse.absolute(1.0)}) == 2
