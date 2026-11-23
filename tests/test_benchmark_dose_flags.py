"""The checks of BMDS, NTP, ETAP and EFSA of benchmark doses, at each threshold's edge.

The inequalities are written here from their sources: pybmds 25.2's default rules of
BMDS's model recommendation (its ``default.json`` and ``checks.py``, MIT), NTP 2018 pp.
8 to 9, US EPA 2024 Sec. 3.4.5.3 and EFSA 2022 Sec. 2.6.5."""

import math

import numpy as np
import pytest

from cytepool.stats._benchmark_dose_flags import FlagInputs, failed_checks

LOWEST, TOP = 0.1, 30.0


def inputs(**rows: list[float]) -> FlagInputs:
    """One row per value of the longest list; unspecified values are those of a
    gene that passes every check."""
    n = max(len(v) for v in rows.values())
    passing = {
        "bmd": 1.0,
        "bmdl": 0.5,
        "bmdu": 2.0,
        "fit_p_value": 0.5,
        "residual_near_bmd": 0.5,
        "residual_at_vehicle": 0.5,
        "vehicle_sd_ratio": 1.0,
        "constant_variance_p_value": 0.5,
    }
    arrays = {
        key: np.asarray(rows.get(key, [value] * n), dtype=np.float64)
        for key, value in passing.items()
    }
    return FlagInputs(bounds=True, lowest_dose=LOWEST, top_dose=TOP, **arrays)


def flags_of(flag_set: str, **rows: list[float]) -> list[str]:
    return failed_checks(flag_set, inputs(**rows))


def test_a_gene_that_passes_every_set() -> None:
    for flag_set in ("bmds_2026", "ntp_2018", "etap_2024", "efsa_2022"):
        assert flags_of(flag_set, bmd=[1.0]) == [""]


def test_etap_ratio_is_at_least_20() -> None:
    # ETAP accepts BMD/BMDL < 20: a ratio of exactly 20 fails.
    got = flags_of("etap_2024", bmd=[19.99, 20.0, 20.01], bmdl=[1.0, 1.0, 1.0])
    assert got == ["", "BMD / BMDL >= 20", "BMD / BMDL >= 20"]


def test_efsa_ratios_are_strict() -> None:
    # EFSA's indicators: BMD/BMDL > 20 and BMDU/BMDL > 50.
    got = flags_of("efsa_2022", bmd=[20.0, 20.01], bmdl=[1.0, 1.0], bmdu=[50.0, 50.01])
    assert got == ["", "BMD / BMDL > 20; BMDU / BMDL > 50"]


def test_ntp_bmdu_over_bmdl_is_at_least_40() -> None:
    got = flags_of("ntp_2018", bmdl=[0.5, 0.5], bmdu=[19.99, 20.0])
    assert got == ["", "BMDU / BMDL >= 40"]


def test_fit_p_values_at_the_edge() -> None:
    # NTP and ETAP accept p > 0.1; BMDS flags p < 0.1.
    p = [0.0999, 0.1, 0.1001]
    assert flags_of("ntp_2018", fit_p_value=p) == [
        "fit p-value <= 0.1",
        "fit p-value <= 0.1",
        "",
    ]
    assert flags_of("etap_2024", fit_p_value=p)[1] == "fit p-value <= 0.1"
    assert flags_of("bmds_2026", fit_p_value=p) == [
        "fit p-value < 0.1 (questionable)",
        "",
        "",
    ]


def test_bmd_against_the_highest_dose() -> None:
    near_top = {"bmdl": [TOP / 2, TOP / 2], "bmdu": [2 * TOP, 2 * TOP]}
    got = flags_of("ntp_2018", bmd=[TOP * (1 - 1e-12), TOP], **near_top)
    assert got == ["", "BMD not below the highest dose"]
    got = flags_of("bmds_2026", bmd=[TOP, TOP * (1 + 1e-12)], **near_top)
    assert got == ["", "BMD / highest dose > 1 (warning)"]


def test_low_bmds() -> None:
    # pybmds: lowest dose / BMD > 3 warns and > 10 is questionable; NTP keeps a BMD
    # more than 10-fold below the lowest dose with a flag; EFSA's indicator is a BMD
    # 10 times lower than the lowest dose.
    bmd = [LOWEST / 3, LOWEST / 3.0001, LOWEST / 10, LOWEST / 10.0001]
    bmdl = [b / 2 for b in bmd]
    got = flags_of("bmds_2026", bmd=bmd, bmdl=bmdl)
    assert got[0] == "lowest dose / BMDL > 3 (warning)"
    assert got[1] == "lowest dose / BMD > 3 (warning); lowest dose / BMDL > 3 (warning)"
    assert "lowest dose / BMDL > 10 (questionable)" in got[2]
    assert "lowest dose / BMD > 10 (questionable)" not in got[2]
    assert "lowest dose / BMD > 10 (questionable)" in got[3]
    ntp = flags_of("ntp_2018", bmd=bmd, bmdl=bmdl)
    assert [("10-fold below" in f) for f in ntp] == [False, False, False, True]
    efsa = flags_of("efsa_2022", bmd=bmd, bmdl=bmdl)
    assert [("10 times below" in f) for f in efsa] == [False, False, False, True]


def test_bmds_ratio_and_residual_thresholds() -> None:
    got = flags_of(
        "bmds_2026",
        bmd=[3.0, 3.0001, 20.0001],
        bmdl=[1.0, 1.0, 1.0],
        bmdu=[30.0, 30.0, 30.0],
        residual_near_bmd=[2.0, -2.0001, 0.0],
        residual_at_vehicle=[-2.0, 2.0001, 0.0],
        vehicle_sd_ratio=[1.5, 1.5001, 1.0],
        constant_variance_p_value=[0.05, 0.0499, 0.5],
    )
    assert got[0] == ""
    assert got[1] == (
        "|scaled residual near the BMD| > 2 (questionable); constant variance test "
        "p-value < 0.05 (questionable); BMD / BMDL > 3 (warning); curve's vehicle SD / "
        "observed vehicle SD > 1.5 (warning); |scaled residual at the vehicle| > 2 "
        "(warning)"
    )
    assert got[2] == "BMD / BMDL > 20 (questionable); BMD / BMDL > 3 (warning)"


def test_bmds_edges_are_strict() -> None:
    # pybmds's default rules fail a ratio strictly above its threshold: BMD / BMDL
    # > 20 (questionable) and > 3 (warning), BMDL / highest dose > 1 (warning),
    # lowest dose / BMDL > 10 (questionable) and > 3 (warning).
    got = flags_of("bmds_2026", bmd=[20.0, 20.0001], bmdl=[1.0, 1.0], bmdu=[40.0, 40.0])
    assert got == [
        "BMD / BMDL > 3 (warning)",
        "BMD / BMDL > 20 (questionable); BMD / BMDL > 3 (warning)",
    ]
    got = flags_of(
        "bmds_2026",
        bmd=[2 * TOP, 2 * TOP],
        bmdl=[TOP, TOP * (1 + 1e-12)],
        bmdu=[4 * TOP, 4 * TOP],
    )
    assert got == [
        "BMD / highest dose > 1 (warning)",
        "BMD / highest dose > 1 (warning); BMDL / highest dose > 1 (warning)",
    ]
    # A BMD at the lowest dose and BMDLs at exactly a third and a tenth of it.
    bmdl = [LOWEST / 3, LOWEST / 3.0001, LOWEST / 10, LOWEST / 10.0001]
    got = flags_of("bmds_2026", bmd=[LOWEST] * 4, bmdl=bmdl, bmdu=[1.0] * 4)
    assert got == [
        "",
        "lowest dose / BMDL > 3 (warning); BMD / BMDL > 3 (warning)",
        "lowest dose / BMDL > 3 (warning); BMD / BMDL > 3 (warning)",
        "lowest dose / BMDL > 10 (questionable); lowest dose / BMDL > 3 (warning); "
        "BMD / BMDL > 3 (warning)",
    ]


def test_missing_values() -> None:
    nan = math.nan
    rows = {"bmd": [nan, 1.0, 1.0, 1.0], "bmdl": [nan, nan, 0.0, 0.5]}
    rows["bmdu"] = [nan, 2.0, 2.0, math.inf]
    assert flags_of("bmds_2026", **rows) == [
        "BMD not computed (unusable); BMDL not computed (unusable)",
        "BMDL not computed (unusable)",
        # pybmds counts a BMDL of 0 as not estimated, and skips its ratios
        "BMDL not computed (unusable)",
        "",
    ]
    # a BMDL of 0 or a BMDU of infinity widens BMDU / BMDL beyond any threshold
    assert flags_of("ntp_2018", **rows) == [
        "BMD, BMDL or BMDU not computed, or beyond the search",
        "BMD, BMDL or BMDU not computed, or beyond the search",
        "BMD, BMDL or BMDU not computed, or beyond the search; BMDU / BMDL >= 40",
        "BMD, BMDL or BMDU not computed, or beyond the search; BMDU / BMDL >= 40",
    ]
    assert flags_of("etap_2024", **rows) == [
        "BMD or BMDL not computed",
        "BMD or BMDL not computed",
        "BMD / BMDL >= 20",
        "",
    ]
    assert flags_of("efsa_2022", **rows) == [
        "",
        "",
        "BMD / BMDL > 20; BMDU / BMDL > 50",
        "BMDU / BMDL > 50",
    ]


def test_checks_whose_inputs_a_row_lacks_are_not_made() -> None:
    nan = math.nan
    got = flags_of(
        "bmds_2026",
        fit_p_value=[nan],
        residual_at_vehicle=[nan],
        vehicle_sd_ratio=[nan],
        constant_variance_p_value=[nan],
    )
    assert got == [""]
    without_bounds = FlagInputs(
        bmd=np.array([1.0]),
        bmdl=np.array([nan]),
        bmdu=np.array([nan]),
        bounds=False,
        lowest_dose=LOWEST,
        top_dose=TOP,
        fit_p_value=np.array([nan]),
        residual_near_bmd=np.array([0.0]),
        residual_at_vehicle=np.array([0.0]),
        vehicle_sd_ratio=np.array([nan]),
        constant_variance_p_value=np.array([nan]),
    )
    for flag_set in ("bmds_2026", "ntp_2018", "etap_2024", "efsa_2022"):
        assert failed_checks(flag_set, without_bounds) == [""]


@pytest.mark.parametrize(
    "flag_set", ["ntp_2018", "etap_2024", "efsa_2022", "bmds_2026"]
)
def test_rows_are_independent(flag_set: str) -> None:
    single = [flags_of(flag_set, bmd=[b], bmdl=[b / 30]) for b in (0.001, 1.0, 50.0)]
    joint = flags_of(
        flag_set, bmd=[0.001, 1.0, 50.0], bmdl=[0.001 / 30, 1 / 30, 50 / 30]
    )
    assert [s[0] for s in single] == joint


def test_a_mistyped_set_gets_a_suggestion() -> None:
    from cytepool.stats._benchmark_dose_flags import checked_flag_sets

    # A set's name without its year, or in capitals, starts a set's name.
    with pytest.raises(
        ValueError,
        match=r"unknown set of checks 'ntp'; .*\(did you mean 'ntp_2018'\?\)$",
    ):
        checked_flag_sets("ntp")
    with pytest.raises(ValueError, match=r"\(did you mean 'etap_2024'\?\)$"):
        checked_flag_sets(["ETAP_2024"])
    # Otherwise, close spellings.
    with pytest.raises(ValueError, match=r"\(did you mean 'efsa_2022'"):
        checked_flag_sets("efsa_2021")
    assert checked_flag_sets(["etap_2024", "ntp_2018"]) == ("etap_2024", "ntp_2018")
