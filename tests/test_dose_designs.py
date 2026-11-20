"""check_dose_design against scanpy, R and limma, ETAP's minimums, GSE208161, rules
derived by hand, and limma_voom's own choice of samples.

The fixtures come from validation/dose_design.py: scanpy's sums and pandas' tables of
a simulated TCDD-like series, R's qr() and limma's nonEstimable on 19 designs, ETAP's
minimums coded in R, and the cells and RNA counts of GSE208161.
"""

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import cytepool
from cytepool import (
    Cube,
    Design,
    DoseAxis,
    DoseDesignReport,
    Table,
    check_dose_design,
    limma_voom,
)

DATA = Path(__file__).resolve().parent / "data" / "dose_design"
CHAPTER = (
    Path(__file__).resolve().parents[1] / "docs/source/statistics/dose_response.md"
)
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
DOSE9 = [0, 0, 0, 1, 1, 10, 10, 30, 30]
# The doses of an ETAP design: half a log10 apart down from 300, the lowest a full
# log10 below the second lowest.
ETAP_LADDER = [0.03, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]


def samples_of(doses: list[float], **columns: Any) -> dict[str, Any]:
    ids = [f"S{i + 1}" for i in range(len(doses))]
    return {"sample_id": ids, "dose": np.asarray(doses, dtype=float), **columns}


def cube_of(
    samples: dict[str, Any],
    cells: list | None = None,
    totals: list | None = None,
    group: str = "cells",
) -> Cube:
    """A cube of one cell type and one feature: each sample's counts, 100 by
    default and none without cells, and its cells, 30 by default."""
    n = len(samples["sample_id"])
    counts = np.array(totals if totals is not None else [100] * n, dtype=np.int64)
    n_cells = None
    if cells is not None or totals is None:
        n_cells = np.array(cells if cells is not None else [30] * n)[:, None]
        counts[n_cells[:, 0] == 0] = 0
    return Cube(
        {"counts": counts[None, :, None]},
        samples=samples,
        groups={"group_id": [group]},
        features={"feature_id": ["all"]},
        n_cells=n_cells,
    )


def check(
    doses: list[float],
    formula: str = "~ 1",
    *,
    cells: list | None = None,
    random: str | None = None,
    **options: Any,
) -> DoseDesignReport:
    """The report of a design of one cell type in which every sample has 30 cells
    and 100 counts, unless ``cells`` says otherwise; options go to the check, or,
    for lists, into the sample table."""
    columns = {key: value for key, value in options.items() if isinstance(value, list)}
    settings = {key: value for key, value in options.items() if key not in columns}
    samples = samples_of(doses, **columns)
    design = Design(samples, formula, random=random)
    return check_dose_design(cube_of(samples, cells), design, dose=AXIS, **settings)


def one(report: DoseDesignReport, column: str) -> Any:
    """The value of a column of the summary of a report of one cell type."""
    (value,) = report.summary[column].tolist()
    return value


def dose_column(report: DoseDesignReport, column: str) -> list:
    return report.doses[column].tolist()


# Against the references.


def cube_from_fixture(case: dict[str, Any]) -> Cube:
    """The cube of one feature that holds a fixture case's cells and counts."""
    totals = np.array([c["totals"] for c in case["cell_types"]], dtype=np.int64)
    cells = [c["cells"] for c in case["cell_types"]]
    n_cells = None if cells[0] is None else np.array(cells, dtype=np.int64).T
    return Cube(
        {"counts": totals[:, :, None]},
        samples=case["samples"],
        groups={"group_id": [c["name"] for c in case["cell_types"]]},
        features={"feature_id": ["all"]},
        n_cells=n_cells,
    )


def test_counts_per_dose_group_match_scanpy() -> None:
    data = np.load(DATA / "scanpy_counts.npz")
    cell_types = data["cell_type"].tolist()
    samples = {
        "sample_id": data["sample_id"].tolist(),
        "dose": data["dose"],
        "lane": data["lane"].tolist(),
    }
    # scanpy's sums of every mouse and cell type, with the cells it counted.
    sums = data["sums"]
    cube = Cube(
        {"counts": sums},
        samples=samples,
        groups={"group_id": cell_types},
        features={"feature_id": [f"g{i}" for i in range(sums.shape[2])]},
        n_cells=data["n_cells"],
    )
    totals = sums.sum(axis=2)  # cell types x mice
    for k, min_counts in enumerate(data["min_counts"].tolist()):
        report = check_dose_design(
            cube, Design(samples, "~ 1"), dose=AXIS, min_counts=min_counts
        )
        table = report.samples
        position = {
            (t, m): i
            for i, (t, m) in enumerate(
                zip(data[f"setting{k}__cell_type"], data[f"setting{k}__mouse"])
            )
        }
        order = [
            position[t, m]
            for t, m in zip(table["group_id"].tolist(), table["sample_id"].tolist())
        ]
        assert table["used"].tolist() == data[f"setting{k}__used"][order].tolist()
        g = [cell_types.index(t) for t in table["group_id"].tolist()]
        i = [samples["sample_id"].index(m) for m in table["sample_id"].tolist()]
        assert table["n_cells"].tolist() == data["n_cells"][i, g].tolist()
        assert table["total_counts"].tolist() == totals[g, i].astype(float).tolist()
        doses = report.doses
        expected = {
            (t, float(d)): row
            for row, (t, d) in enumerate(
                zip(
                    data[f"setting{k}__group_cell_type"],
                    data[f"setting{k}__group_dose"],
                )
            )
        }
        rows = [
            expected[t, d]
            for t, d in zip(doses["group_id"].tolist(), doses["dose"].tolist())
        ]
        for column in (
            "n_samples_used",
            "n_samples_left_out",
            "n_cells_min",
            "total_counts_min",
        ):
            np.testing.assert_array_equal(
                doses[column], data[f"setting{k}__{column}"][rows], err_msg=column
            )
    # The second setting leaves samples out that the first keeps.
    assert data["setting1__used"].sum() < data["setting0__used"].sum()


def test_aliasing_matches_r() -> None:
    fixture = json.loads((DATA / "r_aliasing.json").read_text())
    assert fixture["versions"] == "R 4.4.3 (qr), limma 3.62.1 (nonEstimable)"
    for case in fixture["cases"]:
        report = check_dose_design(
            cube_from_fixture(case),
            Design(case["samples"], case["formula"]),
            dose=AXIS,
            batch=case["batch"],
            min_cells=case["min_cells"],
            min_counts=case["min_counts"],
        )
        for cell_type, found in zip(case["cell_types"], case["r"], strict=True):
            name, label = cell_type["name"], f"{case['name']}, {cell_type['name']}"
            doses = report.doses.filter(group_id=name)
            summary = report.summary.filter(group_id=name)
            used = report.samples.filter(group_id=name)["used"]
            assert used.tolist() == found["used"], label
            (df,) = summary["df_residual"].tolist()
            if found["df_residual"] is None:  # R's NA: no sample is used
                assert np.isnan(df), label
            else:
                assert df == found["df_residual"], label
            assert doses["estimable"].tolist() == found["estimable"], label
            if case["batch"] is not None:
                assert doses["confounded_with_batch"].tolist() == found["confounded"]
            # The statuses follow the rules coded again in R (self-coded).
            assert summary["status"].tolist() == [found["status"]], label


def test_etap_minimums_match_r() -> None:
    fixture = json.loads((DATA / "etap.json").read_text())
    for design in fixture["designs"]:
        axis = DoseAxis("dose", unit=design["unit"], metric="administered")
        report = check_dose_design(
            cube_from_fixture(design),
            Design(design["samples"], "~ 1"),
            dose=axis,
            rules="etap_2024",
            sex="sex",
            min_cells=design["min_cells"],
        )
        table = report.minimums
        assert table is not None
        ours = [
            {
                "group_id": table["group_id"][i],
                "sex": table["sex"][i],
                "minimum": table["minimum"][i],
                "observed": table["observed"][i],
                "met": bool(table["met"][i]),
            }
            for i in range(len(table))
        ]
        # The rows of ETAP's minimums coded in R (self-coded), in the same order.
        assert ours == design["r"], design["name"]


def test_gse208161_is_descriptive() -> None:
    # One participant per dose level, placebo included, each sampled at day 1 and
    # day 29: no dose has a second participant (Sonigra et al. 2022).
    data = np.load(DATA / "gse208161.npz")
    clusters = data["cluster"].tolist()
    samples = {
        "sample_id": data["sample_id"].tolist(),
        "patient": data["patient"].tolist(),
        "timepoint": data["timepoint"].tolist(),
        "dose": data["dose"],
    }
    rna = data["rna"]
    cube = Cube(
        {"counts": rna.T[:, :, None]},
        samples=samples,
        groups={"group_id": clusters},
        features={"feature_id": ["rna"]},
        n_cells=data["n_cells"],
    )
    millilitres = DoseAxis("dose", unit="mL", metric="administered")
    by_patient = check_dose_design(cube, Design(samples, "~ patient"), dose=millilitres)
    assert set(by_patient.summary["status"].tolist()) == {"descriptive"}
    # 13 clusters keep one residual degree of freedom, the placebo participant's
    # change from baseline, which a rule on residual df alone would call enough.
    assert sorted(by_patient.summary["df_residual"].tolist()) == [0.0] * 5 + [1.0] * 13
    late = [i for i, day in enumerate(samples["timepoint"]) if day == "day29"]
    day29 = {key: [values[i] for i in late] for key, values in samples.items()}
    report = check_dose_design(cube, Design(day29, "~ 1"), dose=millilitres)
    statuses = dict(zip(clusters, report.summary["status"].tolist(), strict=True))
    assert [c for c, s in statuses.items() if s == "refused"] == ["14", "17"]
    assert set(statuses.values()) == {"descriptive", "refused"}
    # The placebo participant's day-29 sample keeps 3 cells in those clusters.
    placebo = samples["sample_id"].index("patient19_day29")
    columns = [clusters.index("14"), clusters.index("17")]
    assert data["n_cells"][placebo, columns].tolist() == [3, 3]


def test_the_samples_are_those_limma_voom_fits() -> None:
    rng = np.random.default_rng(5)
    doses = [0, 0, 0, 1, 1, 1, 10, 10, 10]
    samples = samples_of(doses, group=["vehicle"] * 3 + ["treated"] * 6)
    cells = rng.integers(5, 40, (9, 2))
    counts = rng.poisson(20, (2, 9, 30)) * cells.T[:, :, None]
    cube = Cube(
        {"counts": counts.astype(np.int64)},
        samples=samples,
        groups={"group_id": ["a", "b"]},
        features={"feature_id": [f"g{i}" for i in range(30)]},
        n_cells=cells,
    )
    fit_design = Design(
        samples,
        "~ group",
        levels={"group": ["vehicle", "treated"]},
        contrasts={"treated": "grouptreated"},
    )
    for min_cells in (10, 20):
        report = check_dose_design(
            cube, Design(samples, "~ 1"), dose=AXIS, min_cells=min_cells
        )
        fitted = limma_voom(cube, fit_design, min_cells=min_cells)
        for column in ("group_id", "sample_id", "n_cells", "used", "reason"):
            assert report.samples[column].tolist() == fitted.samples[column].tolist()
    assert not report.samples["used"].all()


# Rules derived by hand.


def test_statuses_by_hand() -> None:
    # Every dose group replicated: 6 samples, an intercept and 2 indicators.
    report = check([0, 0, 1, 1, 10, 10])
    assert one(report, "status") == "inferential"
    assert one(report, "df_residual") == 3.0
    assert one(report, "n_dose_groups") == 3
    assert dose_column(report, "estimable") == [False, True, True]
    assert one(report, "reason") == ""
    # One sample per dose: the model fits every sample.
    report = check([0, 1, 10, 30])
    assert (one(report, "status"), one(report, "df_residual")) == ("descriptive", 0.0)
    assert "keeps a second sample" in one(report, "reason")
    # A replicated vehicle leaves residual df, but no dose has a replicate.
    report = check([0, 0, 0, 0, 1, 10, 30])
    assert (one(report, "status"), one(report, "df_residual")) == ("descriptive", 3.0)
    # A sample with exactly min_cells cells is used, as in limma_voom; the doses
    # table gives each group's fewest cells and counts among the samples used.
    samples = samples_of([0, 0, 1, 1])
    report = check_dose_design(
        cube_of(samples, [10, 9, 40, 25], [100, 80, 300, 200]),
        Design(samples, "~ 1"),
        dose=AXIS,
    )
    assert report.samples["used"].tolist() == [True, False, True, True]
    assert report.samples["reason"].tolist()[1] == "9 cells, fewer than min_cells (10)"
    assert dose_column(report, "n_cells_min") == [10.0, 25.0]
    assert dose_column(report, "total_counts_min") == [100.0, 200.0]
    assert dose_column(report, "n_samples_left_out") == [1, 0]
    # The top dose keeps one sample among replicated groups: kept, and flagged.
    report = check([0, 0, 1, 1, 10])
    assert one(report, "status") == "inferential"
    assert dose_column(report, "n_samples_used") == [2, 2, 1]
    # The vehicle and one dose, in duplicate.
    report = check([0, 0, 1, 1])
    assert (one(report, "status"), one(report, "df_residual")) == ("inferential", 2.0)
    # Two covariates use up the residual df of replicated dose groups: rank 4 on
    # 4 samples, and the dose-1 column outside the span of the intercept and both.
    report = check(
        [0, 0, 1, 1], "~ a + b", a=[0.1, 0.5, 0.2, 0.9], b=[1.0, 0.3, 0.7, 0.2]
    )
    assert (one(report, "status"), one(report, "df_residual")) == ("descriptive", 0.0)
    assert dose_column(report, "estimable") == [False, True]
    assert one(report, "reason") == (
        "no residual degrees of freedom: the model has rank 4 on the 4 samples left, "
        "so it fits them exactly"
    )
    # A design without an intercept has the residual df of one with it.
    assert one(check([0, 0, 0, 1, 1, 10, 10], "~ 0"), "df_residual") == 4.0
    assert one(check([0, 0, 0, 1, 1, 10, 10], "~ 1"), "df_residual") == 4.0
    # The same samples refused: the vehicle, or every treated sample, left out.
    report = check([0, 0, 1, 1, 10, 10], cells=[5, 5, 30, 30, 30, 30])
    assert one(report, "status") == "refused"
    assert one(report, "reason") == (
        "no vehicle sample is left (2 vehicle samples left out); compare the doses "
        "with each other with limma_voom"
    )
    assert one(report, "n_dose_groups") == 0
    report = check([0, 0, 1, 1, 10, 10], cells=[30, 30, 5, 5, 5, 5])
    assert (
        one(report, "reason")
        == "no treated sample is left (4 treated samples left out)"
    )
    assert (one(report, "n_dose_groups"), one(report, "n_dose_groups_dropped")) == (
        1,
        2,
    )
    report = check([0, 0, 1, 1], cells=[0, 0, 0, 0])
    assert one(report, "reason") == (
        "every one of the 4 samples was left out (fewer than 10 cells, or no counts)"
    )
    assert np.isnan(one(report, "df_residual"))
    assert np.isnan(report.doses["n_cells_min"]).all()
    assert np.isnan(report.doses["total_counts_min"]).all()
    # A bulk cube, which records no cells: samples without counts, or with fewer
    # than min_counts, are left out.
    samples = samples_of([0, 0, 1, 1])
    for min_counts, rules in ((0, "no counts"), (50, "fewer than 50 counts")):
        report = check_dose_design(
            cube_of(samples, totals=[0, 0, 0, 0]),
            Design(samples, "~ 1"),
            dose=AXIS,
            min_counts=min_counts,
        )
        assert one(report, "reason") == (
            f"every one of the 4 samples was left out ({rules})"
        )
        assert "n_cells_min" not in report.doses and "n_cells" not in report.samples
    # Only the vehicle in the design.
    report = check([0, 0, 0])
    assert one(report, "status") == "refused"
    assert one(report, "reason") == "the design has no treated sample, only the vehicle"
    # No sample of dose 0 in the design.
    report = check([1, 1, 10, 10])
    assert one(report, "status") == "refused"
    assert one(report, "reason").startswith("the design has no sample of dose 0")


def test_designs_sampled_before_and_after_dosing() -> None:
    # Four participants, each sampled before dosing (dose 0) and after, one per dose.
    doses = [0, 0, 0, 0, 0, 0.3, 1, 3]
    patient = ["P19", "P11", "P04", "P21"] * 2
    timepoint = ["day1"] * 4 + ["day29"] * 4
    # With the participant as a fixed term, only the placebo participant's change
    # from baseline is left to estimate the variance.
    report = check(doses, "~ patient", patient=patient)
    assert (one(report, "status"), one(report, "df_residual")) == ("descriptive", 1.0)
    report = check(doses, "~ patient + timepoint", patient=patient, timepoint=timepoint)
    assert (one(report, "status"), one(report, "df_residual")) == ("descriptive", 0.0)
    # As a random intercept the participant leaves residual df, and no dose has a
    # second participant either.
    report = check(doses, random="(1 | patient)", patient=patient)
    assert one(report, "status") == "descriptive"
    assert dose_column(report, "estimable") == [False, True, True, True]


def test_batch_confounding_by_hand() -> None:
    def confounded(lanes: list) -> list[float]:
        """The doses that the lanes confound, on the nine samples of DOSE9."""
        doses = check(DOSE9, batch="lane", lane=lanes).doses
        return doses["dose"][doses["confounded_with_batch"]].tolist()

    # Each dose in its own lane: every lane indicator is a dose indicator.
    assert confounded(list("aaabbccdd")) == [1.0, 10.0, 30.0]
    report = check(DOSE9, batch="lane", lane=list("aaabbccdd"))
    assert one(report, "status") == "descriptive"
    assert one(report, "reason") == (
        "the difference of 1, 10 and 30 ug/kg from the vehicle is also a difference "
        "between values of lane; add lane to the design, as a random intercept, "
        "random='(1 | lane)', when each dose spans two or more values of it, or as "
        "a fixed term, which sets the confounded doses aside"
    )
    # Lane b holds 10 and 30 alone, so 10 = b - 30; lane a holds the vehicle and 1,
    # whose difference stays within a lane.
    assert confounded(list("aaaaabbbb")) == [10.0, 30.0]
    # The vehicle in both lanes links every dose to it within a lane.
    assert confounded(list("abaababba")) == []
    # A lane per sample: a lane's effect is its sample's own deviation.
    report = check(DOSE9, batch="lane", lane=[f"L{i}" for i in range(9)])
    assert one(report, "status") == "inferential"
    assert not report.doses["confounded_with_batch"].any()
    # Two vehicle samples share a lane: the pooled lanes of one sample still hold the
    # third vehicle sample beside every dose.
    assert confounded(["L0", "L0", *[f"L{i}" for i in range(2, 9)]]) == []
    # Two dose-1 samples share a lane that holds nothing else.
    assert confounded(
        ["L0", "L1", "L2", "L3", "L3", *[f"L{i}" for i in range(5, 9)]]
    ) == [1.0]
    # The vehicle alone in its lane, the doses spread over two others.
    assert confounded(list("aaabcbcbc")) == [1.0, 10.0, 30.0]
    # Numeric lane codes are labels, not one slope.
    assert confounded([1, 1, 1, 2, 2, 3, 3, 4, 4]) == [1.0, 10.0, 30.0]
    # The design's sex and the lane absorb dose 1 together, though neither does alone.
    report = check(
        [0, 0, 1, 1, 1, 1, 10, 10],
        "~ sex",
        batch="lane",
        sex=["F", "F", "M", "M", "F", "F", "F", "F"],
        lane=list("aaaabbaa"),
    )
    assert dose_column(report, "confounded_with_batch") == [False, True, False]
    assert one(report, "status") == "descriptive"


def test_a_modelled_batch() -> None:
    # Lane as a fixed term absorbs every dose group that it confounds.
    report = check(DOSE9, "~ lane", batch="lane", lane=list("aaabbccdd"))
    assert one(report, "status") == "refused"
    assert one(report, "reason") == (
        "no treated dose group can be compared with the vehicle apart from the "
        "design's other terms: dose 1 ug/kg = laneb; dose 10 ug/kg = lanec; and "
        "dose 30 ug/kg = laned"
    )
    assert not report.doses["confounded_with_batch"].any()
    # Lane as a random intercept, two lanes per dose: the groups differ between lanes,
    # and the variance between lanes is estimated from the lanes of each dose.
    doses = [0, 0, 0, 0, 1, 1, 1, 1, 10, 10, 10, 10]
    report = check(doses, random="(1 | lane)", batch="lane", lane=list("aabbccddeeff"))
    assert one(report, "status") == "inferential"
    assert dose_column(report, "confounded_with_batch") == [False, True, True]
    # One lane per dose: the lanes cannot be separated from the dose groups.
    report = check(DOSE9, random="(1 | lane)", batch="lane", lane=list("aaabbccdd"))
    assert one(report, "status") == "refused"
    assert one(report, "reason").startswith(
        "each dose group holds the samples of a single lane"
    )


def test_a_unit_nested_one_to_one_in_the_dose_groups() -> None:
    # One animal per dose group, two samples each: the samples of an animal are not
    # replicates of its dose.
    doses = [0, 0, 1, 1, 10, 10, 30, 30]
    animal = ["A1", "A1", "A2", "A2", "A3", "A3", "A4", "A4"]
    samples = samples_of(doses, animal=animal)
    cube = cube_of(samples)
    report = check_dose_design(
        cube, Design(samples, "~ 1", random="(1 | animal)"), dose=AXIS
    )
    assert one(report, "status") == "refused"
    assert one(report, "reason") == (
        "each dose group holds the samples of a single animal, so no dose has a "
        "second animal to replicate it and the random intercept of animal cannot be "
        "estimated beside the dose groups; pool the samples of each animal at each "
        "dose, with cube.collapse_samples('animal') when each animal received one "
        "dose, and check the pooled cube"
    )
    # Pooled as the reason says, one profile per dose remains.
    pooled = cube.collapse_samples("animal")
    report = check_dose_design(pooled, Design(pooled.samples, "~ 1"), dose=AXIS)
    assert (one(report, "status"), one(report, "df_residual")) == ("descriptive", 0.0)
    # Two animals per dose separate the animals from the doses.
    report = check(
        [0] * 4 + [1] * 4 + [10] * 4,
        random="(1 | animal)",
        animal=[f"A{i // 2}" for i in range(12)],
    )
    assert one(report, "status") == "inferential"
    # Two random intercepts that group the samples alike: Design.check's message,
    # since the dose groups each hold two animals.
    report = check(
        [0] * 4 + [1] * 4 + [10] * 4,
        random="(1 | animal) + (1 | cage)",
        animal=[f"A{i // 2}" for i in range(12)],
        cage=[f"C{i // 2}" for i in range(12)],
    )
    assert one(report, "status") == "refused"
    assert "group the samples identically" in one(report, "reason")
    # One sample per animal: the random effect has nothing to estimate, as
    # Design.check says.
    report = check([0, 0, 1, 1, 10, 10], random="(1 | animal)", animal=list("abcdef"))
    assert one(report, "status") == "refused"
    assert "every group of the random effect (1 | animal) has a single sample" in one(
        report, "reason"
    )


def test_other_terms_absorb_dose_groups() -> None:
    # Sex equal to the top dose's indicator: the top dose cannot be told from sex.
    report = check([0, 0, 1, 1, 10, 10, 30, 30], "~ sex", sex=["F"] * 6 + ["M"] * 2)
    assert dose_column(report, "estimable") == [False, True, True, False]
    assert one(report, "status") == "inferential"
    assert one(report, "n_dose_groups") == 3
    # A covariate equal to the treated indicator: no dose apart from it.
    report = check([0, 0, 1, 1, 10, 10], "~ x", x=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert one(report, "status") == "refused"
    assert one(report, "reason") == (
        "no treated dose group can be compared with the vehicle apart from the "
        "design's other terms: dose 1 ug/kg = x - dose 10 ug/kg; and dose 10 ug/kg = "
        "x - dose 1 ug/kg"
    )


# ETAP's minimums, derived from the text by hand.


def etap(
    doses: list[float],
    per_sex: int = 4,
    *,
    cells: list | None = None,
    short: tuple | None = None,
) -> DoseDesignReport:
    """An ETAP-like design: per_sex animals of each sex per dose group, one fewer
    female at the dose ``short`` names."""
    dose, sex = [], []
    for level in [0.0, *doses]:
        for s in ("F", "M"):
            count = per_sex - (short is not None and (level, s) == short)
            dose += [level] * count
            sex += [s] * count
    samples = samples_of(dose, sex=sex)
    axis = DoseAxis("dose", unit="mg/kg", metric="administered")
    return check_dose_design(
        cube_of(samples, cells),
        Design(samples, "~ 1"),
        dose=axis,
        rules="etap_2024",
        sex="sex",
    )


def unmet(report: DoseDesignReport) -> dict[tuple, str]:
    table = report.minimums
    assert table is not None
    return {
        (table["minimum"][i], table["sex"][i]): table["observed"][i]
        for i in range(len(table))
        if not table["met"][i]
    }


def test_etap_minimums_by_hand() -> None:
    # The ETAP design: 8 doses and a vehicle, 4 of each sex, every sample passing.
    report = etap(ETAP_LADDER)
    assert unmet(report) == {}
    assert report.rules_status() == "met"
    assert report.minimums is not None
    assert report.minimums.filter(minimum="sexes")["observed"].tolist() == ["2 (F, M)"]
    # The TCDD ladder, males only, 2 or 3 per dose: its lowest step is 0.48 log10.
    doses = np.repeat(
        [0.0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30], [4, 2, 3, 3, 2, 3, 2, 3, 2]
    )
    samples = samples_of(doses.tolist(), sex=["M"] * doses.size)
    report = check_dose_design(
        cube_of(samples),
        Design(samples, "~ 1"),
        dose=AXIS,
        rules="etap_2024",
        sex="sex",
    )
    found = unmet(report)
    assert set(found) == {
        ("dose_spacing", None),
        ("sexes", None),
        ("animals_per_sex_and_dose", "M"),
    }
    assert found["sexes", None] == "1 (M)"
    assert found["dose_spacing", None] == (
        "steps of 0.48, 0.52, 0.48, 0.52, 0.48, 0.52, 0.48 log10 from the lowest dose"
    )
    assert report.rules_status() == "deviation"
    # 3 females at 3 mg/kg.
    found = unmet(etap(ETAP_LADDER, short=(3.0, "F")))
    assert found == {("animals_per_sex_and_dose", "F"): "3 at 3 mg/kg"}


@pytest.mark.parametrize(
    ("factor", "half"),
    [
        (3.0, True),
        (10 / 3, True),
        (3.5, True),
        (2.5, False),
        (3.6, False),
        (4.0, False),
    ],
)
def test_a_step_counts_as_half_a_log10_within_0_05(factor: float, half: bool) -> None:
    # The fourth step from the top of an otherwise exact ladder changes.
    doses = [300.0]
    for k in range(6):
        doses.append(doses[-1] / (factor if k == 3 else 10**0.5))
    doses.append(doses[-1] / 10)
    found = unmet(etap(sorted(doses), per_sex=1))
    assert (("dose_spacing", None) not in found) is half


@pytest.mark.parametrize(
    ("lowest", "full"), [(10.0, True), (9.0, True), (8.5, False), (3.0, False)]
)
def test_the_lowest_step_counts_as_a_full_log10_within_0_05(
    lowest: float, full: bool
) -> None:
    doses = [300.0 / 10 ** (k / 2) for k in range(7)]
    doses.append(doses[-1] / lowest)
    found = unmet(etap(sorted(doses), per_sex=1))
    assert (("dose_spacing", None) not in found) is full


def test_etap_excludes_a_dose_level_and_goes_on() -> None:
    n = 2 * 4 * 9
    # Three of the four males at 1 mg/kg fail the cell type's filter.
    dose = [level for level in [0.0, *ETAP_LADDER] for _ in range(8)]
    sex = (["F"] * 4 + ["M"] * 4) * 9
    cells = [30] * n
    for i in [i for i in range(n) if dose[i] == 1.0 and sex[i] == "M"][:3]:
        cells[i] = 3
    report = etap(ETAP_LADDER, cells=cells)
    table = report.minimums
    assert table is not None
    excluded = table.filter(minimum="passing_samples_per_dose", met=False)
    assert excluded["observed"].tolist() == ["1 at 1 mg/kg"]
    assert excluded["sex"].tolist() == ["M"]
    assert excluded["consequence"].tolist() == ["dose level excluded"]
    # ETAP's procedure goes on without the level: the design meets ETAP.
    assert report.rules_status() == "met"
    # Two passing female vehicle samples: ETAP does not model the cell type.
    cells = [30] * n
    for i in [i for i in range(n) if dose[i] == 0.0 and sex[i] == "F"][:2]:
        cells[i] = 3
    report = etap(ETAP_LADDER, cells=cells)
    assert unmet(report) == {("passing_vehicle_samples", "F"): "2"}
    assert report.rules_status() == "deviation"
    assert report.minimums is not None
    vehicle = report.minimums.filter(minimum="passing_vehicle_samples", met=False)
    assert vehicle["consequence"].tolist() == ["dose response not modelled"]


# Invariants.


def test_order_units_and_labels_do_not_matter() -> None:
    rng = np.random.default_rng(3)
    doses = np.repeat([0.0, 0.1, 1.0, 10.0], [4, 3, 3, 2])
    n = doses.size
    cells = rng.integers(4, 40, n)
    lanes = list("aabbccddeeff")
    sexes = ["F", "M"] * 6
    samples = samples_of(doses.tolist(), lane=lanes, sex=sexes)

    def tables(samples: dict, axis: DoseAxis, cells: np.ndarray) -> tuple:
        report = check_dose_design(
            cube_of(samples, cells.tolist()),
            Design(samples, "~ 1"),
            dose=axis,
            batch="lane",
            rules="etap_2024",
            sex="sex",
        )
        order = np.argsort(report.samples["sample_id"], kind="stable")
        minimums = report.minimums
        assert minimums is not None
        return (
            report.summary.select(["status", "df_residual", "n_dose_groups"]).to_dict(),
            report.doses.select(
                ["n_samples_used", "estimable", "confounded_with_batch"]
            ).to_dict(),
            report.samples.take(order)["used"].tolist(),
            minimums["met"].tolist(),
        )

    expected = tables(samples, AXIS, cells)
    # The samples in another order, in the design and the cube.
    order = rng.permutation(n)
    permuted = {
        key: [list(values)[i] for i in order] for key, values in samples.items()
    }
    found = tables(permuted, AXIS, cells[order])
    for ours, theirs in zip(found[:2], expected[:2], strict=True):
        for column in ours:
            assert ours[column].tolist() == theirs[column].tolist(), column
    assert found[2:] == expected[2:]
    # Doses in ng/kg, 1000 times larger, and other labels.
    renamed = {
        **samples,
        "dose": doses * 1000,
        "lane": [f"lane {label}" for label in lanes],
        "sex": ["female" if s == "F" else "male" for s in sexes],
    }
    found = tables(
        renamed, DoseAxis("dose", unit="ng/kg", metric="administered"), cells
    )
    for ours, theirs in zip(found[:2], expected[:2], strict=True):
        for column in ours:
            assert ours[column].tolist() == theirs[column].tolist(), column
    assert found[2:] == expected[2:]


# Arguments.


def test_arguments_are_refused_with_a_fix() -> None:
    samples = samples_of(DOSE9, lane=list("aaabbccdd"), sex=["M"] * 9)
    cube = cube_of(samples)
    design = Design(samples, "~ 1")
    with pytest.raises(TypeError, match="cube must be a Cube"):
        check_dose_design(samples, design, dose=AXIS)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"design must be a Design .* Design\(cube"):
        check_dose_design(cube, "~ 1", dose=AXIS)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dose must be a DoseAxis, such as"):
        check_dose_design(cube, design, dose="dose")  # type: ignore[arg-type]
    for formula in ("~ dose", "~ dose_treated + dose_s1"):
        columns = {**samples, "dose_treated": np.zeros(9), "dose_s1": np.zeros(9)}
        with pytest.raises(ValueError, match="adds the dose groups itself"):
            check_dose_design(cube, Design(columns, formula), dose=AXIS)
    with pytest.raises(ValueError, match="'dose'.*adds the dose groups itself"):
        check_dose_design(cube, Design(samples, "~ 1", random="(1 | dose)"), dose=AXIS)
    with pytest.raises(ValueError, match="random slopes"):
        check_dose_design(
            cube, Design(samples, "~ 1", random="(1 + dose | lane)"), dose=AXIS
        )
    with pytest.raises(ValueError, match="rules must be one of 'etap_2024'"):
        check_dose_design(cube, design, dose=AXIS, rules="ntp_2018", sex="sex")
    with pytest.raises(TypeError, match="rules must be one of"):
        check_dose_design(cube, design, dose=AXIS, rules=1, sex="sex")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs sex, the column of each sample's sex"):
        check_dose_design(cube, design, dose=AXIS, rules="etap_2024")
    with pytest.raises(ValueError, match="sex is read by a rule set"):
        check_dose_design(cube, design, dose=AXIS, sex="sex")
    with pytest.raises(TypeError, match="batch must name a column"):
        check_dose_design(cube, design, dose=AXIS, batch=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="did you mean 'lane'"):
        check_dose_design(cube, design, dose=AXIS, batch="lanes")
    with_missing = {**samples, "lane": ["a", None, *list("aabbccd")]}
    with pytest.raises(ValueError, match=r"missing for sample\(s\) 'S2'"):
        check_dose_design(cube, Design(with_missing, "~ 1"), dose=AXIS, batch="lane")
    with_missing = {**samples, "sex": ["M"] * 8 + [None]}
    with pytest.raises(ValueError, match=r"sex column 'sex' is missing for .*'S9'"):
        check_dose_design(
            cube, Design(with_missing, "~ 1"), dose=AXIS, rules="etap_2024", sex="sex"
        )
    with pytest.raises(ValueError, match="dose column 'conc' is not a column"):
        check_dose_design(
            cube, design, dose=DoseAxis("conc", unit="uM", metric="nominal")
        )
    for bad in (-1.0, np.nan):
        columns = {**samples, "dose": np.r_[bad, DOSE9[1:]]}
        with pytest.raises(ValueError, match="finite and at least 0"):
            check_dose_design(cube, Design(columns, "~ 1"), dose=AXIS)
    columns = {**samples, "dose": np.r_[0.0, 0.0, 0.0, 0.1 * 3, 0.3, DOSE9[5:]]}
    with pytest.raises(ValueError, match="differ by less than 1e-8 of the dose"):
        check_dose_design(cube, Design(columns, "~ 1"), dose=AXIS)
    unknown = {**samples, "sample_id": ["X1", *samples["sample_id"][1:]]}
    with pytest.raises(ValueError, match="not samples of the cube"):
        check_dose_design(cube, Design(unknown, "~ 1"), dose=AXIS)
    without_ids = {key: value for key, value in samples.items() if key != "sample_id"}
    with pytest.raises(ValueError, match="no 'sample_id' column"):
        check_dose_design(cube, Design(without_ids, "~ 1"), dose=AXIS)
    with pytest.raises(ValueError, match="min_cells must be non-negative"):
        check_dose_design(cube, design, dose=AXIS, min_cells=-1)
    with pytest.raises(ValueError, match="min_counts must be non-negative"):
        check_dose_design(cube, design, dose=AXIS, min_counts=-1)
    with pytest.raises(KeyError, match="no group with id 'liver'"):
        check_dose_design(cube, design, dose=AXIS, groups="liver")


# The report.


def test_the_report_reads_back() -> None:
    samples = samples_of(DOSE9, sex=["M"] * 9)
    cells = np.array([[30] * 9, [30, 30, 30, 4, 30, 30, 30, 30, 30]]).T
    cube = Cube(
        {"counts": np.full((2, 9, 1), 100, dtype=np.int64)},
        samples=samples,
        groups={"group_id": ["a", "b"]},
        features={"feature_id": ["all"]},
        n_cells=cells,
    )
    report = check_dose_design(
        cube, Design(samples, "~ 1"), dose=AXIS, rules="etap_2024", sex="sex"
    )
    assert (report.status("a"), report.status("b")) == ("inferential", "inferential")
    assert report.rules_status("a") == "deviation"
    with pytest.raises(
        ValueError, match="the result has 2 groups; name one with group"
    ):
        report.status()
    with pytest.raises(KeyError, match="the report did not check 'c'"):
        report.status("c")
    single = check_dose_design(cube, Design(samples, "~ 1"), dose=AXIS, groups="b")
    assert single.status() == "inferential"
    assert single.settings["groups"] == ["b"]
    with pytest.raises(ValueError, match="no rule set was checked"):
        single.rules_status()
    assert single.minimums is None
    assert "rules_status" not in single.summary
    # The settings: a new dict each time, which rebuilds the axis.
    settings = report.settings
    settings["batch"] = "changed"
    assert report.settings["batch"] is None
    assert DoseAxis(**report.settings["dose"]) == AXIS
    assert report.settings["min_cells"] == 10 and report.settings["groups"] is None
    # The repr: one line per cell type, with its dose groups and status.
    assert repr(report).splitlines() == [
        "DoseDesignReport: 2 cell types, 4 dose groups from 0 to 30 ug/kg, rules",
        "                  etap_2024",
        "  a: inferential; 4 of 4 dose groups keep a sample, 2 to 3 each; 5 residual df;",
        "     rules: deviation",
        "  b: inferential; 4 of 4 dose groups keep a sample, 1 to 3 each; 4 residual df;",
        "     rules: deviation",
    ]
    # Pickled and read back.
    copy = pickle.loads(pickle.dumps(report))
    assert repr(copy) == repr(report)
    assert copy.summary.equals(report.summary) and copy.doses.equals(report.doses)
    assert copy.samples.equals(report.samples)
    assert copy.minimums is not None and report.minimums is not None
    assert copy.minimums.equals(report.minimums)
    assert copy.settings == report.settings
    with pytest.raises(AttributeError, match="immutable"):
        report.summary = Table()  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del report.summary


def test_the_repr_of_a_cell_type_that_keeps_no_sample() -> None:
    report = check([0, 0, 1, 1], cells=[0, 0, 0, 0])
    assert repr(report).splitlines() == [
        "DoseDesignReport: 1 cell type, 2 dose groups from 0 to 1 ug/kg",
        "  cells: refused; every one of the 4 samples was left out (fewer than 10 cells,",
        "         or no counts)",
    ]


def test_etap_minimums_of_a_design_without_doses() -> None:
    samples = samples_of([0.0] * 8, sex=["F", "M"] * 4)
    report = check_dose_design(
        cube_of(samples),
        Design(samples, "~ 1"),
        dose=AXIS,
        rules="etap_2024",
        sex="sex",
    )
    table = report.minimums
    assert table is not None
    passing = table.filter(minimum="passing_samples_per_dose")
    # No dose level to exclude; the design minimums say what is missing.
    assert passing["observed"].tolist() == ["no doses", "no doses"]
    assert passing["met"].tolist() == [True, True]
    assert unmet(report) == {
        ("dose_levels", None): "0 doses and a vehicle",
        ("dose_spacing", None): "fewer than two doses",
    }


def test_the_repr_names_the_design_checks_findings() -> None:
    plate = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    report = check(DOSE9, "~ plate", plate=plate, batch="lane", lane=list("abcabcabc"))
    assert report.design_check.kinds() == ("numeric_codes",)
    assert repr(report).splitlines() == [
        "DoseDesignReport: 1 cell type, 4 dose groups from 0 to 30 ug/kg, batch 'lane'",
        "  cells: inferential; 4 of 4 dose groups keep a sample, 2 to 3 each; 4 residual",
        "         df",
        "  design check: 1 warning (numeric_codes); print report.design_check for the",
        "                messages",
    ]


def test_the_report_refuses_fields_of_the_wrong_type() -> None:
    report = check([0, 0, 1, 1])
    fields = [
        report.summary,
        report.doses,
        report.samples,
        None,
        report.design_check,
        report.settings,
    ]
    for position, label in enumerate(
        ["summary", "doses", "samples", "minimums", "design_check", "settings"]
    ):
        wrong = list(fields)
        wrong[position] = "text"
        with pytest.raises(TypeError, match=label):
            DoseDesignReport(*wrong)
    with pytest.raises(TypeError, match="not JSON serializable"):
        DoseDesignReport(*fields[:5], {"array": np.zeros(2)})
    with pytest.raises(ValueError, match="Out of range float values"):
        DoseDesignReport(*fields[:5], {"value": np.nan})


def test_the_chapter_example_runs() -> None:
    section = CHAPTER.read_text().split("(dose-design-check)=")[1].split("\n(")[0]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 1
    namespace = {"cp": cytepool, "cube": tcdd_like_cube()}
    exec(blocks[0], namespace)  # noqa: S102
    report = namespace["report"]
    assert dict(
        zip(report.summary["group_id"].tolist(), report.summary["status"].tolist())
    ) == {
        "hepatocyte": "inferential",
        "stellate": "inferential",
        "cholangiocyte": "descriptive",
        "macrophage": "refused",
    }
    flagged = namespace["flagged"]
    assert flagged["dose"].tolist() == [30.0]
    missed = namespace["missed"]
    assert set(missed["minimum"].tolist()) == {
        "dose_spacing",
        "sexes",
        "animals_per_sex_and_dose",
        "passing_samples_per_dose",
        "passing_vehicle_samples",
    }
    # The values the chapter states.
    text = CHAPTER.read_text()
    assert "keeps one mouse at 30 ug/kg" in text
    assert report.summary.filter(group_id="hepatocyte")["df_residual"].tolist() == [
        15.0
    ]
    assert "15 residual degrees of freedom" in text


def tcdd_like_cube() -> Cube:
    """The chapter's series: 24 mice, the vehicle and 8 doses, in lanes of two, with
    hepatocytes (abundant), stellate cells (depleted at the top dose), cholangiocytes
    (rare) and macrophages (absent from the vehicle mice)."""
    doses = np.repeat(
        [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], [4, 2, 3, 3, 2, 3, 2, 3, 2]
    )
    n = doses.size
    lanes = [
        f"L{position // 2 + 1}" for position in np.random.default_rng(7).permutation(n)
    ]
    samples = {
        "sample_id": [f"M{i + 1:02d}" for i in range(n)],
        "dose": doses,
        "lane": lanes,
        "sex": ["M"] * n,
    }
    step = np.searchsorted(np.unique(doses), doses)
    cells = np.zeros((n, 4), dtype=np.int64)
    cells[:, 0] = 300 + 10 * np.arange(n)
    cells[:, 1] = np.maximum(60 - 6 * step, 12)
    cells[doses == 30.0, 1] = [12, 4]
    cells[:, 2] = [
        3,
        12,
        4,
        5,
        11,
        4,
        2,
        13,
        6,
        3,
        2,
        5,
        3,
        4,
        2,
        6,
        3,
        5,
        2,
        4,
        3,
        5,
        2,
        6,
    ]
    cells[:, 3] = np.where(doses == 0, 0, 5 + 5 * step)
    counts = np.random.default_rng(8).poisson(5.0, (4, n, 40)) * (
        cells.T[:, :, None] > 0
    )
    return Cube(
        {"counts": counts.astype(np.int64)},
        samples=samples,
        groups={"group_id": ["hepatocyte", "stellate", "cholangiocyte", "macrophage"]},
        features={"feature_id": [f"g{i}" for i in range(40)]},
        n_cells=cells,
    )
