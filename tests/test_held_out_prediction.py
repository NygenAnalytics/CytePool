"""Tests of held-out prediction.

The scores of the samples and their AUC are recomputed here from the counts with
scipy's rankdata and Mann-Whitney statistic; the genes chosen for a held-out study must
be those that fit_sumrank and fit_meta_analysis rank first over the other studies; and
swapping cases and controls must turn the AUC into one minus it. validation/sumrank.py checks
the AUC against scikit-learn and the genes chosen against the authors' SumRank.
"""

import pickle
from functools import cache

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import mannwhitneyu, rankdata

from cytepool import (
    Cube,
    CubeCollection,
    Design,
    HeldOutPrediction,
    StudyResults,
    held_out_prediction,
    limma_voom,
)
from cytepool.stats.meta_analysis import fit_meta_analysis
from cytepool.stats.sumrank import fit_sumrank

CONTRAST = "case_vs_ctrl"
LEVELS = {"condition": ["ctrl", "case"]}


def study_cube(rng: np.random.Generator, study: str, n: int, groups: list[str]) -> Cube:
    genes = [f"g{j:03d}" for j in range(80)]
    condition = np.array(["ctrl", "case"] * (n // 2))
    change = np.zeros(80)
    change[:8], change[8:14] = 0.8, -0.8
    mean = np.exp(rng.normal(4.0, 1.0, size=(len(groups), 1, 80)))
    mean = mean * np.where(condition[None, :, None] == "case", 2.0**change, 1.0)
    n_cells = np.full((n, len(groups)), 30)
    n_cells[0, -1] = 3  # a sample with too few cells for the last group's fit
    return Cube(
        {"counts": rng.negative_binomial(10, 10 / (10 + mean))},
        samples={
            "sample_id": [f"{study}{i}" for i in range(n)],
            "condition": condition,
        },
        groups={"group_id": groups},
        features={"feature_id": genes},
        n_cells=n_cells,
    )


def design(samples: object) -> Design:
    return Design(samples, "~ condition", levels=LEVELS, contrasts={CONTRAST: [0, 1]})


@cache
def studies() -> StudyResults:
    """Four studies with groups T and B, the last without B."""
    rng = np.random.default_rng(20261031)
    cubes = {
        "first": study_cube(rng, "a", 10, ["T", "B"]),
        "second": study_cube(rng, "b", 12, ["T", "B"]),
        "third": study_cube(rng, "c", 8, ["T", "B"]),
        "fourth": study_cube(rng, "d", 10, ["T"]),
    }
    collection = CubeCollection(cubes)
    results = {
        s: limma_voom(
            collection.aligned_cube(s), design(collection.aligned_cube(s).samples)
        )
        for s in collection.studies
    }
    return StudyResults(collection, results)


EXPLICIT = {"variable": "condition", "case": "case", "control": "ctrl"}


@cache
def prediction() -> HeldOutPrediction:
    return held_out_prediction(
        studies(), CONTRAST, n_features=6, proportion_of_studies=0.7, **EXPLICIT
    )


def chosen(found: HeldOutPrediction, study: str, method: str, group: str, direction: str) -> list[str]:  # fmt: skip
    genes = found.features
    rows = (
        (genes["study"] == study)
        & (genes["method"] == method)
        & (genes["group_id"] == group)
        & (genes["direction"] == direction)
    )
    assert_array_equal(genes["rank"][rows], np.arange(1, rows.sum() + 1))
    return genes["feature_id"][rows].tolist()


def recomputed(found: HeldOutPrediction, study: str, method: str) -> dict[str, float]:
    """Every sample's score, from the counts of the aligned cube."""
    results = studies()
    collection = results.collection
    cube, samples = collection.aligned_cube(study), results[study].samples
    ids = cube.samples["sample_id"].tolist()
    columns = {f: i for i, f in enumerate(cube.features["feature_id"].tolist())}
    total = np.zeros(len(ids))
    for group in cube.groups["group_id"].tolist():
        used = samples["sample_id"][(samples["group_id"] == group) & samples["used"]]
        counts = np.asarray(cube.layer("counts", group), dtype=float)
        for direction, sign in (("up", 1.0), ("down", -1.0)):
            genes = [columns[g] for g in chosen(found, study, method, group, direction)]
            n = len(genes)
            if n == 0:
                continue
            raw = {}
            for sample in used.tolist():
                ranks = rankdata(-counts[ids.index(sample)])  # 1 for the highest
                raw[sample] = 1 - (ranks[genes].sum() - n * (n + 1) / 2) / (
                    n * counts.shape[1]
                )
            low, high = min(raw.values()), max(raw.values())
            spread = high - low
            scaled = {s: (v - low) / spread if spread else 0.0 for s, v in raw.items()}
            mean = np.mean(list(scaled.values()))
            total += sign * np.array([scaled.get(s, mean) for s in ids])
    return dict(zip(ids, total.tolist(), strict=True))


@pytest.mark.parametrize("method", ["sumrank", "meta_analysis"])
def test_scores_and_auc_match_an_independent_computation(method: str) -> None:
    found = prediction()
    collection = studies().collection
    for study in collection.studies:
        scores = found.scores.filter(
            (found.scores["study"] == study) & (found.scores["method"] == method)
        )
        expected = recomputed(found, study, method)
        assert_allclose(
            scores["score"], [expected[s] for s in scores["sample_id"].tolist()],
            rtol=1e-12, atol=1e-14,
        )  # fmt: skip
        case = scores["case"]
        statistic = mannwhitneyu(scores["score"][case], scores["score"][~case])
        row = (found.auc["study"] == study) & (found.auc["method"] == method)
        n_pairs = case.sum() * (~case).sum()
        assert found.auc["auc"][row][0] == pytest.approx(
            statistic.statistic / n_pairs, rel=1e-12
        )
        assert found.auc["n_cases"][row][0] == case.sum()
        assert found.auc["reason"][row][0] == ""
    # A sample left out of a group's fit takes the mean there, so it is scored too.
    assert len(found.scores) == 2 * sum(
        collection.aligned_cube(s).n_samples for s in collection.studies
    )


def test_genes_are_chosen_from_the_other_studies() -> None:
    found = prediction()
    results = studies()
    collection = results.collection
    for group in ("T", "B"):
        table = results.table(group, CONTRAST)
        shape = (collection.n_studies, collection.n_features)
        p = table["p_value"].reshape(shape)
        change = table["log2_fold_change"].reshape(shape)
        error = table["standard_error"].reshape(shape)
        ids = np.array(collection.features["feature_id"].tolist())
        for held, study in enumerate(collection.studies):
            others = [
                i for i in range(collection.n_studies)
                if i != held and not np.isnan(p[i]).all()
            ]  # fmt: skip
            statistics = np.where(change < 0, np.log10(p), -np.log10(p))[others]
            fit = fit_sumrank(statistics, proportion_of_studies=0.7)
            cap = fit.n_studies_summed / 2
            for direction in ("up", "down"):
                values = getattr(fit, f"{direction}_statistic")
                order = np.lexsort((np.arange(values.size), values))
                best = [i for i in order if fit.ranked[i] and values[i] < cap][:6]
                assert (
                    chosen(found, study, "sumrank", group, direction)
                    == ids[best].tolist()
                )
            universe = fit.ranked
            combined = fit_meta_analysis(
                change[others][:, universe], error[others][:, universe],
                method="reml", test="knapp_hartung",
            )  # fmt: skip
            positions = np.flatnonzero(universe)
            for direction, sign in (("up", 1.0), ("down", -1.0)):
                candidates = np.flatnonzero(sign * combined.estimate > 0)
                order = candidates[
                    np.lexsort((candidates, combined.p_value[candidates]))
                ]
                expected = ids[positions[order[:6]]].tolist()
                assert (
                    chosen(found, study, "meta_analysis", group, direction) == expected
                )


def test_the_held_out_studys_own_results_choose_none_of_its_genes() -> None:
    results = studies()
    collection = results.collection
    # The first study analysed with its labels permuted: its own genes stay, the
    # others' change, as the first study is among their training studies.
    cube = collection.aligned_cube("first")
    labels = cube.samples["condition"][[1, 0, 3, 2, 5, 4, 7, 6, 9, 8]]
    shuffled = dict(results)
    shuffled["first"] = limma_voom(
        cube, design({"sample_id": cube.samples["sample_id"], "condition": labels})
    )
    other = held_out_prediction(
        StudyResults(collection, shuffled),
        CONTRAST,
        n_features=6,
        proportion_of_studies=0.7,
        **EXPLICIT,
    )
    found = prediction()
    for method in ("sumrank", "meta_analysis"):
        assert chosen(other, "first", method, "T", "up") == chosen(
            found, "first", method, "T", "up"
        )
    assert chosen(other, "second", "sumrank", "T", "up") != chosen(
        found, "second", "sumrank", "T", "up"
    )


def test_swapping_cases_and_controls_gives_one_minus_the_auc() -> None:
    swapped = held_out_prediction(
        studies(),
        CONTRAST,
        n_features=6,
        proportion_of_studies=0.7,
        variable="condition",
        case="ctrl",
        control="case",
    )
    found = prediction()
    assert_allclose(swapped.auc["auc"], 1 - found.auc["auc"], rtol=1e-13)
    assert_array_equal(swapped.scores["case"], ~found.scores["case"])
    assert_array_equal(swapped.scores["score"], found.scores["score"])


def test_a_study_without_controls_has_no_auc() -> None:
    found = held_out_prediction(
        studies(),
        CONTRAST,
        n_features=3,
        variable="condition",
        case="case",
        control="healthy",
    )
    assert np.isnan(found.auc["auc"]).all()
    assert (found.auc["n_controls"] == 0).all()
    assert "0 controls" in found.auc["reason"][0]
    assert found.scores["case"].all()  # only cases are scored


def test_a_group_tested_by_no_other_study_gives_no_genes() -> None:
    results = studies()
    # Group B with the first study alone: held out, it has no other study.
    kept = results.collection.select(studies=["first", "fourth"])
    found = held_out_prediction(
        StudyResults(kept, {s: results[s] for s in kept.studies}),
        CONTRAST,
        groups="B",
        n_features=4,
        **EXPLICIT,
    )
    row = (found.auc["study"] == "first") & (found.auc["method"] == "sumrank")
    assert found.auc["n_groups"][row][0] == 0
    assert found.auc["n_features_up"][row][0] == 0
    # Every score is then 0, and the AUC 0.5.
    assert found.auc["auc"][row][0] == 0.5


def test_the_contrast_gives_the_variable_and_the_levels_it_compares() -> None:
    # CONTRAST, [0, 1] on ~ condition, compares the level case with ctrl, the
    # reference: the variable, case and control that prediction() gives.
    found = prediction()
    read = held_out_prediction(
        studies(), CONTRAST, n_features=6, proportion_of_studies=0.7
    )
    for name in ("auc", "scores", "features"):
        assert getattr(read, name).equals(getattr(found, name)), name
    assert read.settings == found.settings
    # The studies tested one contrast, which may be left out.
    alone = held_out_prediction(studies(), n_features=6, proportion_of_studies=0.7)
    assert alone.auc.equals(found.auc) and alone.scores.equals(found.scores)


@cache
def studies_with_age() -> StudyResults:
    """The four studies with a covariate in their designs, and contrasts of it and of
    the mean of the cases, which compare no two levels of a factor."""
    results = studies()
    collection = results.collection
    fitted = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        samples = cube.samples.with_columns(
            {"age": np.linspace(30.0, 60.0, cube.n_samples)}
        )
        # On the columns (Intercept), conditioncase and age.
        contrasts = {CONTRAST: [0, 1, 0], "age": [0, 0, 1], "case_mean": [1, 1, 0]}
        design = Design(
            samples, "~ condition + age", levels=LEVELS, contrasts=contrasts
        )
        fitted[study] = limma_voom(cube, design)
    return StudyResults(collection, fitted)


def test_a_contrast_that_compares_no_two_levels_needs_them_given() -> None:
    results = studies_with_age()
    for contrast in ("age", "case_mean"):
        with pytest.raises(
            ValueError,
            match=f"contrast '{contrast}' does not compare two levels of one factor "
            "in every study's design; give the variable that tells cases from "
            "controls",
        ):
            held_out_prediction(results, contrast, n_features=6)
    given = held_out_prediction(results, "age", n_features=6, **EXPLICIT)
    assert given.settings["contrast"] == "age"
    # A covariate beside the factor leaves the comparison of its levels as it is.
    settings = held_out_prediction(results, CONTRAST, n_features=6).settings
    assert [settings[name] for name in EXPLICIT] == list(EXPLICIT.values())
    # Some of the variable and its values are not enough.
    for partial in ({"variable": "condition"}, {"case": "case", "control": "ctrl"}):
        with pytest.raises(
            ValueError,
            match="variable, case and control are given together, or none of them",
        ):
            held_out_prediction(studies(), CONTRAST, **partial)


def test_a_number_of_studies_summed_is_the_proportion_that_gives_it() -> None:
    # In group T each held-out study leaves three others, of which 0.7 sums
    # floor(2.1) = 2, as n_studies_summed=2 does; every study, by default, differs.
    options = {"groups": "T", "n_features": 6}
    by_number = held_out_prediction(studies(), CONTRAST, n_studies_summed=2, **options)
    by_proportion = held_out_prediction(
        studies(), CONTRAST, proportion_of_studies=0.7, **options
    )
    for name in ("auc", "scores", "features"):
        assert getattr(by_number, name).equals(getattr(by_proportion, name)), name
    every = held_out_prediction(studies(), CONTRAST, **options)
    assert not every.features.equals(by_number.features)
    settings = by_number.settings, by_proportion.settings
    assert [s["n_studies_summed"] for s in settings] == [2, None]
    assert [s["proportion_of_studies"] for s in settings] == [None, 0.7]
    with pytest.raises(
        ValueError,
        match=r"\(n_studies_summed\) or their proportion \(proportion_of_studies\), "
        "not both",
    ):
        held_out_prediction(
            studies(), CONTRAST, n_studies_summed=2, proportion_of_studies=0.7
        )
    for number in (0, 4):
        with pytest.raises(
            ValueError,
            match="n_studies_summed must be from 1 to the number of studies, 3, got "
            f"{number}",
        ):
            held_out_prediction(studies(), CONTRAST, n_studies_summed=number)


def test_the_result_is_a_record_that_cannot_change() -> None:
    found = prediction()
    assert found.settings["n_features"] == 6
    assert found.settings["meta_analysis"] == {
        "method": "reml",
        "test": "knapp_hartung",
    }
    assert found.provenance["analysis"] == "held_out_prediction"
    with pytest.raises(AttributeError, match="immutable"):
        found.auc = found.scores  # type: ignore[misc]
    copy = pickle.loads(pickle.dumps(found))
    assert copy.auc.equals(found.auc) and copy.features.equals(found.features)
    assert "sumrank: mean AUC" in repr(found)
    assert max(map(len, repr(found).splitlines())) <= 80
    with pytest.raises(TypeError, match="features must be a Table"):
        HeldOutPrediction(found.auc, found.scores, {}, {})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"case": "ctrl"}, ValueError, "both 'ctrl'"),
        ({"variable": "donor"}, ValueError, "not a variable of the design"),
        ({"variable": 1}, TypeError, "variable must be a string"),
        ({"n_features": 0}, ValueError, "at least 1"),
        ({"n_features": 2.5}, TypeError, "whole number"),
        ({"groups": []}, ValueError, "groups is empty"),
        ({"groups": ["T", "T"]}, ValueError, "more than once"),
        ({"groups": "NK"}, KeyError, "no group"),
        ({"proportion_of_studies": 2}, ValueError, r"in \(0, 1\]"),
        ({"contrast": "other"}, KeyError, "no study tested the contrast"),
    ],
)
def test_invalid_arguments_are_refused(
    arguments: dict[str, object], error: type[Exception], match: str
) -> None:
    given = {"contrast": CONTRAST, "variable": "condition", "case": "case"}
    given |= {"control": "ctrl"} | arguments
    with pytest.raises(error, match=match):
        held_out_prediction(studies(), **given)  # type: ignore[arg-type]


def test_the_figure_shows_every_study_and_method() -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    from cytepool.plotting import plot_held_out_auc

    found = prediction()
    ax = plot_held_out_auc(found, ax=Figure().subplots())
    studies_shown = [label.get_text() for label in ax.get_yticklabels()]
    assert studies_shown == ["first", "second", "third", "fourth"]
    for method in ("sumrank", "meta_analysis"):
        (points,) = [c for c in ax.collections if c.get_gid() == method]
        expected = found.auc["auc"][found.auc["method"] == method]
        assert_allclose(points.get_offsets()[:, 0], expected)
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    mean = found.auc["auc"][found.auc["method"] == "sumrank"].mean()
    assert legend[0] == f"SumRank (mean {mean:.2f})"
    (line,) = [line for line in ax.lines if line.get_gid() == "chance"]
    assert line.get_xdata()[0] == 0.5
    with pytest.raises(TypeError, match="HeldOutPrediction"):
        plot_held_out_auc(found.auc)  # type: ignore[arg-type]
