"""Every study of a collection analysed alike, and the results lined up gene by gene.

``analyse_each_study`` is checked against the loop it replaces, written out with the
validated functions, and ``StudyResults`` against the same results given as a mapping
and each study's own table.
"""

import pickle
import re
from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from test_cube_collection import LEVELS, fixture_collection

from cytepool import (
    CubeCollection,
    Design,
    DifferentialExpressionResult,
    StudyResults,
    analyse_each_study,
    dream_voom,
    limma_trend,
    limma_voom,
    meta_analysis,
)

FORMULA = "~ donor + condition"
CONTRASTS = {"stim": "conditionstim"}


@cache
def pair() -> CubeCollection:
    """The two studies whose donors were sampled in both conditions."""
    return fixture_collection().select(studies=["lupus", "covid"])


@cache
def voom_results() -> StudyResults:
    return analyse_each_study(pair(), FORMULA, levels=LEVELS, contrasts=CONTRASTS)


def loop(
    collection: CubeCollection,
    formula: str,
    analysis: Any = limma_voom,
    random: str | None = None,
    **options: Any,
) -> dict[str, DifferentialExpressionResult]:
    """The loop over studies, written out."""
    results = {}
    for study in collection.studies:
        cube = collection.aligned_cube(study)
        design = Design(
            cube.samples, formula, random=random, levels=LEVELS, contrasts=CONTRASTS
        )
        results[study] = analysis(cube, design, **options)
    return results


def assert_same_result(
    ours: DifferentialExpressionResult, expected: DifferentialExpressionResult
) -> None:
    assert ours.method == expected.method
    assert ours.summary.equals(expected.summary)
    assert ours.samples.equals(expected.samples)
    assert list(ours.tables) == list(expected.tables)
    for key, table in expected.tables.items():
        assert ours.tables[key].equals(table), key
    assert ours.provenance == expected.provenance
    assert ours.cube is not None and expected.cube is not None
    assert ours.cube.fingerprint() == expected.cube.fingerprint()
    assert ours.design is not None and expected.design is not None
    assert ours.design.formula == expected.design.formula


def test_each_study_is_analysed_as_the_loop_analyses_it() -> None:
    results = voom_results()
    expected = loop(pair(), FORMULA)
    assert results.studies == ("lupus", "covid")
    for study in pair().studies:
        assert_same_result(results[study], expected[study])
    # Options reach the analysis, and random effects the design.
    first = pair().groups["group_id"][0]
    trend = analyse_each_study(
        pair(),
        FORMULA,
        analysis=limma_trend,
        levels=LEVELS,
        contrasts=CONTRASTS,
        groups=first,
    )
    for study, result in loop(pair(), FORMULA, limma_trend, groups=first).items():
        assert_same_result(trend[study], result)
    dream = analyse_each_study(
        pair(),
        "~ condition",
        analysis=dream_voom,
        random="(1 | donor)",
        levels=LEVELS,
        contrasts=CONTRASTS,
        groups=[first],
    )
    expected_dream = loop(
        pair(), "~ condition", dream_voom, random="(1 | donor)", groups=[first]
    )
    for study, result in expected_dream.items():
        assert_same_result(dream[study], result)


def test_the_table_lines_up_each_study_and_the_matrix_takes_one_column() -> None:
    results = voom_results()
    collection = pair()
    features = collection.features["feature_id"].tolist()
    for group in collection.groups["group_id"].tolist():
        table = results.table(group, "stim")
        assert table.equals(
            StudyResults(collection, dict(results)).table(group, "stim")
        )
        for column in ("p_value", "t", "log2_fold_change"):
            matrix = results.matrix(column, group, "stim")
            assert matrix.shape == (collection.n_studies, collection.n_features)
            assert matrix.dtype == np.float64 and not matrix.flags.writeable
            # Each study's own table, placed at its features; NaN elsewhere.
            for row, study in enumerate(collection.studies):
                expected = np.full(len(features), np.nan)
                if group in results[study].groups:
                    own = results[study].table(group, "stim")
                    positions = [features.index(f) for f in own["feature_id"].tolist()]
                    expected[positions] = own[column]
                assert_array_equal(matrix[row], expected)


def test_the_keys_may_be_left_out_when_the_studies_analysed_one() -> None:
    results = voom_results()
    groups = pair().groups["group_id"].tolist()
    # One contrast: only the group is needed.
    assert results.table(groups[1]).equals(results.table(groups[1], "stim"))
    assert_array_equal(
        results.matrix("t", groups[1]), results.matrix("t", groups[1], "stim")
    )
    message = (
        f"the studies analysed 5 groups; name one with group, one of '{groups[0]}', "
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        results.table()
    one = analyse_each_study(
        pair(), FORMULA, levels=LEVELS, contrasts=CONTRASTS, groups=groups[0]
    )
    assert one.table().equals(results.table(groups[0], "stim"))
    assert_array_equal(one.matrix("p_value"), results.matrix("p_value", groups[0]))


def test_the_summary_designs_and_repr_cover_every_study() -> None:
    results = voom_results()
    first = pair().groups["group_id"][0]
    covid = pair().aligned_cube("covid")
    dream = dream_voom(
        covid,
        Design(covid.samples, "~ condition", random="(1 | donor)", levels=LEVELS),
        contrasts=CONTRASTS,
        groups=[first],
    )
    mixed = StudyResults(pair(), {"covid": dream, "lupus": results["lupus"]})
    assert mixed.studies == ("lupus", "covid")  # the collection's order
    summary = mixed.summary
    lupus, other = results["lupus"].summary, dream.summary
    assert summary.columns[0] == "study"
    assert summary["study"].tolist() == ["lupus"] * len(lupus) + ["covid"]
    for name in lupus.columns:
        assert summary[name].tolist()[: len(lupus)] == lupus[name].tolist(), name
    for name in other.columns:
        assert summary[name].tolist()[len(lupus) :] == other[name].tolist(), name
    # dream's df_method is missing in limma's rows.
    assert summary["df_method"].tolist() == [None] * len(lupus) + ["kenward_roger"]
    assert mixed.designs["covid"] is dream.design
    with pytest.raises(TypeError):
        mixed.designs["covid"] = None  # type: ignore[index]
    lines = repr(results).splitlines()
    assert lines == [
        "StudyResults of 2 studies: 5 groups, 1 contrast: stim",
        "  lupus: limma_voom, 4 groups, 4 fitted",
        "  covid: limma_voom, 4 groups, 4 fitted",
    ]


def test_the_results_are_a_read_only_mapping_that_pickles() -> None:
    results = voom_results()
    assert list(results) == ["lupus", "covid"] and len(results) == 2
    assert results.collection is pair()
    with pytest.raises(KeyError, match="no study named 'atlas'; studies are: 'lupus'"):
        results["atlas"]
    with pytest.raises(AttributeError, match="immutable"):
        results.extra = 1  # type: ignore[attr-defined]
    restored = pickle.loads(pickle.dumps(results))
    for study in results:
        assert_same_result(restored[study], results[study])
    group = pair().groups["group_id"][0]
    assert restored.table(group).equals(results.table(group))
    # The combining functions take the results as they take a dict of them.
    meta = meta_analysis(results)
    expected = meta_analysis(dict(results))
    assert meta.table(group, "stim").equals(expected.table(group, "stim"))


def test_invalid_results_and_analyses_are_refused() -> None:
    results = voom_results()
    lupus = results["lupus"]
    with pytest.raises(TypeError, match="collection must be a CubeCollection"):
        StudyResults({"lupus": lupus}, {"lupus": lupus})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="results must be a mapping"):
        StudyResults(pair(), [lupus])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no other: missing 'covid'"):
        StudyResults(pair(), {"lupus": lupus})
    extra = {"lupus": lupus, "covid": lupus, "atlas": lupus}
    with pytest.raises(ValueError, match="not in the collection: 'atlas'"):
        StudyResults(pair(), extra)
    with pytest.raises(TypeError, match=r"results\['covid'\] must be a Differential"):
        StudyResults(pair(), {"lupus": lupus, "covid": lupus.table("CL:0000236")})
    with pytest.raises(KeyError, match="no study's table has a column 'pvalue'"):
        results.matrix("pvalue", "CL:0000236")
    with pytest.raises(ValueError, match="'status' holds text"):
        results.matrix("status", "CL:0000236")
    with pytest.raises(TypeError, match="column must be a column name"):
        results.matrix(3, "CL:0000236")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="collection must be a CubeCollection"):
        analyse_each_study(dict(results), FORMULA)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="formula must be a formula"):
        analyse_each_study(pair(), 3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="analysis must be limma_voom, limma_trend"):
        analyse_each_study(pair(), FORMULA, analysis=len)
    # A study's error says which study it came from.
    with pytest.raises(ValueError, match="uses site, which is not a column") as raised:
        analyse_each_study(pair(), "~ site + condition", levels=LEVELS)
    assert "(in study 'lupus' of the collection)" in raised.value.__notes__
