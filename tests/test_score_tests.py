"""limma on scores of samples, checked against the steps a user would take by hand.

By hand, the design is built from the samples the scores kept, and the scores go
through the validated ``fit_linear_model``, ``contrasts_fit`` and
``moderated_statistics``; ``limma`` instead matches the rows of the design built for
the whole cube by ``sample_id``. The two must agree bit for bit.
"""

import re
from functools import cache
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import (
    Activities,
    Cube,
    Design,
    Network,
    PathwayScores,
    Table,
    gsva,
    limma,
    limma_voom,
    ulm,
)
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import contrasts_fit, fit_linear_model
from cytepool.stats.multiple_testing import adjust_p_values

LEVELS = {"condition": ["control", "treated"]}
CONTRASTS = {"treated": "conditiontreated"}
COLUMNS = (
    "estimate",
    "standard_error",
    "average",
    "t",
    "p_value",
    "adjusted_p_value",
    "log_odds",
)


@cache
def cube() -> Cube:
    """Six donors sampled in both conditions; S3 has too few cells to be scored."""
    rng = np.random.default_rng(8)
    n_samples, n_genes = 12, 400
    means = np.exp(rng.uniform(2.0, 7.0, n_genes))
    counts = rng.negative_binomial(10, 10 / (10 + means), (n_samples, n_genes))
    condition = np.array(["control", "treated"] * (n_samples // 2))
    counts[condition == "treated", :30] *= 3
    cells = np.full((n_samples, 1), 40)
    cells[3] = 5
    return Cube(
        {"counts": counts[np.newaxis]},
        samples={
            "sample_id": [f"S{i}" for i in range(n_samples)],
            "donor": np.repeat([f"D{i}" for i in range(6)], 2),
            "condition": condition,
        },
        groups={"group_id": ["B cells"]},
        features={
            "feature_id": [f"ENSG{j}" for j in range(n_genes)],
            "symbol": [f"G{j}" for j in range(n_genes)],
        },
        n_cells=cells,
    )


@cache
def scores() -> PathwayScores:
    sets = {
        "treated": [f"G{j}" for j in range(30)],
        "other": [f"G{j}" for j in range(50, 90)],
        "third": [f"G{j}" for j in range(100, 160)],
    }
    return gsva(cube(), sets, id_column="symbol")


@cache
def activities() -> Activities:
    network = Network(
        {
            "A": {f"G{j}": 1.0 for j in range(30)},
            "B": {f"G{j}": (-1.0) ** j for j in range(40, 80)},
        }
    )
    return ulm(cube(), network, group="B cells", id_column="symbol")


def design(samples: Any = None, formula: str = "~ donor + condition") -> Design:
    table = cube().samples if samples is None else samples
    return Design(table, formula, levels=LEVELS, contrasts=CONTRASTS)


def by_hand(
    values: np.ndarray, built: Design, *, weights: Any = None, **moderation: Any
) -> dict[str, np.ndarray]:
    """The steps of GSVA's guide: lmFit, contrasts.fit, eBayes and topTable."""
    fit = fit_linear_model(values, built, weights=weights)
    tested = contrasts_fit(fit, built.contrasts, method="exact")
    moderated = moderated_statistics(tested, **moderation)
    p_value = moderated.p_value[:, 0]
    return {
        "estimate": tested.coefficients[:, 0],
        "standard_error": tested.stdev_unscaled[:, 0]
        * np.sqrt(moderated.posterior_variance),
        "average": fit.mean_expression,
        "t": moderated.t[:, 0],
        "p_value": p_value,
        "adjusted_p_value": adjust_p_values(p_value, "bh"),
        "log_odds": moderated.log_odds[:, 0],
    }


def assert_table(table: Table, id_column: str, ids: list[str], expected: dict) -> None:
    assert table.columns == (id_column, *COLUMNS)
    assert table[id_column].tolist() == ids
    for name in COLUMNS:
        assert_array_equal(table[name], expected[name], err_msg=name)


def test_scores_are_fitted_on_the_cubes_design_rows_of_their_samples() -> None:
    result = scores()
    # The scores left S3 out; the guide builds the design from the samples kept.
    assert "S3" not in result.profiles["sample_id"].tolist()
    expected = by_hand(result.scores, design(result.profiles))
    table = limma(result, design())
    assert_table(table, "gene_set", list(result.gene_sets), expected)
    # The set of the tripled genes is higher in treated samples.
    assert table["t"][0] > 10 and table["p_value"][0] < 1e-6
    # The contrast may be named; a sort reads the table from the smallest p-value.
    assert limma(result, design(), contrast="treated").equals(table)
    assert table.sort("p_value")["gene_set"].tolist()[0] == "treated"


def test_activities_of_samples_are_fitted_alike() -> None:
    result = activities()
    expected = by_hand(result.scores, design(result.profiles))
    assert_table(limma(result, design()), "source", list(result.sources), expected)


def test_an_array_takes_the_design_rows_or_the_samples_named() -> None:
    result = scores()
    values = result.scores
    rows = result.profiles["sample_id"].tolist()
    expected = by_hand(values, design(result.profiles))
    ids = [f"set{j}" for j in range(values.shape[1])]
    named = limma(values, design(), sample_ids=rows, feature_ids=ids)
    assert_table(named, "feature_id", ids, expected)
    # Without sample_ids, the rows are the design's samples in order.
    in_order = limma(values, design(result.profiles))
    assert_table(
        in_order, "feature_id", ["feature_0", "feature_1", "feature_2"], expected
    )
    # Samples that the design lacks are left out; the rows' order does not matter.
    shuffled = [10, 0, 3, 7, 1, 9, 2, 8, 4, 6, 5]
    extra = np.vstack([values[shuffled], values[:1] + 1.0])
    found = limma(extra, design(), sample_ids=[rows[i] for i in shuffled] + ["X"])
    for name in COLUMNS:
        assert_allclose(found[name], expected[name], rtol=1e-12)


def test_a_donor_without_samples_leaves_the_fit_as_its_columns_do() -> None:
    # Without D1's samples, the cube's design has a column of zeros (donorD1), which
    # the fit leaves out, as a design built from the other samples has no D1.
    result = scores()
    keep = [
        i for i, donor in enumerate(result.profiles["donor"].tolist()) if donor != "D1"
    ]
    samples = result.profiles.take(np.array(keep))
    values = result.scores[keep]
    expected = by_hand(values, design(samples))
    table = limma(values, design(), sample_ids=samples["sample_id"].tolist())
    assert_table(table, "feature_id", ["feature_0", "feature_1", "feature_2"], expected)


def test_weights_trend_and_robust_reach_the_fit_and_the_moderation() -> None:
    result = scores()
    weights = np.linspace(0.5, 2.0, len(result.profiles))
    expected = by_hand(
        result.scores, design(result.profiles), weights=weights, trend=True, robust=True
    )
    table = limma(result, design(), weights=weights, trend=True, robust=True)
    assert_table(table, "gene_set", list(result.gene_sets), expected)
    sizes = np.asarray(result.n_genes, dtype=np.float64)
    expected = by_hand(result.scores, design(result.profiles), trend=sizes)
    table = limma(result, design(), trend=sizes)
    assert_table(table, "gene_set", list(result.gene_sets), expected)
    # Weights per value fit each set alone, so they agree to rounding.
    per_value = np.tile(weights[:, None], (1, result.scores.shape[1]))
    alike = limma(result, design(), weights=per_value)
    expected = by_hand(result.scores, design(result.profiles), weights=per_value)
    assert_table(alike, "gene_set", list(result.gene_sets), expected)
    once = limma(result, design(), weights=weights)
    for name in COLUMNS:
        assert_allclose(alike[name], once[name], rtol=1e-12, atol=1e-15)


def test_what_limma_cannot_test_is_refused() -> None:
    result = scores()
    values = result.scores
    full = cube()
    of_contrasts = ulm(
        limma_voom(full, design(), min_cells=0),
        Network({"A": {f"G{j}": 1.0 for j in range(30)}}),
        id_column="symbol",
        min_targets=3,
    )
    with pytest.raises(ValueError, match="these activities are of the statistics"):
        limma(of_contrasts, design())
    with pytest.raises(ValueError, match="limma fits fixed effects only"):
        limma(
            result,
            Design(full.samples, "~ condition", random="(1 | donor)", levels=LEVELS),
        )
    with pytest.raises(ValueError, match="feature_ids applies to an array"):
        limma(result, design(), feature_ids=["a", "b", "c"])
    with pytest.raises(ValueError, match="the design has 0 contrasts; name one"):
        limma(result, Design(full.samples, "~ donor + condition", levels=LEVELS))
    two = Design(
        full.samples,
        "~ donor + condition",
        levels=LEVELS,
        contrasts={"treated": "conditiontreated", "d1": "donorD1"},
    )
    with pytest.raises(ValueError, match="the design has 2 contrasts; name one with"):
        limma(result, two)
    with pytest.raises(KeyError, match="the design has no contrast 'stim'"):
        limma(result, design(), contrast="stim")
    # Without D1's samples, a contrast of D1 is not estimable.
    others = [i for i, d in enumerate(result.profiles["donor"].tolist()) if d != "D1"]
    rows = result.profiles["sample_id"][others].tolist()
    with pytest.raises(ValueError, match="the contrast 'd1' is not estimable"):
        limma(values[others], two, contrast="d1", sample_ids=rows)
    # One sample of each donor: the donors' columns span the samples.
    single = ["S0", "S2", "S5", "S6", "S9", "S10"]
    where = [result.profiles["sample_id"].tolist().index(i) for i in single]
    with pytest.raises(ValueError, match="no residual degrees of freedom: the design"):
        limma(values[where], design(), sample_ids=single)
    with pytest.raises(ValueError, match="none of the samples of the data is in"):
        limma(values, design(), sample_ids=[f"X{i}" for i in range(len(values))])
    with pytest.raises(
        ValueError, match=re.escape("data has 11 rows and the design 12")
    ):
        limma(values, design())
    with pytest.raises(TypeError, match="data must be PathwayScores, Activities"):
        limma({"a": 1.0}, design())
    with pytest.raises(TypeError, match="design must be a Design"):
        limma(result, full.samples)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weights must hold one weight per sample"):
        limma(result, design(), weights=np.ones(3))
    with pytest.raises(ValueError, match="data holds infinite values"):
        limma(np.where(values > 0.5, np.inf, values), design(result.profiles))
    with pytest.raises(ValueError, match="sample_ids must be unique"):
        limma(values, design(), sample_ids=["S0"] * len(values))
