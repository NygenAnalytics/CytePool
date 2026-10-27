"""limma_voom on published single-cell data against limma, from a fixture.

``tests/data/interferon_stimulated_pbmcs/limma.npz`` holds pseudobulk counts of the
megakaryocytes of Kang et al. (2018), GEO GSE96583: peripheral blood mononuclear cells
of 8 lupus patients, each in a control culture and one stimulated with interferon-beta,
16 samples in all. Four samples have fewer than 10 cells, among them both samples of
donor 107, so the fit leaves them out and the design's column for that donor is zero
in the samples left. The fixture has the counts of the 97 genes of MSigDB's Hallmark
interferon-alpha response (every Ensembl id of their symbols) and of 300 genes that
the full analysis tested, the cells behind every profile, and the results of the limma
User's Guide workflow that R ran on exactly these counts: the samples used, the
features tested, the aliased columns, the degrees of freedom, the span of voom's trend
(limma's ``chooseLowessSpan`` of the number of genes tested) and topTable.
``validation/interferon_stimulated_pbmcs.py`` writes it.
"""

from pathlib import Path

import numpy as np
import pytest

from cytepool import Cube, Design, limma_voom

DATA = Path(__file__).parent / "data" / "interferon_stimulated_pbmcs" / "limma.npz"
EXACT = 1e-10
# The columns of limma's topTable in the fixture, in order.
STATISTICS = (
    "log2_fold_change",
    "average_log2_cpm",
    "t",
    "p_value",
    "adjusted_p_value",
    "log_odds",
)


def scaled_difference(ours: np.ndarray, reference: np.ndarray, p_value: bool) -> float:
    """|ours - reference| over max(|reference|, 1), or for p-values over
    max(p, 1e-300) max(1, |log p|): about the error in log p relative to |log p|."""
    if p_value:
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(reference), 1.0)
    return float(np.max(np.abs(ours - reference) / scale))


def test_limma_voom_matches_limma_on_published_counts() -> None:
    with np.load(DATA) as data:
        fixture = {key: data[key] for key in data.files}
    group = str(fixture["group_id"])
    samples, features = (
        {
            key.removeprefix(f"{axis} "): values
            for key, values in fixture.items()
            if key.startswith(f"{axis} ")
        }
        for axis in ("sample", "feature")
    )
    cube = Cube(
        {"counts": fixture["counts"][np.newaxis]},
        samples=samples,
        groups={"group_id": [group]},
        features=features,
        n_cells=fixture["n_cells"][:, np.newaxis],
    )
    design = Design(
        cube.samples,
        "~ donor + condition",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim_vs_ctrl": "conditionstim"},
    )
    result = limma_voom(cube, design)

    # The workflow chooses voom's span from the number of genes tested; R computed
    # limma's chooseLowessSpan for them, and voom ran with that span.
    record = result.provenance["groups"][0]
    assert record["voom_span"] == pytest.approx(fixture["limma span"][0], rel=EXACT)
    assert result.samples["used"].tolist() == fixture["limma used"].tolist()
    assert record["aliased_columns"] == fixture["limma aliased"].tolist()
    assert result.summary["df_residual"][0] == fixture["limma df"][0]
    prior_df = result.summary["df_prior"][0]
    assert prior_df == pytest.approx(fixture["limma df"][1], rel=EXACT)

    table = result.table(group, "stim_vs_ctrl")
    tested = fixture["limma keep"]
    assert table["feature_id"].tolist() == features["feature_id"][tested].tolist()
    assert table["symbol"].tolist() == features["symbol"][tested].tolist()
    reference = fixture["limma table"]
    for row, column in enumerate(STATISTICS):
        difference = scaled_difference(
            table[column], reference[row], column.endswith("p_value")
        )
        assert difference <= EXACT, (column, difference)
