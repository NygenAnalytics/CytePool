"""The expression values that per-sample scores and activities read: a samples x genes
array with its gene and sample ids, or a group of a cube as log-CPM or counts.

A cube's group is read as limma_voom reads it: the samples with at least ``min_cells``
cells and some counts, over the genes that edgeR's ``filterByExpr`` keeps
(:func:`~cytepool.normalization.filter_by_expression`), with the samples as one group
or grouped by a column of the cube's samples (``{"sample_groups": "condition"}``), as
log2 counts per million after TMM normalization with a prior count of 2, edgeR's
``cpm(log = TRUE)`` (Robinson and Oshlack 2010; Chen, Lun and Smyth 2016), or as
counts.

References:
    Robinson MD, Oshlack A (2010). A scaling normalization method for differential
    expression analysis of RNA-seq data. Genome Biology 11, R25.
    doi:10.1186/gb-2010-11-3-r25

    Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential
    expression analysis of RNA-Seq experiments using Rsubread and the edgeR
    quasi-likelihood pipeline. F1000Research 5, 1438. doi:10.12688/f1000research.8987.2
"""

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._columns import text_column
from ._gene_ids import is_missing
from ._group_fits import (
    FILTER_DEFAULTS,
    SAMPLE_GROUPS,
    check_sample_groups,
    filter_setting,
)
from ._masked_arrays import Missing, real_array, reject_masked, unmasked_array
from ._messages import quoted
from ._options import checked_choice, checked_flag, integral_number
from .cube import Cube
from .normalization import filter_by_expression, log2_cpm, tmm_factors
from .table import Table

FloatArray = npt.NDArray[np.float64]

VALUES = ("log2_cpm", "counts")
# As limma_voom leaves samples out, and edgeR's cpm(log = TRUE) shifts counts.
MIN_CELLS = 10
PRIOR_COUNT = 2.0


class ExpressionValues(NamedTuple):
    """The values read, with their samples and genes."""

    values: FloatArray  # samples x genes, NaN where missing
    integer: bool  # whether the values were stored as integers
    samples: Table
    gene_ids: list[str | None]  # one per column; None for a gene without an id
    settings: dict[str, Any]  # how a cube's values were taken
    cube: Cube | None = None  # the cube whose group was read


def expression_values(
    data: Any,
    id_column: Any,
    feature_ids: Any,
    sample_ids: Any,
    group: Any,
    values: Any,
    min_cells: Any,
    expression_filter: Any,
    *,
    missing: Missing = "NaN",
    matched_to: str = "the gene sets",
    other_forms: str = "",
) -> ExpressionValues:
    """The values to score, samples x genes, with their samples and gene ids.

    Args:
        data: A 2-D array of values, samples x genes, or a Cube.
        id_column: For a cube, the column of its features table with the ids
            (``"feature_id"`` by default).
        feature_ids: For an array, one id per column.
        sample_ids: For an array, one id per row; by default "1", "2", ...
        group: The group of a cube; None for its only one.
        values: For a cube, ``"log2_cpm"`` or ``"counts"``.
        min_cells: For a cube, the fewest cells a sample needs in the group.
        expression_filter: For a cube, whether to keep only the genes that
            filter_by_expression keeps: True or False, or a mapping of its settings,
            which may name a column of the cube's samples as ``sample_groups``.
        missing: How the values mark missing ones, for messages.
        matched_to: What the gene ids are matched to, for messages, such as "the
            gene sets".
        other_forms: The other forms that ``data`` may take, for messages, such as
            ", a DifferentialExpressionResult".
    """
    reject_masked(feature_ids, "feature_ids", missing="None")
    reject_masked(sample_ids, "sample_ids")
    reject_masked(id_column, "id_column")
    if isinstance(data, Cube):
        if sample_ids is not None:
            raise ValueError(
                "sample_ids applies to an array; a cube's samples have their ids"
            )
        if feature_ids is not None:
            raise ValueError(
                "feature_ids applies to an array; a cube's features are named by a "
                "column of its features table, which id_column names (add one with "
                "Cube.with_columns(features=...))"
            )
        return _cube_values(
            data, id_column, group, values, min_cells, expression_filter, matched_to
        )
    defaults = (
        ("group", group is None),
        ("values", isinstance(values, str) and values == "log2_cpm"),
        ("id_column", isinstance(id_column, str) and id_column == "feature_id"),
        (
            "min_cells",
            isinstance(min_cells, int)
            and not isinstance(min_cells, bool)
            and min_cells == MIN_CELLS,
        ),
        ("expression_filter", expression_filter is True),
    )
    for name, default in defaults:
        if not default:
            raise ValueError(
                f"{name} applies to a cube, and data is a {type(data).__name__}"
            )
    array = unmasked_array(data, "data", missing=missing)
    integer = array.dtype.kind in "iu"
    try:
        matrix = real_array(array, "data", boolean=False, objects=False)
    except TypeError:
        raise TypeError(
            f"data must hold real numbers, samples x genes, or be a Cube{other_forms}; "
            f"got dtype {array.dtype}"
        ) from None
    if matrix.ndim != 2:
        raise ValueError(
            "data must be a 2-D array of shape (n_samples, n_genes), or a "
            f"Cube{other_forms}; got shape {matrix.shape}"
        )
    n_samples, n_genes = matrix.shape
    if n_samples == 0 or n_genes == 0:
        raise ValueError(
            f"data must have at least one sample and one gene, got shape {matrix.shape}"
        )
    if np.isinf(matrix).any():
        raise ValueError("data holds infinite values; mark missing values with NaN")
    if feature_ids is None:
        raise ValueError(
            "feature_ids is needed with an array: one id per column, by which "
            f"{matched_to} name the genes"
        )
    if isinstance(feature_ids, str):
        raise TypeError(
            "feature_ids gives one id per column of an array; id_column names a "
            "column of a cube's features table"
        )
    ids = _gene_id_list(feature_ids, n_genes, "feature_ids")
    _check_unique(ids, "feature_ids", matched_to)
    if sample_ids is None:
        names = [str(i + 1) for i in range(n_samples)]
    else:
        names = _sample_id_list(sample_ids, n_samples)
    return ExpressionValues(
        np.ascontiguousarray(matrix, dtype=np.float64),
        integer,
        Table({"sample_id": text_column(names)}),
        ids,
        {},
    )


PROFILE_ID = "profile_id"


def profile_table(samples: Table, what: str) -> Table:
    """The profiles of scores read from a sample table: ``profile_id``, the sample
    ids, then the sample table's columns.

    Raises:
        ValueError: If the sample table has a column ``profile_id``.
    """
    if PROFILE_ID in samples:
        raise ValueError(
            f"the sample table has a column {PROFILE_ID!r}, the name {what} give the "
            "ids of their profiles; rename it"
        )
    return Table({PROFILE_ID: samples["sample_id"], **samples.to_dict()})


def _gene_id_list(values: Any, n: int, name: str) -> list[str | None]:
    """One gene id per gene, None where missing.

    Raises:
        TypeError: If ``values`` is not a sequence, or an id is neither a string nor
            missing.
        ValueError: If there is not one id per gene.
    """
    if isinstance(values, np.ndarray):
        found = values.tolist()
    else:
        try:
            found = list(values)
        except TypeError:
            raise TypeError(
                f"{name} must be a sequence of gene ids, got {type(values).__name__}"
            ) from None
    if len(found) != n:
        raise ValueError(f"{name} must hold one id per gene ({n}), got {len(found)}")
    ids: list[str | None] = []
    for value in found:
        if isinstance(value, str) and value:
            ids.append(str(value))
        elif is_missing(value):
            ids.append(None)
        else:
            raise TypeError(
                f"{name} must hold gene ids as strings, got {type(value).__name__} "
                f"{value!r}; convert numeric ids, such as Entrez ids, to text first"
            )
    return ids


def _cube_values(
    cube: Cube,
    id_column: Any,
    group: Any,
    values: Any,
    min_cells: Any,
    expression_filter: Any,
    matched_to: str,
) -> ExpressionValues:
    """A cube group's log-CPM or counts, of its samples with enough cells and counts,
    over the genes kept by filterByExpr."""
    kind = checked_choice(values, "values", VALUES)
    fewest = integral_number(min_cells, "min_cells")
    if fewest < 0:
        raise ValueError(f"min_cells must be at least 0, got {fewest}")
    if isinstance(expression_filter, Mapping):
        filter_settings = filter_setting(expression_filter)
        check_sample_groups({"expression_filter": filter_settings}, cube.samples)
        filtered: Any = dict(filter_settings or {})
    else:
        filtered = checked_flag(expression_filter, "expression_filter")
        filter_settings = dict(FILTER_DEFAULTS) if filtered else None
    if group is None:
        if cube.n_groups != 1:
            raise ValueError(
                f"the cube has {cube.n_groups} groups; name the one to score with "
                f"group, one of {quoted(cube.groups['group_id'].tolist())}"
            )
        group = cube.groups["group_id"][0]
    if not isinstance(group, str):
        reject_masked(group, "group")
        raise TypeError(f"group must be a group id (str), got {type(group).__name__}")
    counts = cube.layer("counts", group)
    used = np.asarray(counts.sum(axis=1, dtype=np.float64) > 0.0)
    if cube.n_cells is not None:
        position = cube.groups["group_id"].tolist().index(group)
        used &= np.asarray(cube.n_cells)[:, position] >= fewest
    rows = np.flatnonzero(used)
    if rows.size == 0:
        raise ValueError(
            f"no sample of group {group!r} has counts and at least {fewest} cells"
        )
    kept = counts[rows]
    genes = np.ones(kept.shape[1], dtype=bool)
    if filter_settings is not None:
        numbers = {
            name: value
            for name, value in filter_settings.items()
            if name != SAMPLE_GROUPS
        }
        if SAMPLE_GROUPS in filter_settings:
            labels = cube.samples[filter_settings[SAMPLE_GROUPS]][rows]
        else:
            labels = np.zeros(rows.size)
        genes = filter_by_expression(kept, sample_groups=labels, **numbers)
        if not genes.any():
            raise ValueError(
                f"no gene of group {group!r} passes filter_by_expression; pass "
                "expression_filter=False to score every gene"
            )
    kept = kept[:, genes]
    if kind == "log2_cpm":
        matrix = log2_cpm(kept, norm_factors=tmm_factors(kept), prior_count=PRIOR_COUNT)
        integer = False
    else:
        matrix = np.asarray(kept, dtype=np.float64)
        integer = kept.dtype.kind in "iu"
    if not isinstance(id_column, str):
        raise TypeError(
            "id_column must name a column of the cube's features table (str), got "
            f"{type(id_column).__name__}"
        )
    features = cube.features
    if id_column not in features:
        raise KeyError(
            f"the cube's features table has no column {id_column!r}; its columns "
            f"are {quoted(list(features.columns))}"
        )
    label = f"the features column {id_column!r}"
    ids = _gene_id_list(features[id_column], cube.n_features, label)
    ids = [ids[j] for j in np.flatnonzero(genes).tolist()]
    _check_unique(ids, label, matched_to)
    settings = {
        "group": group,
        "values": kind,
        "min_cells": fewest,
        "expression_filter": filtered,
    }
    return ExpressionValues(
        np.ascontiguousarray(matrix, dtype=np.float64),
        integer,
        cube.samples.take(rows),
        ids,
        settings,
        cube,
    )


def _check_unique(ids: list[str | None], name: str, matched_to: str) -> None:
    """Refuse an id that names two of the genes scored."""
    seen: set[str] = set()
    for gene in ids:
        if gene is not None:
            if gene in seen:
                raise ValueError(
                    f"{name} names two of the genes scored {gene!r}; each gene needs "
                    f"an id of its own to be matched to {matched_to}: keep one gene "
                    f"per id, or match {matched_to} by other ids"
                )
            seen.add(gene)


def _sample_id_list(values: Any, n: int) -> list[str]:
    array = unmasked_array(values, "sample_ids")
    if array.ndim != 1 or array.size != n:
        raise ValueError(
            f"sample_ids must hold one id per sample ({n}), got shape {array.shape}"
        )
    names = []
    for value in array.tolist():
        if is_missing(value):
            raise ValueError("sample_ids has a missing id; give every sample one")
        names.append(str(value))
    if len(set(names)) != n:
        raise ValueError("sample_ids must be unique")
    return names
