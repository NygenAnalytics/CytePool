"""Genes ranked by a statistic, gene sets matched to them, and the running sum of gene
set enrichment analysis along them.

Gene set enrichment analysis, its figures and over-representation analysis read their
genes the same way: from a group and contrast of a
:class:`~cytepool.DifferentialExpressionResult`, from a table of results, or, for a
ranking, from a mapping of gene id to statistic, and their gene sets from a
:class:`~cytepool.GeneSets` or a mapping of set name to genes.

The ranking:
    Genes are ranked by decreasing statistic. Genes with equal statistics keep the
    order in which they are given, as R's ``sort(decreasing = TRUE)``, with which
    fgsea ranks them, keeps them (found by probing R 4.4.3). Rows without a gene id
    (None, NaN or an empty string) and rows without a statistic (NaN) are left out;
    an infinite statistic, or a gene id on two rows, is refused.

The running sum:
    For a set of m of the N ranked genes, at ranks p_1 < ... < p_m, with weights
    w_i = |r_i|^p of their statistics r_i (Subramanian et al. 2005), the running sum
    after the set's i-th gene is the share of the set's weight reached, minus the
    share of the N - m other genes passed:

        top_i = (w_1 + ... + w_i) / W - (p_i - i) / (N - m),  W = w_1 + ... + w_m,

    and just before it, bottom_i = top_i - w_i / W; when every weight is zero, each
    gene weighs 1 / m. The enrichment score is the extreme of the walk, the largest
    top or the smallest bottom (fgsea's ``calcGseaStat``; Korotkevich et al. 2021).
    The partial sums and W are rounded once from their exact values, as R's
    ``cumsum`` and ``sum``, which accumulate in extended precision, round them
    (found by probing R 4.4.3), so that ties between steps of the walk, which decide
    the leading edge, fall as in fgsea.

References:
    Subramanian A, Tamayo P, Mootha VK, Mukherjee S, Ebert BL, Gillette MA, Paulovich
    A, Pomeroy SL, Golub TR, Lander ES, Mesirov JP (2005). Gene set enrichment
    analysis: a knowledge-based approach for interpreting genome-wide expression
    profiles. Proceedings of the National Academy of Sciences 102, 15545-15550.
    doi:10.1073/pnas.0506580102

    Korotkevich G, Sukhov V, Budin N, Shpak B, Artyomov MN, Sergushichev A (2021).
    Fast gene set enrichment analysis. bioRxiv 060012, version 3.
    doi:10.1101/060012
"""

from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from ._columns import is_data_frame
from ._gene_ids import is_missing
from ._jit import jit
from ._masked_arrays import real_array, reject_masked
from ._messages import quoted
from ._summation import two_sum
from .differential_expression import DifferentialExpressionResult
from .gene_sets import GeneSets
from .table import Table

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# The names of statistics, for the axes of figures.
_STATISTIC_LABELS = {"t": "Moderated t-statistic", "statistic": "Statistic"}


class RankedGenes(NamedTuple):
    """Genes in decreasing order of their statistic."""

    ids: list[str]
    values: FloatArray
    label: str  # what the statistic is, for the axes of figures


def gene_set_collection(gene_sets: Any) -> GeneSets:
    """``gene_sets`` as a GeneSets, refusing an empty collection."""
    if isinstance(gene_sets, GeneSets):
        collection = gene_sets
    elif isinstance(gene_sets, Mapping):
        collection = GeneSets(gene_sets)
    else:
        raise TypeError(
            "gene_sets must be a GeneSets or a mapping of set name to gene ids, such "
            f"as {{'interferon': ['ISG15', 'IFIT1']}}; got {type(gene_sets).__name__}"
        )
    if not len(collection):
        raise ValueError("gene_sets is empty; give at least one gene set")
    return collection


def group_and_contrast(
    result: DifferentialExpressionResult, group: Any, contrast: Any
) -> tuple[str, str]:
    """The group and contrast of a result that are meant: those given, or its only
    ones."""
    chosen = []
    for value, name, options in (
        (group, "group", list(result.groups)),
        (contrast, "contrast", list(result.contrasts.names)),
    ):
        if value is None:
            if len(options) != 1:
                raise ValueError(
                    f"the result has {len(options)} {name}s; name one with {name}, "
                    f"one of {quoted(options)}"
                )
            value = options[0]
        elif not isinstance(value, str):
            reject_masked(value, name)
            raise TypeError(f"{name} must be a string, got {type(value).__name__}")
        chosen.append(value)
    return chosen[0], chosen[1]


def ranked_genes(
    statistics: Any,
    *,
    group: str | None,
    contrast: str | None,
    id_column: str,
    statistic_column: str,
) -> RankedGenes:
    """The genes of ``statistics`` in decreasing order of their statistic.

    Args:
        statistics: A DifferentialExpressionResult, with ``group`` and ``contrast``;
            a Table or pandas DataFrame with the columns ``id_column`` and
            ``statistic_column``; a pandas Series of statistics indexed by gene id;
            or a mapping of gene id to statistic.
        group: The group of a DifferentialExpressionResult; None for its only one.
        contrast: Its contrast; None for its only one.
        id_column: The column of gene ids of a table.
        statistic_column: The column of statistics of a table.

    Raises:
        TypeError: If ``statistics`` is none of these, a gene id is not a string, a
            statistic is not a real number, or an argument is or holds a numpy masked
            array.
        ValueError: If ``group`` or ``contrast`` is given without a
            DifferentialExpressionResult or left out of one that has several, a
            table lacks a column, a statistic is infinite, a gene id is on two rows,
            or fewer than two genes have an id and a statistic.
        KeyError: If the result has no table of the group and contrast.
    """
    for value, name in (
        (id_column, "id_column"),
        (statistic_column, "statistic_column"),
    ):
        reject_masked(value, name)
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a column name (a string), got {value!r}")
    label = _STATISTIC_LABELS.get(statistic_column, statistic_column)
    if isinstance(statistics, DifferentialExpressionResult):
        table = statistics.table(*group_and_contrast(statistics, group, contrast))
        ids, values = _table_columns(table, id_column, statistic_column)
    else:
        if group is not None or contrast is not None:
            reject_masked(group, "group")
            reject_masked(contrast, "contrast")
            raise ValueError(
                "group and contrast choose a table of a DifferentialExpressionResult; "
                f"statistics is a {type(statistics).__name__}"
            )
        if isinstance(statistics, Table) or is_data_frame(statistics):
            table = statistics if isinstance(statistics, Table) else Table(statistics)
            ids, values = _table_columns(table, id_column, statistic_column)
        elif _is_series(statistics):
            ids = statistics.index.tolist()
            values = _real_values(statistics.to_numpy(), "statistics")
            label = str(statistics.name) if statistics.name is not None else "Statistic"
        elif isinstance(statistics, Mapping):
            reject_masked(statistics, "statistics", missing="NaN")
            ids = list(statistics.keys())
            values = _real_values(list(statistics.values()), "statistics")
            label = "Statistic"
        else:
            reject_masked(statistics, "statistics")
            raise TypeError(
                "statistics must be a DifferentialExpressionResult, a table of results "
                "(a Table or a pandas DataFrame), a pandas Series indexed by gene id or "
                "a mapping of gene id to statistic, got "
                f"{type(statistics).__name__}"
            )
    return _ranking(ids, values, label)


def matched_positions(genes: tuple[str, ...], index: Mapping[str, int]) -> IntArray:
    """The positions of a set's genes in the ranking, in the set's order; genes that
    are not ranked are left out, as fgsea leaves them out."""
    return np.array([index[gene] for gene in genes if gene in index], dtype=np.int64)


@jit
def running_sum(
    weights: FloatArray, positions: IntArray
) -> tuple[FloatArray, FloatArray]:
    """The running sum at and just before each gene of a set, fgsea's tops and
    bottoms.

    Args:
        weights: The weights |r|^p of all genes, in ranked order.
        positions: The 0-based positions of the set's genes, increasing; fewer than
            all the genes.

    Returns:
        ``(tops, bottoms)``, one value per gene of the set.
    """
    n = weights.size
    m = positions.size
    # The partial sums of the weights, each rounded once from the exact sum.
    partial = np.empty(m)
    high = 0.0
    low = 0.0
    for i in range(m):
        high, error = two_sum(high, weights[positions[i]])
        low += error
        partial[i] = high + low
    total = partial[m - 1] if m else 0.0
    tops = np.empty(m)
    bottoms = np.empty(m)
    for i in range(m):
        reached = (i + 1) / m if total == 0.0 else partial[i] / total
        tops[i] = reached - (positions[i] - i) / (n - m)
        step = 1.0 / m if total == 0.0 else weights[positions[i]] / total
        bottoms[i] = tops[i] - step
    return tops, bottoms


# Reading the statistics.


def _table_columns(
    table: Table, id_column: str, statistic_column: str
) -> tuple[list[Any], FloatArray]:
    missing = [name for name in (id_column, statistic_column) if name not in table]
    if missing:
        raise ValueError(
            f"the table has no column {quoted(missing)}; its columns are "
            f"{quoted(list(table.columns))}. Name the columns of the gene ids and of "
            "the statistics with id_column and statistic_column; a pandas DataFrame "
            "keeps its row index, such as gene ids, as a column only when the index "
            "has a name"
        )
    ids = table[id_column]
    if ids.dtype.kind != "O":
        raise TypeError(
            f"the column {id_column!r} must hold gene ids as text, got dtype "
            f"{ids.dtype}; convert numeric ids, such as Entrez ids, to text first"
        )
    try:
        values = real_array(
            table[statistic_column],
            f"the column {statistic_column!r}",
            boolean=False,
            objects=False,
        )
    except TypeError:
        raise TypeError(
            f"the column {statistic_column!r} must hold numbers, got dtype "
            f"{table[statistic_column].dtype}"
        ) from None
    return ids.tolist(), values


def _real_values(values: Any, name: str) -> FloatArray:
    try:
        array = real_array(values, name, boolean=False, objects=False)
    except TypeError:
        raise TypeError(f"{name} must map gene ids to numbers") from None
    if array.ndim != 1:
        raise ValueError(f"{name} must hold one number per gene")
    return array


def _is_series(value: object) -> bool:
    """Whether ``value`` is a pandas Series, recognised without importing pandas."""
    return all(
        hasattr(value, name) for name in ("index", "to_numpy", "name", "dtype")
    ) and not hasattr(value, "columns")


def _ranking(ids: list[Any], values: FloatArray, label: str) -> RankedGenes:
    """The genes with an id and a statistic, in decreasing order of the statistic."""
    if len(ids) != values.size:
        raise ValueError("statistics must hold one statistic per gene id")
    named = np.empty(len(ids), dtype=bool)
    for i, gene in enumerate(ids):
        if isinstance(gene, str):
            named[i] = bool(gene)
        elif is_missing(gene):
            named[i] = False
        else:
            raise TypeError(
                "gene ids must be strings, got "
                f"{type(gene).__name__} {gene!r}; convert numeric ids, such as Entrez "
                "ids, to text first"
            )
    infinite = np.isinf(values)
    if infinite.any():
        where = int(np.flatnonzero(infinite)[0])
        raise ValueError(
            f"the statistic of {ids[where]!r} is {values[where]}; statistics must be "
            "finite, or NaN for a gene that was not tested"
        )
    kept = np.flatnonzero(named & ~np.isnan(values))
    genes = [str(ids[i]) for i in kept.tolist()]
    seen: dict[str, int] = {}
    for gene in genes:
        seen[gene] = seen.get(gene, 0) + 1
        if seen[gene] == 2:
            raise ValueError(
                f"the gene id {gene!r} is on more than one row; each gene needs one "
                "statistic: keep one row per gene id (for example the one with the "
                "largest absolute statistic), or rank the genes by another id"
            )
    if len(genes) < 2:
        raise ValueError(
            f"{len(genes)} gene(s) have an id and a statistic; a ranking needs at "
            "least two"
        )
    kept_values = values[kept]
    # Decreasing, with ties in the order given, as R's sort(decreasing = TRUE).
    order = np.argsort(-kept_values, kind="stable")
    return RankedGenes(
        [genes[i] for i in order.tolist()],
        np.ascontiguousarray(kept_values[order]),
        label,
    )
