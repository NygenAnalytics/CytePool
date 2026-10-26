"""Over-representation analysis: gene sets among the genes called significant.

:func:`over_representation` asks of each gene set whether the genes called significant
in an analysis hold more of its genes than chance would give, by the one-sided
hypergeometric test against a universe of genes, as limma's ``kegga`` and ``goana``
compute it with their defaults (Young et al. 2010). From a differential-expression
result it tests the genes that go up and the genes that go down separately, as
``kegga`` and ``goana`` test a fit of limma, and the genes that change in either
direction together. Results match limma 3.62.1 except where noted under "Deliberate
differences".

The test:
    The universe holds the N genes that could have been called significant. Of a gene
    set, K genes are in the universe; n genes of the universe are selected, and k of
    them are in the set. Were the n genes drawn at random from the universe, without
    regard to the set, k would follow the hypergeometric distribution of n draws
    without replacement from N genes of which K are in the set, whose mean K n / N is
    the expected overlap. The p-value is P(X >= k), the one-sided p-value of Fisher's
    (1922) exact test of the 2 x 2 table of the genes of the universe, in the set or
    not and selected or not (Rivals et al. 2007); the hypergeometric test of
    functional categories goes back to Tavazoie et al. (1999). It is computed by
    :func:`~cytepool.stats.distributions.hypergeom_sf`, which keeps its relative
    accuracy far into the tail. The adjusted p-values are Benjamini and Hochberg's (1995), over
    the sets of each list of selected genes; limma's ``kegga`` and ``goana`` report
    p-values only.

The genes and the universe:
    - From a :class:`~cytepool.differential_expression.DifferentialExpressionResult`
      (one group and contrast) or any table of results, the genes whose adjusted
      p-value is below ``alpha`` are selected, as ``kegga`` selects the genes of a fit
      whose adjusted p-value is below its ``FDR`` (strictly, found by probing): up
      where the log2 fold change is positive, down where it is negative, and in either
      direction for both lists together. The universe is the genes tested: those
      with a p-value. A gene id on several rows, such as a symbol shared by two
      features, is one gene, in the universe when any of its rows was tested and
      selected up or down when any of its rows is, so that it can be in both lists,
      as ``kegga`` counts it; rows without an id are left out, as ``kegga`` leaves
      them out (both found by probing).
    - From a list of gene ids: the list is the selection, and the universe must be
      given.
    - A universe given replaces the default. Selected genes and genes of a set outside
      it are left out, and genes of the universe that are in no set stay in it, as in
      ``kegga`` by default (``restrict.universe = FALSE``).

    The universe matters: the genes of the genome, or of the sets, include genes that
    could not have been selected, such as genes not expressed or filtered out before
    the tests, and make p-values too small (Timmons et al. 2015; Wijesooriya et al.
    2022). The default, the genes tested, is the universe that the test assumes.

What the test assumes:
    The test takes genes as independent draws. The genes of a set are often
    correlated, as co-regulated genes are, which makes its p-values too small
    (Goeman and Buehlmann 2007); :func:`~cytepool.stats.gene_set_tests.camera` allows
    for the correlation. The test also depends on the cutoff ``alpha`` and ignores the
    size of each change. In RNA-seq data, long and highly expressed genes are called
    significant more often, which biases the test towards sets of such genes (Young et
    al. 2010); the adjustment of ``kegga`` and ``goana`` for such a trend
    (``trend``, Wallenius' noncentral hypergeometric distribution) is not
    implemented.

Deliberate differences from limma:
    - A set without genes in the universe has a row with no p-value (NaN), which the
      adjustment leaves out; ``kegga`` leaves the set out of its table.
    - ``kegga``'s method for a fit of limma keeps the genes without a p-value in the
      universe (found by probing); here a gene without a p-value was not tested and is
      left out, as the test assumes.
    - ``kegga``'s method for a fit adjusts the p-values of the rows with an id only,
      leaving the others out of the Benjamini-Hochberg adjustment, so that the genes
      it selects depend on which rows have an id, such as a symbol (found by
      probing). Here the adjusted p-values of the results are used as they are, so
      that the genes selected are those the analysis called significant.

References:
    Young MD, Wakefield MJ, Smyth GK, Oshlack A (2010). Gene ontology analysis for
    RNA-seq: accounting for selection bias. Genome Biology 11, R14.
    doi:10.1186/gb-2010-11-2-r14

    Fisher RA (1922). On the interpretation of chi-squared from contingency tables,
    and the calculation of P. Journal of the Royal Statistical Society 85, 87-94.
    doi:10.2307/2340521

    Rivals I, Personnaz L, Taing L, Potier MC (2007). Enrichment or depletion of a GO
    category within a class of genes: which test? Bioinformatics 23, 401-407.
    doi:10.1093/bioinformatics/btl633

    Tavazoie S, Hughes JD, Campbell MJ, Cho RJ, Church GM (1999). Systematic
    determination of genetic network architecture. Nature Genetics 22, 281-285.
    doi:10.1038/10343

    Timmons JA, Szkop KJ, Gallagher IJ (2015). Multiple sources of bias confound
    functional enrichment analysis of global -omics data. Genome Biology 16, 186.
    doi:10.1186/s13059-015-0761-7

    Wijesooriya K, Jadaan SA, Perera KL, Kaur T, Ziemann M (2022). Urgent need for
    consistent standards in functional enrichment analysis. PLoS Computational
    Biology 18, e1009935. doi:10.1371/journal.pcbi.1009935

    Goeman JJ, Buehlmann P (2007). Analyzing gene expression data in terms of gene sets:
    methodological issues. Bioinformatics 23, 980-987. doi:10.1093/bioinformatics/btm051

    Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical
    and powerful approach to multiple testing. Journal of the Royal Statistical
    Society B 57, 289-300. doi:10.1111/j.2517-6161.1995.tb02031.x
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from ._columns import is_data_frame, text_column
from ._gene_ids import gene_ids, is_missing
from ._masked_arrays import real_array, reject_masked
from ._messages import listing, quoted
from ._options import real_number
from .differential_expression import DifferentialExpressionResult
from .gene_sets import GeneSets
from .stats.distributions import hypergeom_sf
from .stats.multiple_testing import adjust_p_values
from .table import Table

__all__ = ["over_representation"]

_SEPARATOR = ", "  # between the genes of an overlap


def over_representation(
    genes: Any,
    gene_sets: Mapping[str, Any],
    *,
    group: str | None = None,
    contrast: str | None = None,
    universe: npt.ArrayLike | None = None,
    alpha: float = 0.05,
    id_column: str = "feature_id",
    p_value_column: str = "adjusted_p_value",
    fold_change_column: str | None = "log2_fold_change",
) -> Table:
    """Test each gene set for over-representation among the genes called significant.

    The one-sided hypergeometric test of limma's ``kegga`` and ``goana`` (Young et al.
    2010): whether the selected genes hold more of a set's genes than random draws
    from the universe would. The module docstring gives the method, how the genes and
    the universe are read, and the deliberate differences from limma.

    Args:
        genes: The genes to test: a
            :class:`~cytepool.differential_expression.DifferentialExpressionResult`,
            with ``group`` and ``contrast``; a table of results, a ``Table`` or a
            pandas DataFrame, with a column of gene ids, a column of adjusted p-values
            and, for directions, a column of fold changes; or a list of the ids of the
            selected genes, with ``universe``.
        gene_sets: A :class:`~cytepool.gene_sets.GeneSets`, such as one read by
            :func:`~cytepool.gene_sets.read_gmt`, or a mapping of set name to gene ids.
        group: The group (cell type) of a ``DifferentialExpressionResult``; it may be
            left out when the result has one.
        contrast: The contrast of a ``DifferentialExpressionResult``; it may be left
            out when the result tests one.
        universe: The ids of the genes that could have been selected. By default, for
            a result or a table, the genes tested; a list of genes needs one.
        alpha: The adjusted p-value below which a gene is selected, in (0, 1). Not
            used for a list of genes.
        id_column: The column of gene ids, such as ``"feature_id"`` or ``"symbol"``.
        p_value_column: The column of the p-values that select genes, such as
            ``"adjusted_p_value"`` or ``"adjusted_p_value_across_groups"``.
        fold_change_column: The column whose sign gives the direction of a selected
            gene, or None to test the selected genes without directions.

    Returns:
        A table with a row for each gene set and each list of selected genes. For a
        result or a table with fold changes, the rows hold the sets in the order of
        ``gene_sets`` for the genes up, then for those down, then for those in either
        direction; for a list of genes, or without fold changes, one row per set.
        Columns: ``gene_set``;
        ``direction``, ``"up"``, ``"down"`` or ``"either"``, or None for genes
        selected without a direction; ``n_genes``, the set's genes in the universe
        (K); ``overlap``, how many of them are selected (k); ``expected``, the
        expected overlap K n / N; ``p_value``, P(X >= k); ``adjusted_p_value``,
        Benjamini-Hochberg over the sets of the same direction; ``n_selected``, the
        selected genes in the universe (n); ``n_universe``, the genes of the
        universe (N); and ``overlap_genes``, the selected genes of the set, in the
        set's order, separated by ``", "``. A set without genes in the universe has
        NaN p-values.

    Raises:
        TypeError: If an argument has the wrong type, gene ids are not strings, a
            column has the wrong type, or an argument is or holds a numpy masked
            array.
        ValueError: If ``gene_sets`` is empty, ``alpha`` is not in (0, 1), a table
            lacks a column, has p-values outside [0, 1] or misses the fold change of a
            gene with a p-value, a list of genes is given without a universe,
            ``group`` or ``contrast`` is given without a ``DifferentialExpressionResult``
            or left out of one that has several, or no gene of any set, or no
            selected gene, is in the universe.
        KeyError: If the result has no table of the group and contrast; the message
            says why.
    """
    reject_masked(gene_sets, "gene_sets")
    collection = _gene_sets(gene_sets)
    level = _alpha(alpha)
    columns = (
        _column_name(id_column, "id_column"),
        _column_name(p_value_column, "p_value_column"),
        None
        if fold_change_column is None
        else _column_name(fold_change_column, "fold_change_column"),
    )
    if isinstance(genes, DifferentialExpressionResult):
        chosen = _group_and_contrast(genes, group, contrast)
        selections, tested = _selections(genes.table(*chosen), level, *columns)
    else:
        if group is not None or contrast is not None:
            raise ValueError(
                "group and contrast choose a table of a DifferentialExpressionResult; "
                f"genes is a {type(genes).__name__}"
            )
        if isinstance(genes, Table) or is_data_frame(genes):
            table = genes if isinstance(genes, Table) else Table(genes)
            selections, tested = _selections(table, level, *columns)
        else:
            if isinstance(genes, Mapping):
                raise TypeError(
                    "genes must be a DifferentialExpressionResult, a table of results "
                    "or a list of gene ids, got a mapping; test each list of genes in "
                    "its own call"
                )
            selections = {None: list(dict.fromkeys(gene_ids(genes, "genes")[0]))}
            if universe is None:
                raise ValueError(
                    "a list of genes is tested against a universe: give universe, "
                    "the ids of the genes that could have been selected, such as "
                    "every gene tested"
                )
            tested = []
    if universe is not None:
        population = list(dict.fromkeys(gene_ids(universe, "universe")[0]))
        if not population:
            raise ValueError("universe holds no gene id")
    else:
        population = tested
        if not population:
            raise ValueError(
                f"no gene of the table has a p-value in {columns[1]!r}, so no gene "
                "was tested"
            )
    return _tested_sets(collection, selections, population)


# The test.


def _tested_sets(
    collection: GeneSets,
    selections: dict[str | None, list[str]],
    population: list[str],
) -> Table:
    """The table of every set and list of selected genes."""
    index = {gene: position for position, gene in enumerate(population)}
    n_universe = len(population)
    names = list(collection)
    members = [[gene for gene in collection[name] if gene in index] for name in names]
    sizes = np.array([len(inside) for inside in members], dtype=np.int64)
    if not sizes.any():
        raise ValueError(
            "no gene of any set is in the universe; the sets and the genes tested "
            "must name genes the same way (for example both by symbol, or both by "
            "Ensembl id): the sets name genes such as "
            f"{_example_ids(collection)}, the universe genes such as "
            f"{quoted(population[:3])}"
        )
    positions = np.array(
        [index[gene] for inside in members for gene in inside], dtype=np.intp
    )
    set_of_entry = np.repeat(np.arange(len(names)), sizes)

    masks = {}
    for direction, selected in selections.items():
        mask = np.zeros(n_universe, dtype=bool)
        mask[[index[gene] for gene in selected if gene in index]] = True
        masks[direction] = mask
    given = sum(len(selected) for selected in selections.values())
    if given and not any(mask.any() for mask in masks.values()):
        raise ValueError(
            f"none of the {given} selected genes is in the universe; the genes and "
            "the universe must name genes the same way (for example both by symbol, "
            "or both by Ensembl id)"
        )

    columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "gene_set",
            "direction",
            "n_genes",
            "overlap",
            "expected",
            "p_value",
            "adjusted_p_value",
            "n_selected",
            "n_universe",
            "overlap_genes",
        )
    }
    for direction, mask in masks.items():
        n_selected = int(np.count_nonzero(mask))
        chosen = mask[positions]
        overlap = np.bincount(set_of_entry[chosen], minlength=len(names))
        with np.errstate(invalid="ignore"):
            p_value = np.asarray(
                hypergeom_sf(
                    overlap - 1.0, float(n_universe), sizes.astype(float), n_selected
                ),
                dtype=np.float64,
            )
        p_value[sizes == 0] = np.nan
        texts = [""] * len(names)
        for s in np.flatnonzero(overlap):
            texts[s] = _SEPARATOR.join(gene for gene in members[s] if mask[index[gene]])
        columns["gene_set"] += names
        columns["direction"] += [direction] * len(names)
        columns["n_genes"].append(sizes)
        columns["overlap"].append(overlap.astype(np.int64))
        columns["expected"].append(sizes * (n_selected / n_universe))
        columns["p_value"].append(p_value)
        columns["adjusted_p_value"].append(adjust_p_values(p_value, "bh"))
        columns["n_selected"].append(np.full(len(names), n_selected, dtype=np.int64))
        columns["n_universe"].append(np.full(len(names), n_universe, dtype=np.int64))
        columns["overlap_genes"] += texts
    return Table(
        {
            name: text_column(values)
            if name in ("gene_set", "direction", "overlap_genes")
            else np.concatenate(values)
            for name, values in columns.items()
        }
    )


# The selected genes of a table.


def _selections(
    table: Table,
    alpha: float,
    id_column: str,
    p_value_column: str,
    fold_change_column: str | None,
) -> tuple[dict[str | None, list[str]], list[str]]:
    """The selected genes of each direction, and the genes tested, in table order."""
    needed = [id_column, p_value_column]
    if fold_change_column is not None:
        needed.append(fold_change_column)
    missing = [column for column in needed if column not in table]
    if missing:
        raise ValueError(
            f"the table has no column {quoted(missing)}; its columns are "
            f"{quoted(list(table.columns))}. Name the columns of the gene ids, the "
            "p-values and the fold changes with id_column, p_value_column and "
            "fold_change_column; a pandas DataFrame keeps its row index, such as gene "
            "ids, as a column only when the index has a name"
        )
    ids = table[id_column]
    if ids.dtype.kind != "O":
        raise TypeError(
            f"the column {id_column!r} must hold gene ids as text, got dtype "
            f"{ids.dtype}; convert numeric ids, such as Entrez ids, to text first"
        )
    named = np.array([not is_missing(gene) for gene in ids.tolist()], dtype=bool)
    p = _real_column(table, p_value_column)
    with np.errstate(invalid="ignore"):
        outside = (p < 0.0) | (p > 1.0)
    if np.any(outside):
        raise ValueError(
            f"the column {p_value_column!r} holds values outside [0, 1], such as "
            f"{p[outside][0]}; it must hold p-values"
        )
    tested = named & ~np.isnan(p)
    with np.errstate(invalid="ignore"):
        selected = tested & (p < alpha)
    if fold_change_column is None:
        lists: dict[str | None, np.ndarray] = {None: selected}
    else:
        change = _real_column(table, fold_change_column)
        unknown = tested & np.isnan(change)
        if np.any(unknown):
            raise ValueError(
                f"the column {fold_change_column!r} is missing for "
                f"{int(np.count_nonzero(unknown))} gene(s) with a p-value, such as "
                f"{ids[unknown][0]!r}; their direction is unknown"
            )
        up = selected & (change > 0.0)
        down = selected & (change < 0.0)
        lists = {"up": up, "down": down, "either": up | down}
    selections = {
        direction: list(dict.fromkeys(ids[rows].tolist()))
        for direction, rows in lists.items()
    }
    return selections, list(dict.fromkeys(ids[tested].tolist()))


def _real_column(table: Table, column: str) -> np.ndarray:
    try:
        values = real_array(table[column], f"the column {column!r}", objects=False)
    except TypeError:
        raise TypeError(
            f"the column {column!r} must hold numbers, got dtype {table[column].dtype}"
        ) from None
    return values


# Arguments.


def _gene_sets(gene_sets: Any) -> GeneSets:
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


def _group_and_contrast(
    result: DifferentialExpressionResult, group: Any, contrast: Any
) -> tuple[str, str]:
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


def _alpha(value: Any) -> float:
    reject_masked(value, "alpha")
    number = real_number(value, "alpha")
    if not 0.0 < number < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {number}")
    return number


def _column_name(value: Any, name: str) -> str:
    reject_masked(value, name)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a column name (a string), got {value!r}")
    return value


def _example_ids(collection: GeneSets) -> str:
    examples: list[str] = []
    for genes in collection.values():
        for gene in genes:
            if gene not in examples:
                examples.append(gene)
            if len(examples) == 3:
                return quoted(examples)
    return listing([repr(gene) for gene in examples])
