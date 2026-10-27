"""Converting cubes to and from AnnData and MuData, the data structures of scverse.

A cube holds a samples x features matrix for every group, and an AnnData (Virshup et
al. 2024) holds 2-D matrices only (since anndata 0.13, ``X`` and every layer must be
2-D). scverse therefore keeps pseudobulk profiles in one of two layouts, and CytePool
reads and writes both:

- a long AnnData, with one row per sample and group, as ``scanpy.get.aggregate``
  (Wolf, Angerer and Theis 2018) and ``decoupler.pp.pseudobulk`` (Badia-i-Mompel et
  al. 2022) make it: ``to_anndata`` and ``from_anndata``;
- a MuData (Bredikhin, Kats and Stegle 2022) with one modality per group and the
  samples as its observations, as LIANA's ``liana.multi.adata_to_views`` (Dimitrov et
  al. 2024) makes it: ``to_mudata`` and ``from_mudata``.

anndata and mudata are optional dependencies: install CytePool's ``anndata`` or
``mudata`` extra, or the packages themselves. Only ``to_anndata`` and ``to_mudata``
import them; ``from_anndata`` and ``from_mudata`` read the object they are given.

What an import takes:
    A cube's layers are sums over the cells of each sample and group, so an import
    takes only what is such a sum or gives one exactly: summed counts, the number of
    cells detecting each feature, and sums of normalized or log-normalized values.
    The source of a cube layer is ``X`` or a layer of the AnnData (of every modality,
    for a MuData), read as sums, except where its name says what scanpy or decoupler
    stored in it:

    - ``count_nonzero`` (scanpy) counts the cells with a nonzero value, and gives
      ``detected`` only;
    - ``psbulk_props`` (decoupler) is the fraction of cells with a nonzero value. It
      gives ``detected`` with the cell numbers: each fraction times its number of
      cells must be a whole number, to within the precision the fractions are stored
      in, and rounding it recovers the number of cells exactly;
    - ``mean``, ``var`` and ``median`` hold means, variances and medians, which are
      refused. A variance or a median gives no sum, and a mean times its number of
      cells gives the sum exactly only where the sum is a whole number; every tool
      that gives means gives sums too: aggregate with ``func="sum"`` (scanpy) or
      ``mode="sum"`` (decoupler).

    Counts and ``detected`` become integers, as ``from_cells`` makes them: uint32
    when every value fits, uint64 otherwise, and float64 for counts that are not
    whole numbers; sums of whole numbers that a tool stored as floats become
    integers. Counts are checked against the cells that ``detected`` counts: a sum
    of counts is positive exactly where a cell has a positive count, and a sum of
    whole counts (counts that come as integers, or from a recognised layout) is at
    least the number of such cells, which means, medians and counts of other cells
    break.

    A combination of sample and group without a row is an entry without cells,
    which only cell numbers can record: an import with such a combination and no
    cell numbers is refused. A row removed after pooling, as by decoupler's
    ``filter_samples`` or LIANA's ``min_cells``, is such a combination too, so its
    cells are then missing from merged groups and pooled samples of the cube:
    import the profiles before filtering them, and leave out samples with few cells
    in the analysis instead (``limma_voom`` has ``min_cells``).

Recognised layouts:
    The arguments left as None take their values from the layout of the data; a
    long AnnData that CytePool did not make still needs ``sample=`` and ``group=``,
    the columns of its labels.

    - CytePool's own export, which ``uns["cytepool"]`` marks (see "The record in
      uns" below): every layer, the cell numbers, the tables, the normalization and
      the label columns, as they were exported.
    - ``decoupler.pp.pseudobulk`` (an ``obs`` column ``psbulk_cells``, or
      ``psbulk_n_cells`` as older versions name it): counts from ``X``, cell numbers
      from that column, and ``detected`` from ``psbulk_props``. Whatever ``layers``
      gives, ``X`` must be a sum of counts: positive exactly where ``psbulk_props``
      is, which medians are not; where whole, at least the number of detecting
      cells that the fractions bound even without cell numbers, which maxima and
      means seldom are; and, row by row, at most the total of the pooled counts
      that decoupler computes, whatever its ``mode``, as ``psbulk_counts`` (less
      after a filter of the features, such as ``decoupler.pp.filter_by_expr``),
      without being that total over the cells, as means are.
    - ``scanpy.get.aggregate`` (an ``obs`` column ``n_obs_aggregated``): counts from
      the layer ``sum``, ``detected`` from ``count_nonzero`` when present, and cell
      numbers from ``n_obs_aggregated``.
    - ``liana.multi.adata_to_views``, for a MuData whose every modality has a
      ``psbulk_props`` layer: counts from ``X``, checked as decoupler's, and cell
      numbers, and so ``detected``, from ``uns["psbulk_stats"]`` when it holds them.
      LIANA keeps cell numbers only when its ``keep_stats`` finds decoupler's column
      ``psbulk_n_cells``, which decoupler 2 names ``psbulk_cells``, so they are
      usually missing: pass ``n_cells``, for example ``pandas.crosstab`` of the
      cells' sample and group labels.

    Counts that a recognised layout gives must be whole numbers; give ``layers`` to
    import counts that are not, or to read a ``sum`` of normalized values. Data of
    no recognised layout need ``layers``.

Export:
    ``to_anndata`` writes a row for every entry with cells (every entry of a cube
    without cell numbers), the rows of each group together, groups and samples in
    the cube's order. Entries without cells have no row, as in
    ``scanpy.get.aggregate``. ``obs`` holds ``sample_id`` and ``group_id``, as
    categorical columns, the other columns of the sample and group tables, and
    ``n_cells``; ``obs_names`` are ``<sample id>_<group id>``, or the row positions
    when those repeat. ``var`` holds the feature table, indexed by feature id, and
    each layer of the cube is a layer of the same name. ``X`` is left empty, so
    that no layer is stored twice, unless ``x_layer`` names a layer to copy into it:
    ``to_anndata(cube, x_layer="counts")`` gives tools that read ``X`` the counts.

    ``to_mudata`` writes a modality per group, named by the group id, whose
    observations are the samples with cells in that group (every sample, without
    cell numbers), named by sample id, with ``n_cells`` in its ``obs``, every layer
    under its own name, and ``X`` empty. Its ``var_names`` are ``<group id>:<feature
    id>``, as LIANA writes them, so that they are unique across modalities; ``var``
    holds the other columns of the feature table. The columns of the sample table
    are on the MuData's ``obs``. A MuData file cannot store a modality whose name
    holds ``/`` or NUL, or is ``.`` or ``..``: rename such a group first with
    ``Cube.regroup``.

    anndata writes only some columns, so table columns are written in a form it
    stores and reads back as they were: datetimes as ISO 8601 text (ISO 2019), ``NaT``
    where missing, timedeltas as int64 counts of their unit, and a text column without
    any value as float64 NaN. Column names have ``_`` in place of ``/`` and NUL,
    and ``.``, ``..`` and ``_index``, which anndata reserves, end in ``_``. A sample
    or group column whose name is taken is prefixed with ``sample_`` or ``group_``.

The record in uns:
    ``uns["cytepool"]`` (of the AnnData, or of the MuData) holds ``format``
    (``"cytepool.cube.anndata"`` or ``"cytepool.cube.mudata"``), ``format_version``
    (1), the cube's ``fingerprint``, and as JSON text its ``provenance`` and
    ``normalization``, its ``layers`` with the type of each, whether it has cell
    numbers (``n_cells``), its ``samples`` and ``groups`` tables, with the samples
    and groups that have no row, and the name, ``var`` column and type of each
    column of its ``features`` table. Only text and integers are stored, which
    every anndata writer keeps.

    Reading the record, an import gives back the cube: the same tables (column
    types included), layers of the same types and the same values, so the same
    fingerprint, unless the data were changed in between. A layer converted to
    another type is converted back when that keeps its values; a layer whose values
    its type no longer holds was changed, and is refused (give ``layers`` to read
    it as what it holds). Every argument overrides what the record gives. The
    tables of samples and groups come from the record, so ``obs`` columns added
    later are not read; features come from ``var``, so a subset of features gives a
    cube of those features. The provenance is the exported cube's, followed by a
    record of the import with the fingerprint that the export stored:
    ``cube.fingerprint() == adata.uns["cytepool"]["fingerprint"]`` says whether the
    AnnData still holds the exported cube.

Example:
    >>> import numpy as np
    >>> from cytepool.cube import from_cells
    >>> cube = from_cells(
    ...     np.array([[1, 0], [3, 2], [0, 4]]),
    ...     samples=["S1", "S1", "S2"],
    ...     groups=["B", "T", "B"],
    ...     layers=["counts", "detected"],
    ... )
    >>> adata = to_anndata(cube)  # doctest: +SKIP
    >>> adata.obs_names.tolist()  # doctest: +SKIP
    ['S1_B', 'S2_B', 'S1_T']
    >>> from_anndata(adata).fingerprint() == cube.fingerprint()  # doctest: +SKIP
    True

References:
    Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA (2024). anndata: access and store
    annotated data matrices. Journal of Open Source Software 9, 4371.
    doi:10.21105/joss.04371

    Bredikhin D, Kats I, Stegle O (2022). MUON: multimodal omics analysis framework.
    Genome Biology 23, 42. doi:10.1186/s13059-021-02577-8

    Wolf FA, Angerer P, Theis FJ (2018). SCANPY: large-scale single-cell gene expression
    data analysis. Genome Biology 19, 15. doi:10.1186/s13059-017-1382-0

    Badia-i-Mompel P, Velez Santiago J, Braunger J, Geiss C, Dimitrov D, Mueller-Dott S,
    Taus P, Dugourd A, Holland CH, Ramirez Flores RO, Saez-Rodriguez J (2022).
    decoupleR: ensemble of computational methods to infer biological activities from
    omics data. Bioinformatics Advances 2, vbac016. doi:10.1093/bioadv/vbac016

    Dimitrov D, Schaefer PSL, Farr E, Rodriguez-Mier P, Lobentanzer S, Badia-i-Mompel P,
    Dugourd A, Tanevski J, Ramirez Flores RO, Saez-Rodriguez J (2024). LIANA+ provides
    an all-in-one framework for cell-cell communication inference. Nature Cell Biology
    26, 1613-1622. doi:10.1038/s41556-024-01469-w

    ISO (2019). ISO 8601-1:2019, Date and time: representations for information
    interchange, part 1: basic rules. International Organization for Standardization,
    Geneva.
"""

import importlib
import json
import math
import operator
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .. import __version__
from .._columns import values_equal, with_missing
from .._masked_arrays import reject_masked, reject_masked_table, unmasked_array
from .._messages import quoted
from ..cube._axes import axis_codes, checked_n_cells, count_dtype, first_index
from ..cube._provenance_records import source_record
from ..cube.cube import LAYER_NAMES, Cube
from ..table import Table

__all__ = ["to_anndata", "from_anndata", "to_mudata", "from_mudata"]

_RECORD_KEY = "cytepool"
_FORMATS = {"anndata": "cytepool.cube.anndata", "mudata": "cytepool.cube.mudata"}
_FORMAT_VERSION = 1
_RECORD_PARTS = ("layers", "n_cells", "samples", "groups", "features")
_SAMPLE_ID, _GROUP_ID, _CELLS = "sample_id", "group_id", "n_cells"
_SEPARATOR = ":"  # between a modality's name and a feature id, as LIANA writes them
# The layout names that provenance records give.
_CYTEPOOL = "cytepool"
_SCANPY = "scanpy.get.aggregate"
_DECOUPLER = "decoupler.pp.pseudobulk"
_LIANA = "liana.multi.adata_to_views"
_GIVEN = "given"
# Layers that scanpy and decoupler name by what they hold, when it is not a sum.
_NONZERO = "count_nonzero"  # cells with a nonzero value
_FRACTIONS = "psbulk_props"  # the fraction of cells with a nonzero value
_NOT_SUMS = {"mean": "means", "var": "variances", "median": "medians"}
_SCANPY_CELLS = "n_obs_aggregated"
# decoupler 2 names the cells of a row psbulk_cells, although its documentation says
# psbulk_n_cells, the name of decoupler 1 (found by probing).
_DECOUPLER_CELLS = ("psbulk_cells", "psbulk_n_cells")
_DECOUPLER_TOTALS = "psbulk_counts"
_LIANA_STATS = "psbulk_stats"
_TOOL_COLUMNS = frozenset({_SCANPY_CELLS, *_DECOUPLER_CELLS, _DECOUPLER_TOTALS})
# Columns of the tables of a cube that hold text only.
_TEXT_COLUMNS = {"samples": (), "groups": ("cell_ontology_id", "parent")}
_INT64_MAX = 2**63 - 1
_UINT32_MAX = 2**32 - 1
_WHOLE_LIMIT = 2.0**53  # float64 holds every whole number below it
# A fraction d / n stored in a float type, times n, lies within about eps * n of
# the whole number d, for the type's machine epsilon eps; the slack allows 8 eps
# per cell, and a row whose slack reaches a quarter of a cell is ambiguous.
_FRACTION_EPS = 8
# Relative difference allowed between the row totals of X and decoupler's totals,
# which add the same values in another order.
_TOTAL_SLACK = 1e-9


def to_anndata(cube: Cube, *, x_layer: str | None = None) -> Any:
    """Return a cube as a long AnnData, with a row per sample and group with cells.

    The rows of each group come together, groups and samples in the cube's order; a
    cube without cell numbers gives a row for every entry. ``obs`` holds
    ``sample_id`` and ``group_id`` (categorical), the other columns of the sample and
    group tables, and ``n_cells``; ``var`` the feature table, indexed by feature id;
    each layer of the cube is a layer of the same name, and ``X`` is empty unless
    ``x_layer`` names a layer to copy into it, as tools that read ``X`` need.
    ``uns["cytepool"]`` holds the cube's fingerprint and provenance, and what
    ``from_anndata`` needs to give back the cube with the same fingerprint. The
    module documentation describes the layout ("Export") and the record ("The record
    in uns").

    Args:
        cube: The cube. A lazy cube is read one group at a time.
        x_layer: The layer to copy into ``X`` too, such as ``"counts"`` for tools
            that read counts from ``X``; None leaves ``X`` empty. The copy takes as
            much memory as the layer.

    Returns:
        An ``anndata.AnnData``.

    Raises:
        TypeError: If ``cube`` is not a Cube, or ``x_layer`` is not a string.
        KeyError: If the cube has no layer ``x_layer``.
        ImportError: If anndata is not installed.
        ValueError: If a lazy layer gives a value that breaks a constraint (see
            ``Cube``).
        OverflowError: If a sum of a derived lazy layer does not fit.
    """
    _check_cube(cube, "to_anndata()")
    if x_layer is not None:
        if not isinstance(x_layer, str):
            reject_masked(x_layer, "x_layer")
            raise TypeError(
                f"x_layer must be a layer name (str) or None, got "
                f"{type(x_layer).__name__}"
            )
        if x_layer not in cube.layer_names:
            raise KeyError(
                f"the cube has no layer {x_layer!r} to copy into X; its layers are "
                f"{quoted(list(cube.layer_names))}"
            )
    anndata = _optional("anndata", "to_anndata()", "anndata")
    pandas = _optional("pandas", "to_anndata()", "anndata")
    samples, groups = _entries_with_cells(cube)
    sample_ids, group_ids = cube.samples[_SAMPLE_ID], cube.groups[_GROUP_ID]
    columns: dict[str, Any] = {
        _SAMPLE_ID: _categorical(pandas, sample_ids, samples),
        _GROUP_ID: _categorical(pandas, group_ids, groups),
    }
    taken = {_SAMPLE_ID, _GROUP_ID, _CELLS}
    for table, id_column, prefix, positions in (
        (cube.samples, _SAMPLE_ID, "sample_", samples),
        (cube.groups, _GROUP_ID, "group_", groups),
    ):
        for name in table.columns:
            if name != id_column:
                written = _writable_name(name, taken, prefix)
                columns[written] = _writable(table[name])[positions]
    if cube.n_cells is not None:
        columns[_CELLS] = cube.n_cells[samples, groups]
    names = [
        f"{sample}_{group}"
        for sample, group in zip(
            sample_ids[samples].tolist(), group_ids[groups].tolist(), strict=True
        )
    ]
    if len(set(names)) < len(names):
        names = [str(position) for position in range(len(names))]
    obs = pandas.DataFrame(columns, index=pandas.Index(names, dtype=object))
    var, features = _feature_frame(pandas, cube.features, "")
    layers = {
        name: _long_layer(cube, name, samples, groups) for name in cube.layer_names
    }
    record = _record(cube, _FORMATS["anndata"], features)
    x = None if x_layer is None else layers[x_layer].copy()
    return anndata.AnnData(
        X=x, obs=obs, var=var, layers=layers, uns={_RECORD_KEY: record}
    )


def from_anndata(
    adata: Any,
    *,
    sample: str | None = None,
    group: str | None = None,
    layers: Mapping[str, str] | None = None,
    n_cells: Any = None,
    sample_table: Any = None,
    group_table: Any = None,
    normalization: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> Cube:
    """Build a cube from a long AnnData of pooled profiles, a row per sample and group.

    Each row is the profile of the cells of one sample and group, as
    ``scanpy.get.aggregate`` and ``decoupler.pp.pseudobulk`` make them; a combination
    without a row is an entry without cells. With the defaults, the layout of the
    AnnData chooses the source of each layer and of the cell numbers, and an AnnData
    that ``to_anndata`` made gives back its cube with the same fingerprint. The
    module documentation says what each source may hold ("What an import takes") and
    which layouts are recognised.

    Args:
        adata: The AnnData.
        sample: The ``obs`` column with the sample of each row, as text or integers
            (which become their decimal text), as ``from_cells`` takes labels.
            Required unless CytePool made the AnnData.
        group: The ``obs`` column with the group, usually the cell type, of each
            row. Required unless CytePool made the AnnData.
        layers: Cube layer name to its source, ``"X"`` or the name of a layer, for
            example ``{"counts": "sum", "detected": "count_nonzero"}``. ``counts``
            is required. By default the sources of the recognised layout.
        n_cells: The cells pooled into each row: the name of an ``obs`` column, or a
            table with samples as rows and groups as columns, aligned by its labels,
            such as ``pandas.crosstab(cells.obs["donor"], cells.obs["cell_type"])``
            for the cells that were pooled. By default the column of the recognised
            layout, or none.
        sample_table: Sample metadata with a text column ``sample_id`` that holds
            every sample label, as ``from_cells`` takes it: its rows set the order of
            the samples, and a sample without rows has no cells. By default the
            samples are the sorted distinct labels, with the ``obs`` columns that
            are constant within every sample, over the rows with cells.
        group_table: Group metadata, used in the same way. By default the groups are
            the sorted distinct labels, with the other ``obs`` columns that are
            constant within every group.
        normalization: How the normalized values were computed, required with
            ``log_sum``, ``log_sum_sq`` or ``linear_sum`` (see ``Cube``).
        source: A record of where the profiles came from, as a JSON-serialisable
            mapping, kept in the cube's provenance as given (see ``from_cells``).

    Returns:
        A cube. Its provenance holds ``source`` and how the AnnData was read; for
        an AnnData that CytePool made, it is the provenance of the exported cube,
        followed by a record of this import (``cytepool.cube.provenance``
        documents both).

    Raises:
        TypeError: If ``adata`` is not an AnnData, an argument has the wrong type,
            a source does not hold numbers, a column of ``var`` cannot be a column
            of the feature table, or an argument is or holds a numpy masked array.
        KeyError: If a column or layer it names is missing.
        ValueError: If a source holds means, variances or medians, or values that
            are not the sums, counts or fractions it is read as, such as counts
            that are not sums over the cells that ``detected`` counts; a layer
            that CytePool exported was changed; a row repeats an entry; a label is
            missing or not in a given table; a combination has no row and there are
            no cell numbers to mark it as an entry without cells; a value breaks a
            constraint of ``Cube``; or ``uns["cytepool"]`` is not a record this
            version reads.
    """
    _check_arguments(layers, n_cells, sample_table, group_table, normalization)
    caller_source = source_record(source)
    for argument, value in (("sample", sample), ("group", group)):
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"{argument} must name an obs column (str), got {type(value).__name__}"
            )
    _check_anndata(adata)
    record = _read_record(adata.uns, "anndata")
    obs = adata.obs
    obs_columns = list(obs.columns)
    if record is not None:
        layout = _CYTEPOOL
        sample = _SAMPLE_ID if sample is None else sample
        group = _GROUP_ID if group is None else group
    else:
        layout = _long_layout(obs_columns)
        if sample is None or group is None:
            raise ValueError(
                "name the obs columns that hold the sample and the group of each row, "
                "with sample= and group=; obs has: "
                f"{', '.join(map(str, obs_columns)) or '(no columns)'}"
            )
    if sample == group:
        raise ValueError(f"sample and group both name the obs column {sample!r}")
    for argument, column in (("sample", sample), ("group", group)):
        _require_column(obs_columns, column, argument)
    sources = _source_names(adata)
    if n_cells is None:
        n_cells = _default_cells(layout, obs_columns, record)
    if isinstance(n_cells, str):
        _require_column(obs_columns, n_cells, "n_cells")
    mapping = _layer_mapping(layers, layout, sources, record, n_cells is not None)
    names = [str(name) for name in adata.obs_names]
    feature_ids = [str(name) for name in adata.var_names]

    def row_name(row: int) -> str:
        return f"row {row} ({names[row]!r})"

    rows = _Rows.of(
        obs[sample],
        obs[group],
        len(names),
        (f"obs[{sample!r}]", f"obs[{group!r}]"),
        _tables(record, sample_table, group_table),
        row_name,
    )
    if isinstance(n_cells, str):
        rows.cells_per_row(_whole_numbers(obs[n_cells], f"obs[{n_cells!r}]", row_name))
    else:
        rows.cells_from(n_cells)
    inferred: dict[str, list[str]] = {}
    if record is None and (sample_table is None or group_table is None):
        exclude = {sample, group, *_TOOL_COLUMNS}
        if isinstance(n_cells, str):
            exclude.add(n_cells)
        inferred = rows.infer_columns(
            obs,
            [name for name in obs_columns if name not in exclude],
            infer_samples=sample_table is None,
            infer_groups=group_table is None,
        )
    features = _feature_table(
        adata.var,
        feature_ids,
        None if record is None else record["features"],
        "var",
    )
    decoupler = layout == _DECOUPLER

    def read(name: str) -> np.ndarray:
        values = _read_source(adata, name, "the AnnData")
        if name == "X" and decoupler:  # X of decoupler holds sums of counts
            if _DECOUPLER_TOTALS in obs_columns:
                totals = obs[_DECOUPLER_TOTALS]
                where = f"obs[{_DECOUPLER_TOTALS!r}]"
                _check_totals(values, totals, rows.per_row, where, row_name)
            if _FRACTIONS in sources:
                fractions = _read_source(adata, _FRACTIONS, "the AnnData")
                _check_fractions_pattern(values, fractions, row_name, feature_ids)
        return values

    cube_layers = rows.layers(
        mapping,
        read,
        feature_ids,
        recognised=_hint(layout, sources) if layers is None else None,
        recorded=None if record is None else record["layers"],
    )
    if normalization is None and record is not None:
        normalization = record["normalization"]
    if record is not None:
        provenance = _imported(record, "from_anndata", caller_source)
    else:
        provenance = {
            "source": caller_source,
            "construction": {
                "function": "from_anndata",
                "cytepool_version": __version__,
                "layout": layout,
                "rows": len(names),
                "sample_column": sample,
                "group_column": group,
                "layers": mapping,
                "n_cells": _cells_origin(n_cells),
                **rows.orders(sample_table, group_table),
                "obs_columns": inferred or None,
            },
        }
    return rows.cube(cube_layers, features, normalization, provenance)


def to_mudata(cube: Cube) -> Any:
    """Return a cube as a MuData with a modality per group, the samples as rows.

    Each modality is named by its group id and holds the samples with cells in that
    group (every sample, for a cube without cell numbers), named by sample id, with
    ``n_cells`` in its ``obs``, every layer of the cube under its own name, ``X``
    empty, and the feature table in ``var``, whose names are ``<group
    id>:<feature id>`` as LIANA writes them. The columns of the sample table are on
    the MuData's ``obs``, and ``uns["cytepool"]`` holds the cube's fingerprint,
    provenance and what ``from_mudata`` needs to give back the cube with the same
    fingerprint (see "Export" and "The record in uns" in the module
    documentation).

    Args:
        cube: The cube. A lazy cube is read one group at a time.

    Returns:
        A ``mudata.MuData``.

    Raises:
        TypeError: If ``cube`` is not a Cube.
        ImportError: If anndata or mudata is not installed.
        ValueError: If the cube has no groups (``to_anndata`` keeps such a cube),
            or a lazy layer gives a value that breaks a constraint (see ``Cube``).
        OverflowError: If a sum of a derived lazy layer does not fit.
    """
    _check_cube(cube, "to_mudata()")
    anndata = _optional("anndata", "to_mudata()", "mudata")
    mudata = _optional("mudata", "to_mudata()", "mudata")
    pandas = _optional("pandas", "to_mudata()", "mudata")
    if cube.n_groups == 0:
        raise ValueError(
            "to_mudata() makes a modality for each group, and this cube has no "
            "groups; to_anndata() keeps such a cube"
        )
    sample_ids = cube.samples[_SAMPLE_ID]
    _, features = _feature_frame(pandas, cube.features, "")
    modalities = {}
    for g, group_id in enumerate(cube.groups[_GROUP_ID].tolist()):
        if cube.n_cells is None:
            rows = np.arange(cube.n_samples)
            obs = pandas.DataFrame(
                index=pandas.Index(sample_ids.tolist(), dtype=object)
            )
        else:
            rows = np.flatnonzero(cube.n_cells[:, g])
            obs = pandas.DataFrame(
                {_CELLS: cube.n_cells[rows, g]},
                index=pandas.Index(sample_ids[rows].tolist(), dtype=object),
            )
        var, _ = _feature_frame(pandas, cube.features, f"{group_id}{_SEPARATOR}")
        layers = {name: cube.layer(name, g)[rows] for name in cube.layer_names}
        modalities[group_id] = anndata.AnnData(obs=obs, var=var, layers=layers)
    mdata = mudata.MuData(modalities)
    position = {sample: i for i, sample in enumerate(sample_ids.tolist())}
    present = np.array([position[str(name)] for name in mdata.obs_names], dtype=np.intp)
    taken: set[str] = set()
    for name in cube.samples.columns:
        if name != _SAMPLE_ID:
            written = _writable_name(name, taken, "sample_")
            mdata.obs[written] = _writable(cube.samples[name])[present]
    mdata.uns[_RECORD_KEY] = _record(cube, _FORMATS["mudata"], features)
    return mdata


def from_mudata(
    mdata: Any,
    *,
    layers: Mapping[str, str] | None = None,
    n_cells: Any = None,
    sample_table: Any = None,
    group_table: Any = None,
    normalization: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> Cube:
    """Build a cube from a MuData with a modality per group and the samples as rows.

    Each modality is a group, named by the modality's name, and each of its rows the
    profile of a sample, named by its observation name, as LIANA's
    ``liana.multi.adata_to_views`` makes them; a sample without a row in a modality
    has no cells in that group. Every modality must hold the same features: its
    ``var_names``, without the prefix ``<modality name>:`` that LIANA adds. The
    feature table is the ``var`` of the first modality, and the samples' metadata
    the columns of the MuData's ``obs``. With the defaults, the layout chooses the
    sources, and a MuData that ``to_mudata`` made gives back its cube with the same
    fingerprint (see the module documentation).

    Args:
        mdata: The MuData.
        layers: Cube layer name to its source in every modality, ``"X"`` or the
            name of a layer. ``counts`` is required. By default the sources of the
            recognised layout.
        n_cells: The cells pooled into each row: the name of an ``obs`` column of
            every modality, or a table with samples as rows and groups as columns,
            aligned by its labels, such as ``pandas.crosstab`` of the cells' sample
            and group labels. By default the cell numbers of the recognised
            layout, or none.
        sample_table: Sample metadata with a text column ``sample_id`` that holds
            every sample, as in ``from_anndata``. By default the samples are the
            sorted distinct observation names, with the columns of the MuData's
            ``obs``.
        group_table: Group metadata with a text column ``group_id`` that holds every
            modality name. By default the groups are the sorted modality names.
        normalization: How the normalized values were computed, required with
            ``log_sum``, ``log_sum_sq`` or ``linear_sum`` (see ``Cube``).
        source: A record of where the profiles came from, kept in the cube's
            provenance as given (see ``from_cells``).

    Returns:
        A cube, with a provenance as ``from_anndata`` gives it.

    Raises:
        TypeError: If ``mdata`` is not a MuData, an argument has the wrong type, a
            source does not hold numbers, or an argument is or holds a numpy
            masked array.
        KeyError: If a column or layer it names is missing from a modality.
        ValueError: If the modalities hold different features, or for the reasons
            ``from_anndata`` gives.
    """
    _check_arguments(layers, n_cells, sample_table, group_table, normalization)
    caller_source = source_record(source)
    modalities = _modalities(mdata)
    record = _read_record(mdata.uns, "mudata")
    names = [name for name, _ in modalities]
    if record is not None:
        layout = _CYTEPOOL
    elif modalities and all(
        _FRACTIONS in _source_names(modality) for _, modality in modalities
    ):
        layout = _LIANA
    else:
        layout = _GIVEN
    sources = (
        set.intersection(*(set(_source_names(modality)) for _, modality in modalities))
        if modalities
        else set()
    )
    if record is None and group_table is None:
        # Every modality is a group, also one without rows.
        group_table = {_GROUP_ID: sorted(names)}
        groups_given = False
    else:
        groups_given = group_table is not None and record is None
    stats = mdata.uns.get(_LIANA_STATS) if layout == _LIANA else None
    if not hasattr(stats, "columns"):  # LIANA keeps its statistics as a DataFrame
        stats = None
    cells_source: Any = n_cells
    if n_cells is None:
        if record is not None and record["n_cells"]:
            cells_source = _CELLS
        elif stats is not None:
            cells_source = _liana_cells(stats, modalities)
    if isinstance(cells_source, str):
        for name, modality in modalities:
            _require_column(list(modality.obs.columns), cells_source, "n_cells", name)
    mapping = _layer_mapping(
        layers,
        layout,
        sorted(sources, key=lambda source: (source != "X", source)),
        record,
        cells_source is not None,
    )
    sizes = [modality.n_obs for _, modality in modalities]
    labels = [str(label) for _, modality in modalities for label in modality.obs_names]
    sample_labels = np.empty(len(labels), dtype=object)
    sample_labels[:] = labels
    group_labels = np.repeat(np.array(names, dtype=object), sizes)

    def row_name(row: int) -> str:
        return f"sample {sample_labels[row]!r} of modality {group_labels[row]!r}"

    rows = _Rows.of(
        sample_labels,
        group_labels,
        len(labels),
        ("the observation names of the modalities", "the modality names"),
        _tables(record, sample_table, group_table),
        row_name,
    )
    if isinstance(cells_source, str):
        per_row = [
            _whole_numbers(
                modality.obs[cells_source],
                f"mdata.mod[{name!r}].obs[{cells_source!r}]",
                _modality_row_name(name, modality),
            )
            for name, modality in modalities
        ]
        rows.cells_per_row(np.concatenate(per_row) if per_row else np.empty(0, int))
    elif isinstance(cells_source, np.ndarray):  # from LIANA's statistics, per row
        rows.cells_per_row(cells_source)
    else:
        rows.cells_from(cells_source)
    sample_columns: list[str] = []
    if record is None and sample_table is None:
        sample_columns = rows.sample_columns_from(mdata.obs, names)
    ids, positions = _shared_features(modalities)
    features = _feature_table(
        modalities[0][1].var if modalities else _EmptyFrame(),
        ids,
        None if record is None else record["features"],
        f"mdata.mod[{names[0]!r}].var" if names else "var",
    )

    starts = np.concatenate(([0], np.cumsum(sizes))).astype(np.intp)

    def read(source: str) -> np.ndarray:
        blocks = []
        for index, ((name, modality), order) in enumerate(
            zip(modalities, positions, strict=True)
        ):
            values = _read_source(modality, source, f"the modality {name!r}")
            if order is not None:
                values = values[:, order]
            if source == "X" and layout == _LIANA:  # LIANA's X holds sums of counts
                row_name = _modality_row_name(name, modality)
                column = f"{name}{_SEPARATOR}{_DECOUPLER_TOTALS}"
                if stats is not None and column in stats.columns:
                    totals = stats[column].reindex(list(modality.obs_names))
                    cells = None
                    if rows.per_row is not None:
                        cells = rows.per_row[starts[index] : starts[index + 1]]
                    where = f"uns[{_LIANA_STATS!r}][{column!r}]"
                    _check_totals(values, totals, cells, where, row_name)
                fractions = _read_source(modality, _FRACTIONS, f"the modality {name!r}")
                if order is not None:
                    fractions = fractions[:, order]
                _check_fractions_pattern(values, fractions, row_name, ids)
            blocks.append(values)
        if not blocks:
            return np.zeros((0, len(ids)))
        return np.concatenate(blocks)

    cube_layers = rows.layers(
        mapping,
        read,
        ids,
        recognised=_hint(layout, sources) if layers is None else None,
        recorded=None if record is None else record["layers"],
    )
    if normalization is None and record is not None:
        normalization = record["normalization"]
    if record is not None:
        provenance = _imported(record, "from_mudata", caller_source)
    else:
        provenance = {
            "source": caller_source,
            "construction": {
                "function": "from_mudata",
                "cytepool_version": __version__,
                "layout": layout,
                "modalities": names,
                "rows": len(labels),
                "layers": mapping,
                "n_cells": (
                    f"uns[{_LIANA_STATS!r}]"
                    if isinstance(cells_source, np.ndarray)
                    else _cells_origin(cells_source)
                ),
                **rows.orders(sample_table, group_table if groups_given else None),
                "obs_columns": sample_columns if sample_table is None else None,
            },
        }
    return rows.cube(cube_layers, features, normalization, provenance)


# Reading the rows of a long table into the entries of a cube.


class _Rows:
    """The rows of a long AnnData, or of the modalities of a MuData, as entries.

    Holds where each row goes (its sample and group positions), the sample and
    group tables, and the cell numbers of each entry once they are known.
    """

    __slots__ = (
        "sample_codes",
        "group_codes",
        "samples",
        "groups",
        "integer_labels",
        "row_name",
        "n_cells",
        "per_row",
    )

    def __init__(
        self,
        sample_codes: np.ndarray,
        group_codes: np.ndarray,
        samples: Table,
        groups: Table,
        integer_labels: tuple[bool, bool],
        row_name: Callable[[int], str],
    ) -> None:
        self.sample_codes = sample_codes
        self.group_codes = group_codes
        self.samples = samples
        self.groups = groups
        self.integer_labels = integer_labels
        self.row_name = row_name
        self.n_cells: np.ndarray | None = None
        self.per_row: np.ndarray | None = None

    @classmethod
    def of(
        cls,
        sample_labels: Any,
        group_labels: Any,
        n_rows: int,
        label_names: tuple[str, str],
        tables: tuple[tuple[Any, bool], tuple[Any, bool]],
        row_name: Callable[[int], str],
    ) -> "_Rows":
        """Place every row by its labels, refusing a repeated entry.

        ``tables`` holds the sample and the group table (None to take the sorted
        labels), each with whether it comes from CytePool's record.
        """
        positions = []
        for labels, name, (table, recorded), table_name, id_column in (
            (sample_labels, label_names[0], tables[0], "sample_table", _SAMPLE_ID),
            (group_labels, label_names[1], tables[1], "group_table", _GROUP_ID),
        ):
            try:
                positions.append(
                    axis_codes(
                        labels, name, n_rows, table, table_name, id_column, "row"
                    )
                )
            except (TypeError, ValueError) as error:
                if recorded:
                    error.add_note(
                        f"({table_name} is the table that CytePool recorded in "
                        f"uns[{_RECORD_KEY!r}] when it exported the cube; pass "
                        f"{table_name}= to give another)"
                    )
                raise
        (sample_codes, samples, sample_integers) = positions[0]
        (group_codes, groups, group_integers) = positions[1]
        rows = cls(
            sample_codes,
            group_codes,
            samples,
            groups,
            (sample_integers, group_integers),
            row_name,
        )
        rows._check_unique()
        return rows

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.samples), len(self.groups)

    def _check_unique(self) -> None:
        n_samples = len(self.samples)
        entries = self.group_codes * n_samples + self.sample_codes
        order = np.argsort(entries, kind="stable")
        repeated = np.flatnonzero(entries[order][1:] == entries[order][:-1])
        if repeated.size:
            first, second = sorted(
                int(row) for row in order[repeated[0] : repeated[0] + 2]
            )
            sample = self.samples[_SAMPLE_ID][self.sample_codes[first]]
            group = self.groups[_GROUP_ID][self.group_codes[first]]
            raise ValueError(
                f"{self.row_name(first)} and {self.row_name(second)} are both sample "
                f"{sample!r} in group {group!r}; each sample and group needs at most "
                "one row, a profile of all its cells"
            )

    def cells_per_row(self, values: np.ndarray) -> None:
        """Take the cells of every row; entries without a row have none."""
        n_cells = np.zeros(self.shape, dtype=np.int64)
        n_cells[self.sample_codes, self.group_codes] = values
        self.n_cells = n_cells
        self.per_row = values

    def cells_from(self, table: Any) -> None:
        """Take the cells of every entry from a table of samples x groups, or know
        none (``table`` None), checking that the rows cover the entries."""
        n_samples, n_groups = self.shape
        if table is None:
            if self.sample_codes.size < n_samples * n_groups:
                sample, group = self._missing_entry()
                raise ValueError(
                    f"sample {sample!r} has no row in group {group!r}, and there are "
                    "no cell numbers to make it an entry without cells; give n_cells "
                    "(an obs column, or a table of samples x groups such as "
                    "pandas.crosstab of the cells' sample and group labels)"
                )
            return
        ids = (self.groups[_GROUP_ID], self.samples[_SAMPLE_ID], np.empty(0, object))
        # Integer labels became their decimal text.
        table = table.rename(index=str, columns=str)
        for labels, axis, kind in (
            (table.index, "rows", 1),
            (table.columns, "columns", 0),
        ):
            given, expected = [str(label) for label in labels], ids[kind].tolist()
            missing = [label for label in expected if label not in set(given)]
            extra = [label for label in given if label not in set(expected)]
            if missing or extra or len(set(given)) < len(given):
                name = "sample" if kind == 1 else "group"
                raise ValueError(
                    f"n_cells must have the {name} ids as its {axis}, each once; it "
                    f"lacks {missing[:5]} and has {extra[:5]} besides"
                    + ("" if len(set(given)) == len(given) else ", and repeats one")
                )
        n_cells = checked_n_cells(table, ids)
        rowless = n_cells.copy()
        rowless[self.sample_codes, self.group_codes] = 0
        if rowless.any():
            at = first_index(rowless > 0)
            raise ValueError(
                f"n_cells gives {rowless[at]} cells to sample "
                f"{ids[1][at[0]]!r} in group {ids[0][at[1]]!r}, which has no row: "
                "its profile is missing, and its entry cannot be read as one without "
                "cells"
            )
        self.n_cells = n_cells
        self.per_row = n_cells[self.sample_codes, self.group_codes]

    def _missing_entry(self) -> tuple[str, str]:
        n_samples, _ = self.shape
        present = np.zeros(self.shape[::-1], dtype=bool)
        present[self.group_codes, self.sample_codes] = True
        group, sample = first_index(~present)
        return self.samples[_SAMPLE_ID][sample], self.groups[_GROUP_ID][group]

    def infer_columns(
        self,
        frame: Any,
        names: list[str],
        *,
        infer_samples: bool,
        infer_groups: bool,
    ) -> dict[str, list[str]]:
        """Add the columns of ``frame`` that are constant within every sample to the
        sample table, then those constant within every group to the group table.

        Only rows with cells are compared, when cell numbers are known: decoupler
        gives its rows without cells the metadata of another row (found by probing).
        A column that varies, that a table cannot hold, that a table holds as text
        only (``parent``, ``cell_ontology_id``) and is not text, or that is named as
        an id column, is left out. Returns what went where, for the provenance
        record.
        """
        informative = (
            np.ones(self.sample_codes.size, dtype=bool)
            if self.per_row is None
            else self.per_row > 0
        )
        compared = np.flatnonzero(informative if informative.any() else ~informative)
        axes = []
        for infer, codes, size in (
            (infer_samples, self.sample_codes, len(self.samples)),
            (infer_groups, self.group_codes, len(self.groups)),
        ):
            stands_for = _representatives(codes, size, informative) if infer else None
            axes.append((codes, stands_for))
        found: dict[str, dict[str, np.ndarray]] = {"samples": {}, "groups": {}}
        left_out = []
        for name in names:
            if name in (_SAMPLE_ID, _GROUP_ID):  # it would replace the ids
                left_out.append(name)
                continue
            try:
                column = Table({name: frame[name]})[name]
            except (TypeError, ValueError):
                left_out.append(name)
                continue
            for axis, (codes, stands_for) in zip(found, axes, strict=True):
                if name in _TEXT_COLUMNS[axis] and column.dtype != object:
                    continue  # the table holds this column as text only
                if stands_for is not None and bool(
                    values_equal(
                        column[compared], column[stands_for[codes[compared]]]
                    ).all()
                ):
                    found[axis][name] = column[stands_for]
                    break
            else:
                left_out.append(name)
        if found["samples"]:
            self.samples = self.samples.with_columns(found["samples"])
        if found["groups"]:
            self.groups = self.groups.with_columns(found["groups"])
        return {
            "samples": list(found["samples"]),
            "groups": list(found["groups"]),
            "left_out": left_out,
        }

    def sample_columns_from(self, frame: Any, modalities: list[str]) -> list[str]:
        """Add the columns of a MuData's ``obs``, one row per sample, to the sample
        table, leaving out those that a modality's name prefixes."""
        index = {str(label): row for row, label in enumerate(frame.index)}
        ids = self.samples[_SAMPLE_ID].tolist()
        rows = np.array([index.get(sample, -1) for sample in ids], dtype=np.intp)
        lacking = rows < 0  # samples that the MuData's obs lacks get missing values
        prefixes = tuple(f"{name}{_SEPARATOR}" for name in modalities)
        added: dict[str, np.ndarray] = {}
        for name in frame.columns:
            if not isinstance(name, str) or name.startswith(prefixes):
                continue
            if name == _SAMPLE_ID:
                continue
            try:
                column = Table({name: frame[name]})[name]
            except (TypeError, ValueError):
                continue
            added[name] = with_missing(column[np.where(lacking, 0, rows)], lacking)
        if added:
            self.samples = self.samples.with_columns(added)
        return list(added)

    def layers(
        self,
        mapping: Mapping[str, str],
        read: Callable[[str], np.ndarray],
        feature_ids: list[str],
        *,
        recognised: str | None,
        recorded: Mapping[str, np.dtype] | None,
    ) -> dict[str, np.ndarray]:
        """Read every source once and place its rows in the cube's layers.

        ``recognised`` is the hint that the layout which chose the sources gives
        for counts that are not whole numbers (None when ``layers`` named the
        sources); ``recorded`` the type of each layer that CytePool exported, which
        a layer read from its own name keeps, or None for data it did not export.
        """
        n_samples, n_groups = self.shape
        read_once: dict[str, np.ndarray] = {}
        result = {}
        # Whether the counts are sums of whole counts: integers, or a recognised
        # tool's sums of counts. Sums of other values can be whole by chance.
        whole_counts = recognised is not None
        for name in LAYER_NAMES:
            source = mapping.get(name)
            if source is None:
                continue
            if source not in read_once:
                read_once[source] = read(source)
            values = read_once[source]
            what = "X" if source == "X" else f"the layer {source!r}"
            if values.shape != (self.sample_codes.size, len(feature_ids)):
                raise ValueError(
                    f"{what} has shape {values.shape}, expected "
                    f"{(self.sample_codes.size, len(feature_ids))}"
                )

            def where(index: tuple[int, ...]) -> str:
                row, feature = index
                return f"{self.row_name(row)}, feature {feature_ids[feature]!r}"

            if name == "counts" and values.dtype.kind in "iu":
                whole_counts = True
            if recorded is not None and source == name and name in recorded:
                converted = _as_recorded(values, recorded[name], what, where)
            else:
                converted = self._converted(
                    name, source, values, what, where, recognised
                )
            layer = np.zeros((n_groups, n_samples, len(feature_ids)), converted.dtype)
            layer[self.group_codes, self.sample_codes] = converted
            result[name] = layer
        if recorded is None and "detected" in result:
            ids = (self.groups[_GROUP_ID], self.samples[_SAMPLE_ID], feature_ids)
            _check_detected(result["counts"], result["detected"], ids, whole_counts)
        return result

    def _converted(
        self,
        name: str,
        source: str,
        values: np.ndarray,
        what: str,
        where: Callable[[tuple[int, ...]], str],
        recognised: str | None,
    ) -> np.ndarray:
        """Return a source's values as the values of a cube layer, checked."""
        if name not in ("counts", "detected"):
            return values.astype(np.float64, copy=False)
        if source == _FRACTIONS:
            assert self.per_row is not None  # _layer_mapping requires cell numbers
            return _detected_from_fractions(
                values, self.per_row, what, where, self.row_name
            )
        _check_counts(values, what, where)
        largest = values.max() if values.size else 0
        dtype = count_dtype(values, largest)
        if dtype is not np.float64:
            return values.astype(dtype)
        fractional = values != np.floor(values)
        if not fractional.any():  # whole numbers beyond 2**53
            index = first_index(values >= _WHOLE_LIMIT)
            if name == "detected" or recognised is not None:
                raise ValueError(
                    f"{what} holds {values[index]} at {where(index)}, at or above "
                    "2**53, beyond which float64 does not hold every whole number, so "
                    f"the {'cells' if name == 'detected' else 'counts'} it stands for "
                    "are not known exactly"
                    + (
                        ""
                        if name == "detected"
                        else "; give layers= to read them as float64 counts"
                    )
                )
            return values.astype(np.float64, copy=False)
        index = first_index(fractional)
        found = f"{what} holds {values[index]} at {where(index)}, not a whole number"
        if name == "detected":
            raise ValueError(f"{found}, but 'detected' counts cells")
        if recognised is not None:
            raise ValueError(
                f"{found}, so it does not hold sums of counts{recognised}. Say what "
                "it holds with layers=: {'counts': ...} for counts that are not "
                "whole numbers; sums of normalized values are read with the counts of "
                "the same cells, as in {'counts': ..., 'log_sum': ...}, with "
                "normalization="
            )
        return values.astype(np.float64, copy=False)

    def orders(self, sample_table: Any, group_table: Any) -> dict[str, Any]:
        """The provenance entries that say how samples and groups were ordered."""
        return {
            axis: {
                "order": f"{axis[:-1]}_table" if table is not None else "sorted_labels",
                "labels": "integers" if integers else "text",
            }
            for axis, table, integers in (
                ("samples", sample_table, self.integer_labels[0]),
                ("groups", group_table, self.integer_labels[1]),
            )
        }

    def cube(
        self,
        layers: dict[str, np.ndarray],
        features: Table,
        normalization: Mapping[str, Any] | None,
        provenance: dict[str, Any],
    ) -> Cube:
        return Cube(
            layers,
            samples=self.samples,
            groups=self.groups,
            features=features,
            n_cells=self.n_cells,
            normalization=normalization,
            provenance=provenance,
            copy=False,
        )


def _as_recorded(
    values: np.ndarray,
    dtype: np.dtype,
    what: str,
    where: Callable[[tuple[int, ...]], str],
) -> np.ndarray:
    """Return an exported layer in the type it was exported in, when that type holds
    its values exactly, as it does unless the values were changed since."""
    if values.dtype == dtype:
        return values
    with np.errstate(invalid="ignore", over="ignore"):
        cast = values.astype(dtype)
    same = cast == values
    if not same.all():
        index = first_index(~same)
        raise ValueError(
            f"{what} holds {values[index]} at {where(index)}, which its type when "
            f"CytePool exported it, {dtype}, does not hold: it was changed after the "
            "export. Give layers= to read it as what it holds now"
        )
    return cast


def _check_detected(
    counts: np.ndarray,
    detected: np.ndarray,
    ids: tuple[Any, Any, Any],
    whole_counts: bool,
) -> None:
    """Check that the counts sum over the cells that ``detected`` counts.

    A sum of non-negative counts is positive exactly where a cell has a positive
    count, and a sum of whole counts (``whole_counts``) is at least the number of
    such cells. Means and medians break these, and so do counts of other cells.
    """
    wrong = (counts > 0) != (detected > 0)
    if whole_counts and counts.dtype.kind in "iu":
        wrong |= counts < detected
    if wrong.any():
        group, sample, feature = first_index(wrong)
        raise ValueError(
            f"the counts are not sums over the cells that 'detected' counts: at group "
            f"{ids[0][group]!r}, sample {ids[1][sample]!r}, feature "
            f"{ids[2][feature]!r} the counts are {counts[group, sample, feature]} and "
            f"{detected[group, sample, feature]} cells detect the feature. A sum of "
            "counts is positive exactly where a cell has a positive count, and a sum of "
            "whole counts is at least the number of such cells: these counts hold "
            "another statistic, such as means or medians, or count other cells"
        )


def _check_fractions_pattern(
    values: np.ndarray,
    fractions: np.ndarray,
    row_name: Callable[[int], str],
    feature_ids: list[str],
) -> None:
    """Check X against decoupler's fractions of the cells that detect each feature.

    A sum of the cells' counts is positive exactly where a cell detects the feature.
    A sum of whole counts is also at least the number of detecting cells, d = f n
    for the fraction f of the n cells. n is not needed for a bound: the rarest
    feature of a row, of fraction f_min, is detected by at least one cell, so
    n >= 1 / f_min and d >= f / f_min, which is d itself where a feature is detected
    by one cell only, as some feature of a sparse row usually is.
    """
    detecting = fractions > 0
    wrong = (values > 0) != detecting
    if wrong.any():
        row, feature = first_index(wrong)
        raise ValueError(
            f"X does not hold sums of the counts of its cells: at {row_name(row)}, "
            f"feature {feature_ids[feature]!r}, X holds {values[row, feature]}, but "
            f"{_FRACTIONS} says that {fractions[row, feature]:g} of the cells detect "
            "the feature. A sum is positive exactly where a cell detects the feature: "
            "X holds another statistic, such as medians (mode='median'); pool the "
            "cells with mode='sum'"
        )
    if values.dtype.kind == "f" and not bool(np.all(values == np.floor(values))):
        return  # sums of counts that are not whole numbers have no such bound
    rarest = np.where(detecting, fractions, np.inf).min(axis=1, initial=np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        least = np.where(detecting, fractions / rarest[:, None], 0.0)
    short = values < least * (1 - _TOTAL_SLACK)
    if short.any():
        row, feature = first_index(short)
        raise ValueError(
            f"X does not hold sums of the counts of its cells: at {row_name(row)}, "
            f"feature {feature_ids[feature]!r}, X holds {values[row, feature]}, but at "
            f"least {least[row, feature]:.6g} of its cells detect the feature "
            f"({_FRACTIONS} gives {fractions[row, feature]:g}, and "
            f"{rarest[row]:g}, at least one cell, for the rarest feature of the row). "
            "A sum of whole counts is at least the number of cells with a positive "
            "count: X holds another statistic, such as maxima or means; pool the "
            "cells with mode='sum'"
        )


def _representatives(
    codes: np.ndarray, size: int, informative: np.ndarray
) -> np.ndarray:
    """Return, for each position, the row that stands for it: its first row with
    cells, or its first row when it has none. Every position has a row."""
    order = np.lexsort((np.arange(codes.size), ~informative, codes))
    first: np.ndarray = order[np.searchsorted(codes[order], np.arange(size))]
    return first


def _detected_from_fractions(
    values: np.ndarray,
    cells: np.ndarray,
    what: str,
    where: Callable[[tuple[int, ...]], str],
    row_name: Callable[[int], str],
) -> np.ndarray:
    """Return the cells detecting each feature from the fraction of the cells.

    A fraction d / n stored in a float type holds d to within about eps * n, for the
    type's machine epsilon eps, so its product with n rounds to d exactly; one that
    lies further from a whole number is not such a fraction.
    """
    eps = float(np.finfo(values.dtype).eps) if values.dtype.kind == "f" else 0.0
    fractions = values.astype(np.float64, copy=False)
    bad = ~(np.isfinite(fractions) & (fractions >= 0) & (fractions <= 1))
    if bad.any():
        index = first_index(bad)
        raise ValueError(
            f"{what} holds {fractions[index]} at {where(index)}; fractions of cells are "
            "from 0 to 1"
        )
    slack = _FRACTION_EPS * eps * np.maximum(cells, 1).astype(np.float64)
    coarse = np.flatnonzero(slack >= 0.25)
    if coarse.size:
        row = int(coarse[0])
        raise ValueError(
            f"{what} holds fractions in {values.dtype}, too few digits to give exactly "
            f"how many of the {cells[row]} cells of {row_name(row)} detect each "
            "feature; store them in float64"
        )
    products = fractions * cells[:, None]
    detected = np.rint(products)
    off = ~(np.abs(products - detected) <= slack[:, None]) | (
        (cells[:, None] == 0) & (fractions > 0)
    )
    if off.any():
        index = first_index(off)
        raise ValueError(
            f"{what} holds the fraction {float(fractions[index])!r} of the "
            f"{cells[index[0]]} cells of {where(index)}, which is "
            f"{float(products[index])!r} cells, not a whole number: it is not the "
            "fraction of these cells that detect the feature"
        )
    largest = int(detected.max()) if detected.size else 0
    counted: np.ndarray = detected.astype(
        np.uint32 if largest <= _UINT32_MAX else np.uint64
    )
    return counted


def _check_counts(
    values: np.ndarray, what: str, where: Callable[[tuple[int, ...]], str]
) -> None:
    """Check that values can be counts: finite, non-negative and at most 2**63 - 1."""
    if values.size == 0:
        return
    kind = values.dtype.kind
    if kind == "f" and not np.isfinite(values).all():
        index = first_index(~np.isfinite(values))
        raise ValueError(
            f"{what} has a non-finite value ({values[index]}) at {where(index)}"
        )
    if kind == "u" and values.dtype.itemsize == 8 and values.max() > _INT64_MAX:
        index = first_index(values > _INT64_MAX)
        raise ValueError(
            f"{what} has a value above 2**63 - 1 ({values[index]}) at {where(index)}; "
            "counts are at most 2**63 - 1, the largest int64"
        )
    if kind != "u" and values.min() < 0:
        index = first_index(values < 0)
        raise ValueError(
            f"{what} has a negative value ({values[index]}) at {where(index)}; counts "
            "of molecules and of cells are non-negative"
        )


def _check_totals(
    values: np.ndarray,
    totals: Any,
    cells: np.ndarray | None,
    name: str,
    row_name: Callable[[int], str],
) -> None:
    """Check X against decoupler's totals of the pooled counts, psbulk_counts.

    decoupler computes psbulk_counts from the summed counts of every feature,
    whatever the mode of its pseudobulk (found by probing). A row of sums, perhaps
    of fewer features after a filter such as decoupler.pp.filter_by_expr, adds up to
    at most its total; a row of means of n cells adds up to the total over n.
    """
    expected = _float_column(totals, name)
    found = values.sum(axis=1, dtype=np.float64)
    slack = _TOTAL_SLACK * np.maximum(np.abs(found), np.abs(expected))
    over = ~(found - expected <= slack)
    if over.any():
        row = int(np.argmax(over))
        raise ValueError(
            f"X adds up to {found[row]:g} at {row_name(row)}, more than the "
            f"{expected[row]:g} counts that {name} gives for its cells, so it does not "
            "hold sums of their counts: X may hold normalized values, with the counts "
            "in another layer (give it with layers={'counts': ...}), or come from "
            "other cells"
        )
    if cells is None:
        return
    several = (cells > 1) & (expected > 0)
    means = np.abs(found * cells - expected) <= _TOTAL_SLACK * expected
    if several.any() and bool(means[several].all()):
        row = int(np.flatnonzero(several)[0])
        raise ValueError(
            f"X holds means: at {row_name(row)} it adds up to {found[row]:g}, the "
            f"{expected[row]:g} counts of {name} divided by its {cells[row]} cells, "
            "as it does in every row of two or more cells (mode='mean'). A cube holds "
            "sums, which add up exactly when groups are merged or samples pooled: pool "
            "the cells with mode='sum'"
        )


def _float_column(column: Any, name: str) -> np.ndarray:
    values = column.to_numpy() if hasattr(column, "to_numpy") else column
    array = unmasked_array(values, name)
    try:
        return array.astype(np.float64)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must hold numbers, got dtype {array.dtype}") from None


def _whole_numbers(
    column: Any, name: str, row_name: Callable[[int], str]
) -> np.ndarray:
    """Return a column of cell numbers as int64, checking that each is one."""
    values = column.to_numpy() if hasattr(column, "to_numpy") else column
    array = unmasked_array(values, name)
    if array.dtype == object:
        try:
            array = array.astype(np.float64)
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} holds a missing or non-numeric value; every row needs a number "
                "of cells"
            ) from None
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold numbers of cells, got dtype {array.dtype}")
    if array.dtype.kind == "f":
        whole = np.isfinite(array) & (array == np.floor(array)) & (array < 2.0**63)
        if not whole.all():
            row = int(np.argmax(~whole))
            raise ValueError(
                f"{name} is {array[row]} at {row_name(row)}; numbers of cells are "
                "whole numbers"
            )
    elif array.dtype.kind == "u" and array.size and array.max() > _INT64_MAX:
        row = int(np.argmax(array > _INT64_MAX))
        raise ValueError(
            f"{name} is {array[row]} at {row_name(row)}, above 2**63 - 1, the most "
            "cells a cube records"
        )
    if array.size and array.min() < 0:
        row = int(np.argmax(array < 0))
        raise ValueError(f"{name} is negative ({array[row]}) at {row_name(row)}")
    return array.astype(np.int64)


# Choosing the sources.


def _check_arguments(
    layers: Any, n_cells: Any, sample_table: Any, group_table: Any, normalization: Any
) -> None:
    """Refuse masked arrays and wrong types in the arguments, before any reading."""
    reject_masked(layers, "layers")
    reject_masked(n_cells, "n_cells")
    reject_masked_table(sample_table, "sample_table", _SAMPLE_ID)
    reject_masked_table(group_table, "group_table", _GROUP_ID)
    reject_masked(normalization, "normalization", missing="None")
    if layers is not None and not isinstance(layers, Mapping):
        raise TypeError(
            "layers must map cube layer names to sources, such as {'counts': 'sum'}, "
            f"got {type(layers).__name__}"
        )
    if (
        n_cells is not None
        and not isinstance(n_cells, str)
        and not all(
            hasattr(n_cells, attribute)
            for attribute in ("index", "columns", "to_numpy")
        )
    ):
        raise TypeError(
            "n_cells must name an obs column, or be a table of cell numbers with "
            "samples as rows and groups as columns (a pandas DataFrame); got "
            f"{type(n_cells).__name__}"
        )


def _long_layout(obs_columns: list[Any]) -> str:
    if any(column in obs_columns for column in _DECOUPLER_CELLS):
        return _DECOUPLER
    if _SCANPY_CELLS in obs_columns:
        return _SCANPY
    return _GIVEN


def _default_cells(
    layout: str, obs_columns: list[Any], record: dict[str, Any] | None
) -> str | None:
    if record is not None:
        return _CELLS if record["n_cells"] else None
    if layout == _DECOUPLER:
        return next(column for column in _DECOUPLER_CELLS if column in obs_columns)
    if layout == _SCANPY:
        return _SCANPY_CELLS
    return None


def _layer_mapping(
    layers: Mapping[str, str] | None,
    layout: str,
    sources: list[str] | set[str],
    record: dict[str, Any] | None,
    has_cells: bool,
) -> dict[str, str]:
    """Return the source of each cube layer, given or chosen by the layout."""
    available = set(sources)
    if layers is None:
        if record is not None:
            mapping = {name: name for name in record["layers"]}
        elif layout in (_DECOUPLER, _LIANA):
            mapping = {"counts": "X"}
            if _FRACTIONS in available and has_cells:
                mapping["detected"] = _FRACTIONS
        elif layout == _SCANPY:
            if "sum" not in available:
                held = [_NOT_SUMS[name] for name in _NOT_SUMS if name in available]
                described = f", which hold {' and '.join(held)}" if held else ""
                raise ValueError(
                    "this AnnData has the layout of scanpy.get.aggregate but no layer "
                    f"'sum'; its layers are {sorted(available)}{described}. A cube "
                    "holds sums, which add up exactly when groups are merged or "
                    "samples pooled: aggregate the counts with func=['sum', "
                    "'count_nonzero']"
                )
            mapping = {"counts": "sum"}
            if _NONZERO in available:
                mapping["detected"] = _NONZERO
        else:
            raise ValueError(
                "say which source holds each cube layer, for example layers="
                "{'counts': 'X'}: the layout of this data is not one that CytePool "
                "recognises (CytePool's export, scanpy.get.aggregate, "
                "decoupler.pp.pseudobulk or liana.multi.adata_to_views); its sources "
                f"are {sorted(available, key=str)}"
            )
    else:
        mapping = dict(layers)
    for name, source in mapping.items():
        if not isinstance(name, str) or name not in LAYER_NAMES:
            raise ValueError(
                f"layers names the unknown cube layer {name!r}; cube layers are "
                f"named: {', '.join(LAYER_NAMES)}"
            )
        if not isinstance(source, str):
            raise TypeError(
                f"layers[{name!r}] must name X or a layer (str), got "
                f"{type(source).__name__}"
            )
        if source in _NOT_SUMS and source not in LAYER_NAMES:
            raise ValueError(
                f"layers[{name!r}] is {source!r}, which scanpy.get.aggregate and "
                f"decoupler.pp.pseudobulk fill with {_NOT_SUMS[source]}. A cube holds "
                "sums, which add up exactly when groups are merged or samples are "
                "pooled: a mean times its number of cells gives the sum only up to "
                "rounding, and without cell numbers not at all, and a variance or a "
                "median gives no sum. Aggregate with func='sum' (scanpy) or "
                "mode='sum' (decoupler) instead"
            )
        if source in (_NONZERO, _FRACTIONS) and name != "detected":
            raise ValueError(
                f"layers[{name!r}] is {source!r}, which counts cells, or gives their "
                "fraction, and so can give only the layer 'detected'"
            )
        if source == _FRACTIONS and not has_cells:
            raise ValueError(
                f"the fractions in {_FRACTIONS!r} give 'detected' only with the number "
                "of cells of each row: give n_cells, or leave 'detected' out of layers"
            )
    if "counts" not in mapping:
        raise ValueError(
            f"layers must give the source of 'counts', which every cube holds; got "
            f"{mapping}"
        )
    return {name: mapping[name] for name in LAYER_NAMES if name in mapping}


def _hint(layout: str, sources: Any) -> str | None:
    """What a recognised layout says of counts that are not whole numbers, or None
    for sources that ``layers`` named."""
    if layout not in (_SCANPY, _DECOUPLER, _LIANA):
        return None
    if layout == _SCANPY:
        return " (scanpy.get.aggregate sums whatever matrix it reads)"
    counts = ", such as 'counts'" if "counts" in sources else ""
    return (
        " (decoupler sums whole counts to whole numbers, so X holds means, from "
        "mode='mean', or values that are not counts, such as normalized ones; the "
        f"counts may be in another layer{counts})"
    )


def _source_names(adata: Any) -> list[str]:
    """The sources of an AnnData: X, unless it is empty, and its layers.

    anndata 0.13 lists X among the layers too, under the key None (found by
    probing), which is left out here.
    """
    names = [name for name in adata.layers.keys() if isinstance(name, str)]
    return (["X"] if adata.X is not None else []) + names


def _read_source(adata: Any, source: str, what: str) -> np.ndarray:
    if source == "X":
        value = adata.X
        if value is None:
            raise KeyError(f"X of {what} is empty; give another source in layers")
        name = f"X of {what}"
    else:
        names = [name for name in adata.layers.keys() if isinstance(name, str)]
        if source not in names:
            raise KeyError(
                f"{what} has no layer {source!r}; its layers are: "
                f"{', '.join(names) or '(none)'}"
                + ("" if adata.X is None else ", and X")
            )
        value = adata.layers[source]
        name = f"the layer {source!r} of {what}"
    if hasattr(value, "to_memory"):  # anndata's sparse datasets of a backed file
        value = value.to_memory()
    if hasattr(value, "toarray") and not isinstance(value, np.ndarray):
        value = value.toarray()  # scipy's sparse matrices and arrays
    elif hasattr(value, "compute") and not isinstance(value, np.ndarray):
        value = value.compute()  # dask arrays
    array = unmasked_array(value, name)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {array.shape}")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold numbers, got dtype {array.dtype}")
    return array


def _liana_cells(stats: Any, modalities: list[tuple[str, Any]]) -> np.ndarray | None:
    """Return the cells of every row from LIANA's statistics, when it kept them for
    every modality (its keep_stats), else None."""
    per_row = []
    for name, modality in modalities:
        found = [
            f"{name}{_SEPARATOR}{column}"
            for column in _DECOUPLER_CELLS
            if f"{name}{_SEPARATOR}{column}" in stats.columns
        ]
        if not found:
            return None
        per_row.append(
            _whole_numbers(
                stats[found[0]].reindex([str(label) for label in modality.obs_names]),
                f"uns[{_LIANA_STATS!r}][{found[0]!r}]",
                _modality_row_name(name, modality),
            )
        )
    return np.concatenate(per_row) if per_row else None


def _modality_row_name(name: str, modality: Any) -> Callable[[int], str]:
    """How messages name a row of a modality: its sample and the modality."""

    def row_name(row: int) -> str:
        return f"sample {str(modality.obs_names[row])!r} of modality {name!r}"

    return row_name


def _cells_origin(n_cells: Any) -> str | None:
    if n_cells is None or isinstance(n_cells, str):
        return n_cells
    return "table"


def _require_column(
    columns: list[Any], name: str, argument: str, modality: str | None = None
) -> None:
    if name not in columns:
        where = "obs" if modality is None else f"the obs of the modality {modality!r}"
        raise KeyError(
            f"{argument} names {name!r}, which is not a column of {where}; its columns "
            f"are: {', '.join(map(str, columns)) or '(none)'}"
        )


def _tables(
    record: dict[str, Any] | None, sample_table: Any, group_table: Any
) -> tuple[tuple[Any, bool], tuple[Any, bool]]:
    """The sample and group tables to place rows by: given, else recorded, and
    whether each is the recorded one."""
    if record is None:
        return (sample_table, False), (group_table, False)
    return (
        (record["samples"], True) if sample_table is None else (sample_table, False),
        (record["groups"], True) if group_table is None else (group_table, False),
    )


def _imported(
    record: dict[str, Any], function: str, source: dict[str, Any] | None
) -> dict[str, Any]:
    """The provenance of an exported cube, with a record of its import."""
    provenance: dict[str, Any] = record["provenance"]
    operations = list(provenance.get("operations", []))
    operations.append(
        {"operation": function, "input": record["fingerprint"], "source": source}
    )
    return {**provenance, "operations": operations}


# Containers.


def _check_cube(cube: Any, caller: str) -> None:
    if not isinstance(cube, Cube):
        raise TypeError(f"{caller} converts a Cube, got {type(cube).__name__}")


def _optional(module: str, caller: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            f"{caller} needs {module}, which is not installed: install CytePool with "
            f"its {extra!r} extra (pip install 'cytepool[{extra}]'), or install "
            f"{module} itself"
        ) from error


def _check_anndata(adata: Any) -> None:
    if hasattr(adata, "mod"):
        raise TypeError(
            "from_anndata() reads an AnnData; read a MuData with from_mudata()"
        )
    needed = ("obs", "var", "X", "layers", "uns", "obs_names", "var_names")
    if not all(hasattr(adata, attribute) for attribute in needed):
        raise TypeError(
            f"from_anndata() reads an anndata.AnnData, got {type(adata).__name__}"
        )


def _modalities(mdata: Any) -> list[tuple[str, Any]]:
    modalities: Any = getattr(mdata, "mod", None)
    if not hasattr(modalities, "items") or not all(
        hasattr(mdata, attribute) for attribute in ("obs", "uns")
    ):
        hint = "; read an AnnData with from_anndata()" if hasattr(mdata, "X") else ""
        raise TypeError(
            f"from_mudata() reads a mudata.MuData, got {type(mdata).__name__}{hint}"
        )
    found = []
    for name, modality in modalities.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"modality names are group ids, non-empty text; got {name!r}"
            )
        found.append((name, modality))
    return found


def _shared_features(
    modalities: list[tuple[str, Any]],
) -> tuple[list[str], list[np.ndarray | None]]:
    """Return the feature ids every modality holds, in the first modality's order,
    and where each modality holds them (None where in that order)."""
    if not modalities:
        return [], []
    held = [_feature_ids(name, modality.var_names) for name, modality in modalities]
    ids = held[0]
    position = {feature: i for i, feature in enumerate(ids)}
    positions: list[np.ndarray | None] = []
    for (name, _), own in zip(modalities, held, strict=True):
        if own == ids:
            positions.append(None)
            continue
        own_position = {feature: i for i, feature in enumerate(own)}
        missing = [feature for feature in ids if feature not in own_position]
        extra = [feature for feature in own if feature not in position]
        if missing or extra or len(own) != len(ids):
            raise ValueError(
                f"the modalities hold different features: {name!r} lacks "
                f"{len(missing)} of the features of {modalities[0][0]!r} "
                f"({', '.join(map(repr, missing[:5]))}) and has {len(extra)} others "
                f"({', '.join(map(repr, extra[:5]))}). A cube's features are those of "
                "every group: keep only the features that every modality holds first"
            )
        positions.append(np.array([own_position[feature] for feature in ids]))
    return ids, positions


def _feature_ids(modality: str, var_names: Any) -> list[str]:
    """The feature ids of a modality: its var_names, without LIANA's prefix."""
    names = [str(name) for name in var_names]
    prefix = f"{modality}{_SEPARATOR}"
    if names and all(name.startswith(prefix) for name in names):
        return [name[len(prefix) :] for name in names]
    return names


class _EmptyFrame:
    """The var of a MuData without modalities: no columns."""

    columns: list[str] = []


# Writing tables in the forms anndata stores, and reading them back.


def _entries_with_cells(cube: Cube) -> tuple[np.ndarray, np.ndarray]:
    """The sample and group positions of the rows of a long AnnData, group by group."""
    groups, samples = np.divmod(
        np.arange(cube.n_groups * cube.n_samples), cube.n_samples
    )
    if cube.n_cells is not None:
        kept = cube.n_cells.T.reshape(-1) > 0
        groups, samples = groups[kept], samples[kept]
    return samples.astype(np.intp), groups.astype(np.intp)


def _long_layer(
    cube: Cube, name: str, samples: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Return a layer's rows for the entries of a long AnnData, a group at a time."""
    result = np.empty((samples.size, cube.n_features), dtype=cube._layers[name].dtype)
    starts = np.searchsorted(groups, np.arange(cube.n_groups + 1))
    for group in range(cube.n_groups):
        start, stop = int(starts[group]), int(starts[group + 1])
        if start < stop:
            result[start:stop] = cube.layer(name, group)[samples[start:stop]]
    return result


def _categorical(pandas: Any, ids: np.ndarray, positions: np.ndarray) -> Any:
    used = np.unique(positions)
    return pandas.Categorical(ids[positions].tolist(), categories=ids[used].tolist())


def _writable_name(name: str, taken: set[str], prefix: str) -> str:
    """Return a column name that anndata writes and that is not taken yet.

    HDF5 and Zarr keys cannot hold "/" or NUL, nor be "." or "..", and anndata
    reserves "_index" (found by probing).
    """
    written = name.replace("/", "_").replace("\x00", "_")
    if written in (".", "..", "_index"):
        written += "_"
    if written in taken:
        written = prefix + written
    while written in taken:
        written += "_"
    taken.add(written)
    return written


def _writable(column: np.ndarray) -> np.ndarray:
    """Return a table column in a form anndata writes and reads back as it was.

    Datetimes become ISO 8601 text, with numpy's ``NaT`` for missing values, and
    timedeltas int64 counts of their unit, since anndata writes neither. A text
    column with no value present becomes float64 NaN: anndata cannot write such a
    column to HDF5, and writes its values to Zarr as the text "None" (found by
    probing).
    """
    if column.dtype.kind == "M":
        text: np.ndarray = np.datetime_as_string(column).astype(object)
        return text
    if column.dtype.kind == "m":
        counts: np.ndarray = column.view(np.int64)
        return counts
    if (
        column.dtype.kind == "O"
        and column.size
        and all(value is None for value in column.tolist())
    ):
        return np.full(column.size, np.nan)
    return column


def _feature_frame(
    pandas: Any, table: Table, prefix: str
) -> tuple[Any, list[dict[str, Any]]]:
    """Return the var DataFrame of a feature table, and the record of its columns."""
    columns: dict[str, np.ndarray] = {}
    record: list[dict[str, Any]] = []
    taken: set[str] = set()
    for name in table.columns:
        if name == "feature_id":
            record.append({"name": name, "column": None, "type": "text"})
            continue
        written = _writable_name(name, taken, "feature_")
        columns[written] = _writable(table[name])
        record.append(
            {"name": name, "column": written, "type": _type_name(table[name])}
        )
    ids = [prefix + feature for feature in table["feature_id"].tolist()]
    frame = pandas.DataFrame(columns, index=pandas.Index(ids, dtype=object))
    return frame, record


def _feature_table(
    var: Any, ids: list[str], recorded: list[dict[str, Any]] | None, where: str
) -> Table:
    """Build the feature table from var, restoring the column types recorded."""
    id_column = np.empty(len(ids), dtype=object)
    id_column[:] = ids
    columns: dict[str, np.ndarray] = {}
    used: set[Any] = set()
    if recorded is None:
        columns["feature_id"] = id_column
    else:
        for entry in recorded:
            column = entry["column"]
            if column is None:
                columns[entry["name"]] = id_column
            elif column in var.columns:
                used.add(column)
                values = _frame_column(var, column, where)
                columns[entry["name"]] = _restored(values, entry["type"])
        if "feature_id" not in columns:
            columns = {"feature_id": id_column, **columns}
    for name in var.columns:
        if name in used:
            continue
        if name in columns:
            raise ValueError(
                f"{where} has a column {name!r}, which is also the name of a column of "
                "the feature table (the feature ids are the var_names); rename it"
            )
        columns[name] = _frame_column(var, name, where)
    for name, kind, held in (("symbol", "O", "text"), ("measured", "b", "booleans")):
        if name in columns and columns[name].dtype.kind != kind:
            raise TypeError(
                f"the column {name!r} of {where} has dtype {columns[name].dtype}, but a "
                f"cube's feature table holds {name!r} as {held}; rename or drop the "
                "column to read the data"
            )
    return Table(columns)


def _frame_column(frame: Any, name: Any, where: str) -> np.ndarray:
    try:
        column: np.ndarray = Table({name: frame[name]})[name]
    except (TypeError, ValueError) as error:
        error.add_note(
            f"(while reading the column {name!r} of {where}; drop the column to read "
            "the data without it)"
        )
        raise
    return column


def _restored(column: np.ndarray, kind: str) -> np.ndarray:
    """Return a column read back from var with the type it was written from, when
    that type holds its values exactly; otherwise as it was read."""
    if kind == "text":
        if column.dtype.kind == "f" and bool(np.isnan(column).all()):
            missing = np.empty(column.size, dtype=object)  # written as NaN
            return missing
        return column
    try:
        dtype = np.dtype(kind)
    except TypeError:
        return column
    if dtype.kind == "M" and column.dtype == object:
        text = ["NaT" if value is None else value for value in column.tolist()]
        try:
            return np.array(text, dtype=dtype)
        except (TypeError, ValueError):
            return column
    if dtype.kind == "m" and column.dtype.kind in "iu":
        return column.astype(np.int64).view(dtype)
    if column.dtype == dtype or column.dtype.kind not in "biufc":
        return column
    try:
        cast = column.astype(dtype)
    except (TypeError, ValueError, OverflowError):
        return column
    same = np.array_equal(
        cast.astype(column.dtype), column, equal_nan=column.dtype.kind in "fc"
    )
    return cast if same else column


def _type_name(column: np.ndarray) -> str:
    return "text" if column.dtype.kind == "O" else column.dtype.name


def _record(
    cube: Cube, format_name: str, features: list[dict[str, Any]]
) -> dict[str, Any]:
    """The record of an exported cube, for uns: text and integers only."""

    def text(value: Any) -> str:
        return json.dumps(value, allow_nan=False)

    return {
        "format": format_name,
        "format_version": _FORMAT_VERSION,
        "fingerprint": cube.fingerprint(),
        "provenance": text(cube.provenance),
        "normalization": text(cube.normalization),
        "layers": text(
            {name: cube._layers[name].dtype.name for name in cube.layer_names}
        ),
        "n_cells": text(cube.n_cells is not None),
        "samples": text(_table_record(cube.samples)),
        "groups": text(_table_record(cube.groups)),
        "features": text(features),
    }


def _table_record(table: Table) -> dict[str, Any]:
    return {
        "columns": [
            {
                "name": name,
                "type": _type_name(table[name]),
                "values": _encoded(table[name]),
            }
            for name in table.columns
        ]
    }


def _encoded(column: np.ndarray) -> list[Any]:
    """A column's values as JSON: floats that JSON lacks as text."""
    kind = column.dtype.kind
    if kind in "mM":
        values: list[Any] = column.view(np.int64).tolist()
        return values
    if kind == "f":
        return [_json_float(value) for value in column.tolist()]
    if kind == "c":
        return [
            [_json_float(value.real), _json_float(value.imag)]
            for value in column.tolist()
        ]
    values = column.tolist()
    return values


def _json_float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _from_json_float(value: Any) -> float:
    if isinstance(value, str) and value in ("NaN", "Infinity", "-Infinity"):
        return float(value.replace("Infinity", "inf"))
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{value!r} is not a number")
    return float(value)


def _decoded(kind: str, values: Any) -> np.ndarray:
    """A recorded column as an array of its type."""
    if not isinstance(values, list):
        raise ValueError("values must be a list")
    if kind == "text":
        if not all(value is None or isinstance(value, str) for value in values):
            raise ValueError("a text column holds text or null")
        column = np.empty(len(values), dtype=object)
        column[:] = values
        return column
    dtype = np.dtype(kind)
    if dtype.kind == "b":
        if not all(isinstance(value, bool) for value in values):
            raise ValueError("a bool column holds true or false")
        return np.array(values, dtype=bool)
    if dtype.kind in "iumM":
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            raise ValueError(f"a {kind} column holds integers")
        if dtype.kind in "mM":
            return np.array(values, dtype=np.int64).view(dtype)
        return np.array(values, dtype=dtype)
    if dtype.kind == "f":
        return np.array([_from_json_float(value) for value in values], dtype=dtype)
    if dtype.kind == "c":
        if not all(isinstance(value, list) and len(value) == 2 for value in values):
            raise ValueError("a complex column holds pairs of numbers")
        parts = [
            complex(_from_json_float(re), _from_json_float(im)) for re, im in values
        ]
        return np.array(parts, dtype=dtype)
    raise ValueError(f"unknown column type {kind!r}")


def _read_record(uns: Any, container: str) -> dict[str, Any] | None:
    """Return CytePool's record of an exported cube, or None if there is none."""
    stored = uns.get(_RECORD_KEY) if hasattr(uns, "get") else None
    if stored is None:
        return None
    where = f"uns[{_RECORD_KEY!r}]"
    expected = _FORMATS[container]

    def malformed(detail: str) -> ValueError:
        return ValueError(
            f"{where} is not a record of a cube that this version of CytePool reads: "
            f"{detail}. Delete it to read the data without it"
        )

    if not isinstance(stored, Mapping):
        raise malformed(f"it is a {type(stored).__name__}, not a mapping")
    found = stored.get("format")
    if found != expected:
        for kind, name in _FORMATS.items():
            if found == name:
                raise ValueError(
                    f"{where} records a cube that to_{kind}() exported; read the "
                    f"{'MuData' if kind == 'mudata' else 'AnnData'} with from_{kind}()"
                )
        raise malformed(f"its format is {found!r}, not {expected!r}")
    stored_version: Any = stored.get("format_version")
    try:
        version = operator.index(stored_version)
    except TypeError:
        raise malformed("it has no integer format_version") from None
    if version != _FORMAT_VERSION:
        if version > _FORMAT_VERSION:
            raise ValueError(
                f"{where} has format version {version}, newer than the version "
                f"{_FORMAT_VERSION} this version of CytePool reads; update CytePool"
            )
        raise malformed(f"its format version is {version}")
    record: dict[str, Any] = {}
    try:
        fingerprint = stored["fingerprint"]
        if not isinstance(fingerprint, str):
            raise ValueError("fingerprint is not text")
        record["fingerprint"] = str(fingerprint)
        for key in ("provenance", "normalization", *_RECORD_PARTS):
            text = stored[key]
            if not isinstance(text, str):
                raise ValueError(f"{key} is not JSON text")
            record[key] = json.loads(str(text))
        if not isinstance(record["provenance"], dict):
            raise ValueError("provenance is not an object")
        if record["normalization"] is not None and not isinstance(
            record["normalization"], dict
        ):
            raise ValueError("normalization is not an object or null")
        layers = record["layers"]
        if not isinstance(layers, dict) or not set(layers) <= set(LAYER_NAMES):
            raise ValueError("layers does not map cube layer names to types")
        record["layers"] = {name: np.dtype(kind) for name, kind in layers.items()}
        if any(kind.kind not in "iuf" for kind in record["layers"].values()):
            raise ValueError("layers gives a type that is not a number")
        if not isinstance(record["n_cells"], bool):
            raise ValueError("n_cells is not true or false")
        for axis in ("samples", "groups"):
            record[axis] = Table(
                {
                    entry["name"]: _decoded(entry["type"], entry["values"])
                    for entry in record[axis]["columns"]
                }
            )
        for entry in record["features"]:
            if set(entry) != {"name", "column", "type"}:
                raise ValueError("a feature column lacks its name, column or type")
    except (KeyError, TypeError, ValueError) as error:
        raise malformed(str(error)) from None
    return record
