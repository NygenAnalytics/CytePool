"""The records an analysis keeps about its inputs and the software that ran it.

The result of a workflow on a cube, such as ``limma_voom`` or ``propeller``, holds a
provenance record. These functions build the parts that every workflow records alike:
the cube it read, identified by its fingerprint and the digests it is made of; the design
it fitted, with the contrasts it tested; and the versions of the software.
"""

import json
import platform
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numba
import numpy as np

from . import __version__
from ._messages import listing, quoted
from .cube import Cube
from .design import Contrasts, Design


def cube_record(cube: Cube) -> dict[str, Any]:
    """The part of a result's provenance that identifies the cube it came from.

    It holds the cube's fingerprint, the digests it is built from (those of the
    tables, ``n_cells``, the normalization and each group of each layer, with which
    a cube file can be checked group by group), and the cube's provenance. A
    workflow builds it after its fits, so that the whole groups they read count
    towards the digests. For a cube file this reads nothing; a cube in memory hashes
    its layers once (see "Provenance" in
    ``cytepool.differential_expression.limma_voom`` for the cost).
    """
    digests = cube.digests()
    return {
        "fingerprint": digests["fingerprint"],
        "digests": digests,
        "provenance": cube.provenance,
    }


def design_record(design: Design, contrasts: Contrasts) -> dict[str, Any]:
    """The part of a result's provenance that describes the design and contrasts.

    It holds the formula, the columns, the levels of every factor, the values of the
    variables of the formula for every sample, with its ``sample_id``, and the
    weights of every contrast on the columns.
    """
    samples = design.samples
    variables = ["sample_id", *design.model_matrix.variables]
    return {
        "formula": design.formula,
        "columns": list(design.columns),
        "levels": {name: list(levels) for name, levels in design.levels.items()},
        "samples": {name: samples[name].tolist() for name in variables},
        "contrasts": {
            name: contrasts.matrix[:, position].tolist()
            for position, name in enumerate(contrasts.names)
        },
    }


def mixed_design_record(design: Design, contrasts: Contrasts) -> dict[str, Any]:
    """``design_record`` with the random effects, as ``"random"``, and the values of
    the variables they group by for every sample, among the ``"samples"``."""
    record = design_record(design, contrasts)
    record["random"] = [str(effect) for effect in design.random]
    samples = design.samples
    for effect in design.random:
        for name in effect.grouping:
            record["samples"].setdefault(name, samples[name].tolist())
    return record


def keep_provenance(
    record: Any, method: str, cube: Cube, settings: Mapping[str, Any]
) -> None:
    """Give the record of scores of a cube's group, such as ``PathwayScores``, its
    provenance: the method, the software, the cube's fingerprint, digests and
    provenance, the layer read and the settings. ``provenance_of`` reads it back.

    The record keeps it as JSON text in its own attributes, outside its fields, so
    that a record built from fields alone, or copied with ``dataclasses.replace``,
    has none.
    """
    text = json.dumps(
        {
            "method": method,
            "versions": software_versions(),
            "cube": cube_record(cube),
            "layer": "counts",
            "settings": dict(settings),
        },
        allow_nan=False,
    )
    object.__setattr__(record, "_provenance", text)


def provenance_of(record: Any) -> dict[str, Any] | None:
    """The provenance that ``keep_provenance`` gave a record, as a new dict, or
    None."""
    text = vars(record).get("_provenance")
    if text is None:
        return None
    found: dict[str, Any] = json.loads(text)
    return found


def provenance_text(record: Any) -> str | None:
    """The JSON text of the provenance that ``keep_provenance`` gave a record, or
    None: what a pickled record carries, for ``restore_provenance``."""
    text: str | None = vars(record).get("_provenance")
    return text


def restore_provenance[R](record: R, text: str | None) -> R:
    """Give a record the provenance it was pickled with, if it had one."""
    if text is not None:
        object.__setattr__(record, "_provenance", text)
    return record


def only_key(value: Any, available: list[str], argument: str, what: str) -> Any:
    """The key of a result's table: ``value`` when given, else the only one.

    Results that keep one table per group, contrast or both let a key be left out
    when they have one. ``what`` names the keys in the message, such as "groups".

    Raises:
        ValueError: If ``value`` is None and there is not exactly one key.
    """
    if value is not None:
        return value
    if len(available) == 1:
        return available[0]
    raise ValueError(
        f"the result has {len(available)} {what}; name one with {argument}"
        + (f", one of {quoted(available)}" if available else "")
    )


def absence(result: Any, group: str, contrast: str) -> str:
    """Why a differential-expression result has no table of a group and contrast, in
    the words of its own ``table`` method."""
    try:
        result.table(group, contrast)
    except KeyError as error:
        return str(error.args[0])
    raise AssertionError("a table that is not in result.tables")  # pragma: no cover


def study_keys(
    results: Mapping[str, Any], group: Any, contrast: Any
) -> tuple[Any, Any]:
    """The group and contrast of the studies' differential-expression results asked
    for, or, for one left out, the only one that the studies analysed.

    Raises:
        ValueError: If a key is left out and the studies analysed several, or none.
    """
    return (
        only_study_key(results, group, "group", "groups"),
        only_study_key(results, contrast, "contrast", "contrasts"),
    )


def only_study_key(
    results: Mapping[str, Any], value: Any, argument: str, what: str
) -> Any:
    """``value`` when given, else the only group (``what="groups"``), or contrast
    (``what="contrasts"``), that the studies' results analysed.

    Raises:
        ValueError: If ``value`` is None and the studies analysed several, or none.
    """
    if value is not None:
        return value
    available = study_union(results, what)
    if len(available) != 1:
        raise ValueError(
            f"the studies analysed {len(available)} {what}; name one with "
            f"{argument}, one of {quoted(available)}"
        )
    return available[0]


def study_union(results: Mapping[str, Any], what: str) -> list[str]:
    """The groups (``what="groups"``), or the contrasts and F-tests, of any study's
    result, in the order in which the studies first have them."""
    if what == "groups":
        return list(
            dict.fromkeys(
                group for result in results.values() for group in result.groups
            )
        )
    return list(
        dict.fromkeys(
            name
            for result in results.values()
            for name in (*result.contrasts.names, *result.f_tests)
        )
    )


def software_versions() -> dict[str, str]:
    """The versions of CytePool, numpy, numba and Python that run an analysis."""
    return {
        "cytepool": __version__,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "python": platform.python_version(),
    }


def matched_samples(
    cube: Cube, design: Design, *, no_ids: str, text_ids: str, data: str, data_ids: str
) -> tuple[np.ndarray, np.ndarray]:
    """The position of every design sample in the cube, increasing, and its design row.

    The workflows word the messages for what they read: ``no_ids`` is the whole
    message for a sample table without a 'sample_id' column; ``text_ids`` ends the
    one for ids that are not text ("as ..."); ``data`` names the data and ``data_ids``
    its sample ids in the one for samples it does not have.

    Raises:
        TypeError, ValueError: If the design's samples cannot be matched to those of
            the data by ``sample_id``.
    """
    table = design.samples
    if "sample_id" not in table:
        raise ValueError(no_ids)
    ids = table["sample_id"]
    if ids.dtype != object:
        raise TypeError(
            f"the design's 'sample_id' column must hold text ids, as {text_ids}, got "
            f"dtype {ids.dtype}"
        )
    values = ids.tolist()
    blank = [row for row, value in enumerate(values) if value is None]
    if blank:
        raise ValueError(
            f"the design's 'sample_id' column is missing in row(s) {listing(blank)}; "
            "every sample needs an id"
        )
    repeated = [value for value, count in Counter(values).items() if count > 1]
    if repeated:
        raise ValueError(
            f"the design's sample table repeats the sample id(s) {quoted(repeated)}; "
            "each sample must appear once"
        )
    index = {value: i for i, value in enumerate(cube.samples["sample_id"].tolist())}
    unknown = [value for value in values if value not in index]
    if unknown:
        raise ValueError(
            f"{len(unknown)} of the design's {len(values)} samples are not samples of "
            f"{data}: {quoted(unknown)}; the design's 'sample_id' must hold "
            f"{data_ids}, such as {quoted(list(index)[:3])}"
        )
    positions = np.array([index[value] for value in values], dtype=np.intp)
    order = np.argsort(positions, kind="stable")
    return positions[order], order
