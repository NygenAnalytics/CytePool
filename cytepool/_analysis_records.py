"""The records an analysis keeps about its inputs and the software that ran it.

The result of a workflow on a cube, such as ``limma_voom`` or ``propeller``, holds a
provenance record. These functions build the parts that every workflow records alike:
the cube it read, identified by its fingerprint and the digests behind it; the design
it fitted, with the contrasts it tested; and the versions of the software.
"""

import platform
from collections import Counter
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
