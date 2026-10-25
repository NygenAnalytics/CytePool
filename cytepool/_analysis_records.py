"""The records an analysis keeps about its inputs and the software that ran it.

The result of a workflow on a cube, such as ``limma_voom`` or ``propeller``, holds a
provenance record. These functions build the parts that every workflow records alike:
the cube it read, identified by its fingerprint and the digests behind it; the design
it fitted, with the contrasts it tested; and the versions of the software.
"""

import platform
from typing import Any

import numba
import numpy as np

from . import __version__
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


def software_versions() -> dict[str, str]:
    """The versions of CytePool, numpy, numba and Python that run an analysis."""
    return {
        "cytepool": __version__,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "python": platform.python_version(),
    }
