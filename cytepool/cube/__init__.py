"""Pseudobulk cubes: additive statistics per sample, group and feature.

:class:`cytepool.Cube`, :class:`cytepool.CubeCollection`, :func:`cytepool.from_cells`
and :func:`cytepool.from_bulk` are documented at the top level; the modules
:mod:`cytepool.cube.fingerprints` and :mod:`cytepool.cube.provenance` compute the
fingerprints and provenance records of cubes, which
https://docs.nygen.io/CytePool/reference/cube-files specifies.
"""

from .collection import CubeCollection as CubeCollection
from .cube import Cube as Cube
from .pool import from_bulk as from_bulk
from .pool import from_cells as from_cells

# The names above are the top level's (cytepool.Cube and the others).
__all__: list[str] = []
