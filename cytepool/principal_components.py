"""Principal components of samples, as DESeq2's ``plotPCA`` computes them.

:func:`principal_components` places the samples of a matrix of log-expression values, or
of one group (cell type) of a cube, on their principal components (Pearson 1901), as
DESeq2's ``plotPCA`` does for its transformed values (Love et al. 2014): from the
``top`` genes of largest variance over the samples, each centred on its mean, by the
singular value decomposition of that samples x genes matrix, X = U S V'. The
coordinates of the samples are U S, the loadings of the genes are V, and the share of
the variance that each component explains is its squared singular value over their
sum, DESeq2's ``percentVar``. :func:`~cytepool.plotting.plot_pca` draws them.

What DESeq2's help page leaves open was found by running ``plotPCA`` 1.46.0 as a black
box: it ranks the genes by their variance over the samples, keeps genes of equal
variance in their order, takes every gene when ``ntop`` exceeds their number, and
returns the coordinates of R's ``prcomp``, whose signs, its help page says, are
arbitrary. Here the sign of each component makes the sample with its largest absolute
coordinate positive (of near ties, within a relative 1e-9, the first), as
:func:`~cytepool.plotting.plot_mds` orients its dimensions, so that the result does
not depend on the order of the genes or the sign of the values.

The values:
    A matrix holds one row per sample and one column per gene, finite, such as
    log-CPM values or the variance-stabilized values that DESeq2's guide takes. A
    :class:`~cytepool.stats.voom.VoomResult` gives its log-CPM values. A cube gives
    one group's samples prepared as :func:`~cytepool.plotting.plot_mds` prepares them,
    as the edgeR User's Guide prepares counts for ``plotMDS``: the samples with fewer
    than ``min_cells`` cells or no counts are left out, as
    :func:`~cytepool.differential_expression.limma_voom` leaves them out; the genes are
    kept by :func:`~cytepool.normalization.filter_by_expression` (Chen, Lun and Smyth
    2016) with the samples as one group; and the values are their log2 counts per
    million with TMM normalization factors and an average count of 2 added
    (:func:`~cytepool.normalization.log_cpm`).

References:
    Pearson K (1901). On lines and planes of closest fit to systems of points in space.
    The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 2,
    559-572. doi:10.1080/14786440109462720

    Love MI, Huber W, Anders S (2014). Moderated estimation of fold change and
    dispersion for RNA-seq data with DESeq2. Genome Biology 15, 550.
    doi:10.1186/s13059-014-0550-8

    Chen Y, Lun ATL, Smyth GK (2016). From reads to genes to pathways: differential
    expression analysis of RNA-Seq experiments using Rsubread and the edgeR
    quasi-likelihood pipeline. F1000Research 5, 1438. doi:10.12688/f1000research.8987.2
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from ._columns import text_column
from ._masked_arrays import read_only, real_array, reject_masked, unmasked_array
from ._messages import quoted
from ._options import integral_number, positive_integer
from .cube import Cube
from .normalization import filter_by_expression, log_cpm, tmm_factors
from .stats.voom import VoomResult
from .table import Table

__all__ = ["PrincipalComponents", "principal_components"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

# DESeq2's plotPCA takes the 500 genes of largest variance.
_TOP = 500
# A cube's samples need this many cells, as limma_voom leaves samples out, and its
# log-CPM values add an average count of 2, as edgeR's plotMDS does.
_MIN_CELLS = 10
_PRIOR_COUNT = 2.0
# Coordinates within this relative distance of a component's largest absolute one tie
# for it; the first of them sets the sign.
_NEAR_TIE = 1e-9
# Centred values below this multiple of the largest value are rounding errors.
_ROUNDING = 16.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class PrincipalComponents:
    """The samples on their principal components.

    Rows of ``coordinates`` are the samples used, in the order of ``samples``; columns
    are the components, in decreasing order of the variance they explain. There are
    min(n_samples - 1, n_genes) components: the centred values have no more
    dimensions. The arrays are read-only.

    Attributes:
        coordinates: ``(n_samples, n_components)`` float64, the samples' coordinates,
            the centred values times the loadings; DESeq2's ``PC1``, ``PC2``, ... up
            to sign.
        variance_explained: ``(n_components,)`` float64, each component's share of the
            total variance of the genes used, summing to 1; DESeq2's ``percentVar``.
        loadings: ``(n_genes_used, n_components)`` float64, the unit vectors of the
            components over the genes used, in the order of ``genes``.
        genes: ``(n_genes_used,)`` positions of the genes used, in decreasing order of
            their variance: columns of the matrix, or features of the cube.
        samples: One row per sample used: ``sample_id`` and, for a cube, the columns
            of its sample table; for a matrix, the ids ``"1"``, ``"2"``, ... of its
            rows.
        settings: ``top``, and for a cube its ``group`` and ``min_cells``.
    """

    coordinates: FloatArray
    variance_explained: FloatArray
    loadings: FloatArray
    genes: IntArray
    samples: Table
    settings: Mapping[str, Any]

    @property
    def n_components(self) -> int:
        """The number of components."""
        return int(self.coordinates.shape[1])

    def __repr__(self) -> str:
        n_samples, n_components = self.coordinates.shape
        shares = ", ".join(f"{100 * v:.1f}%" for v in self.variance_explained[:3])
        return (
            f"PrincipalComponents(n_samples={n_samples}, n_genes={self.genes.size}, "
            f"n_components={n_components}, variance_explained=[{shares}"
            f"{', ...' if n_components > 3 else ''}])"
        )


def principal_components(
    data: Any,
    *,
    group: str | None = None,
    top: int = _TOP,
    min_cells: int | None = None,
) -> PrincipalComponents:
    """Place samples on the principal components of their most variable genes.

    As DESeq2's ``plotPCA``: the ``top`` genes of largest variance over the samples
    (every gene when there are fewer, genes of equal variance in their order), each
    centred on its mean over the samples, decomposed by singular values. The module's
    docstring describes the values a cube gives and the sign of each component.

    Args:
        data: ``(n_samples, n_genes)`` log-expression values, finite; a
            :class:`~cytepool.stats.voom.VoomResult`, whose log-CPM values are used;
            or a :class:`~cytepool.cube.Cube`, of which one group's samples are used.
        group: The group (cell type) of a cube; it may be left out when the cube has
            one group. Only cubes take it.
        top: The number of genes of largest variance used, at least 1 (DESeq2's
            ``ntop``).
        min_cells: The fewest cells a sample of a cube needs in the group, 10 by
            default as for ``limma_voom``; for a cube that records cell numbers.

    Returns:
        The coordinates, shares of variance and loadings, with the genes and samples
        used.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If the values are not finite or not 2-D, fewer than two samples
            remain, the group of a cube of several groups is not given, ``group`` or
            ``min_cells`` is given for data other than a cube, no gene of a cube
            passes the filter, the genes used do not vary, or an option is out of
            range.
        KeyError: If the group is not in the cube.
    """
    count = positive_integer(top, "top")
    values, genes, samples, settings = _values(data, group, min_cells)
    n_samples, n_genes = values.shape
    if n_samples < 2:
        raise ValueError(
            f"principal components need at least two samples, got {n_samples}"
        )
    # The genes of largest variance; equal variances keep the genes' order (found by
    # probing DESeq2 1.46.0).
    variances = np.var(values, axis=0, ddof=1)
    chosen = np.argsort(-variances, kind="stable")[: min(count, n_genes)]
    used = values[:, chosen]
    centred = used - used.mean(axis=0)
    left, singular, right = np.linalg.svd(centred, full_matrices=False)
    n_components = min(n_samples - 1, chosen.size)
    singular = singular[:n_components]
    total = float(np.sum(singular**2))
    # Centred values that are rounding errors of their means, as constant genes
    # leave them, do not vary.
    noise = (_ROUNDING * float(np.max(np.abs(used)))) ** 2 * centred.size
    if not total > noise:
        raise ValueError(
            "the values of the genes used do not vary between the samples, so they "
            "have no principal components"
        )
    coordinates = left[:, :n_components] * singular
    loadings = right[:n_components].T.copy()
    signs = _signs(coordinates)
    coordinates *= signs
    loadings *= signs
    settings["top"] = count
    return PrincipalComponents(
        coordinates=read_only(np.ascontiguousarray(coordinates)),
        variance_explained=read_only(singular**2 / total),
        loadings=read_only(np.ascontiguousarray(loadings)),
        genes=read_only(genes[chosen]),
        samples=samples,
        settings=MappingProxyType(settings),
    )


def _signs(coordinates: FloatArray) -> FloatArray:
    """The sign of each component that makes its largest absolute coordinate positive;
    of near ties, the first."""
    magnitude = np.abs(coordinates)
    largest = np.argmax(magnitude >= magnitude.max(axis=0) * (1.0 - _NEAR_TIE), axis=0)
    signs = np.sign(coordinates[largest, np.arange(coordinates.shape[1])])
    result: FloatArray = np.where(signs == 0.0, 1.0, signs)
    return result


def _values(
    data: Any, group: Any, min_cells: Any
) -> tuple[FloatArray, IntArray, Table, dict[str, Any]]:
    """The values (samples x genes), the positions of their genes among the data's,
    the samples and the settings that chose them."""
    if isinstance(data, Cube):
        return _cube_values(data, group, min_cells)
    if group is not None or min_cells is not None:
        given = "group" if group is not None else "min_cells"
        if given == "group" and not isinstance(group, str):
            reject_masked(group, "group")
        raise ValueError(
            f"{given} applies to a cube, and data is a {type(data).__name__}"
        )
    source = data.log_cpm if isinstance(data, VoomResult) else data
    values = real_array(unmasked_array(source, "data"), "data", boolean=False)
    if values.ndim != 2:
        raise ValueError(
            "data must be a Cube, a VoomResult or a 2-D array of shape "
            f"(n_samples, n_genes), got shape {values.shape}"
        )
    if values.shape[1] == 0:
        raise ValueError("data has no genes")
    if not np.isfinite(values).all():
        raise ValueError(
            "data must be finite; leave out the genes with missing values first"
        )
    n_samples = values.shape[0]
    samples = Table({"sample_id": text_column([str(i + 1) for i in range(n_samples)])})
    return values, np.arange(values.shape[1]), samples, {}


def _cube_values(
    cube: Cube, group: Any, min_cells: Any
) -> tuple[FloatArray, IntArray, Table, dict[str, Any]]:
    """A cube group's log-CPM values, of its samples with enough cells and counts,
    over the genes that filterByExpr keeps."""
    fewest = (
        _MIN_CELLS if min_cells is None else integral_number(min_cells, "min_cells")
    )
    if fewest < 0:
        raise ValueError(f"min_cells must be at least 0, got {fewest}")
    ids = cube.groups["group_id"].tolist()
    if group is None:
        if cube.n_groups != 1:
            raise ValueError(
                f"the cube has {cube.n_groups} groups; name the one to use with group, "
                f"one of {quoted(ids)}"
            )
        group = ids[0]
    if not isinstance(group, str):
        reject_masked(group, "group")
        raise TypeError(f"group must be a group id (str), got {type(group).__name__}")
    counts = np.asarray(cube.layer("counts", group), dtype=np.float64)
    used = counts.sum(axis=1) > 0.0
    if cube.n_cells is not None:
        used &= np.asarray(cube.n_cells)[:, ids.index(group)] >= fewest
    rows = np.flatnonzero(used)
    if rows.size < 2:
        raise ValueError(
            f"{rows.size} sample(s) of group {group!r} have counts and at least "
            f"{fewest} cells; principal components need two or more"
        )
    kept = counts[rows]
    keep = filter_by_expression(kept, group=np.zeros(rows.size))
    if not keep.any():
        raise ValueError(
            f"no gene of group {group!r} passes filter_by_expression, so there is "
            "nothing to decompose"
        )
    kept = kept[:, keep]
    values = log_cpm(kept, norm_factors=tmm_factors(kept), prior_count=_PRIOR_COUNT)
    settings: dict[str, Any] = {"group": group, "min_cells": fewest}
    return values, np.flatnonzero(keep), cube.samples.take(rows), settings
