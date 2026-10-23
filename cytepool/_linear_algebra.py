"""Linear algebra shared by the design checks, normalization and the model fits."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

RANK_TOLERANCE = 1e-7
"""Relative size at or below which a column counts as a combination of others.

A column whose part orthogonal to the earlier kept columns has a norm at most this
fraction of its own norm is aliased. It is the default tolerance of R's ``qr()``,
with which ``lm()``, limma and edgeR decide the rank of a design.
"""


@dataclass(frozen=True)
class ColumnSpace:
    """The columns of a matrix that span its column space, taken in order.

    Attributes:
        kept: Columns that extend the span of the kept columns before them.
        aliased: The other columns: each is a linear combination of the kept
            columns before it.
        basis: ``(n_rows, rank)`` orthonormal basis of the column space.
        remainders: For every column, the norm of its part orthogonal to the kept
            columns before it, as a fraction of its own norm (0 for a zero column).
        scales: Largest absolute value of every column (1 for a zero column).
    """

    kept: tuple[int, ...]
    aliased: tuple[int, ...]
    basis: FloatArray
    remainders: FloatArray
    scales: FloatArray

    @property
    def rank(self) -> int:
        """The dimension of the column space."""
        return len(self.kept)

    def leverages(self) -> FloatArray:
        """The diagonal of the hat matrix, one value per row."""
        result: FloatArray = np.einsum("ij,ij->i", self.basis, self.basis)
        return result


def column_space(matrix: npt.ArrayLike) -> ColumnSpace:
    """Find the columns of a matrix that are linear combinations of earlier ones.

    Columns are taken in order and orthogonalised against the columns kept so far,
    by classical Gram-Schmidt applied twice, which is accurate to rounding. A column
    is aliased when it is zero, when the kept columns already span every row, or
    when the norm of its remainder is at most ``RANK_TOLERANCE`` times its own norm.
    Each column is first divided by its largest absolute value, which changes no
    decision but keeps very large or very small columns from overflowing.

    Args:
        matrix: A finite ``(n_rows, n_columns)`` matrix.

    Returns:
        The kept and aliased columns, an orthonormal basis of their span, and the
        relative remainders and scales of every column.

    Raises:
        ValueError: If ``matrix`` is not two-dimensional or not finite.
    """
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"matrix must be two-dimensional, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite")
    n_rows, n_columns = values.shape
    scales = np.max(np.abs(values), axis=0, initial=0.0)
    scales[scales == 0.0] = 1.0
    scaled = values / scales
    basis = np.zeros((n_rows, min(n_rows, n_columns)))
    kept: list[int] = []
    aliased: list[int] = []
    remainders = np.zeros(n_columns)
    for j in range(n_columns):
        column = scaled[:, j]
        norm = float(np.linalg.norm(column))
        current = basis[:, : len(kept)]
        remainder = column - current @ (current.T @ column)
        remainder -= current @ (current.T @ remainder)
        size = float(np.linalg.norm(remainder))
        remainders[j] = size / norm if norm > 0.0 else 0.0
        if norm == 0.0 or len(kept) == n_rows or size <= RANK_TOLERANCE * norm:
            aliased.append(j)
        else:
            basis[:, len(kept)] = remainder / size
            kept.append(j)
    return ColumnSpace(
        tuple(kept), tuple(aliased), basis[:, : len(kept)], remainders, scales
    )
