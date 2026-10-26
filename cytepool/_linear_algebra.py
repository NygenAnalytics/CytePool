"""Linear algebra shared by the design checks, normalization and the model fits.

``column_space`` finds the columns of a matrix that span its column space by classical
Gram-Schmidt applied twice, which keeps the basis orthonormal to rounding (Giraud,
Langou and Rozloznik 2005), with the rank rule of R's ``qr()``.

``cholesky`` and ``cholesky_solve`` factor and solve the small symmetric positive
definite systems of the count models' Newton steps, of the REML of sample weights and
of the mixed models, compiled, in the order of operations of the column-oriented
algorithm (Golub and Van Loan 2013); ``symmetric`` and ``correlation_matrix`` finish
covariance matrices.

References:
    Giraud L, Langou J, Rozloznik M (2005). The loss of orthogonality in the
    Gram-Schmidt orthogonalization process. Computers & Mathematics with Applications
    50, 1069-1075. doi:10.1016/j.camwa.2005.08.009

    Golub GH, Van Loan CF (2013). Matrix Computations, 4th edition. Johns Hopkins
    University Press, Baltimore.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ._jit import jit

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


@jit
def cholesky(matrix: FloatArray, size: int, floor: float) -> bool:
    """The lower Cholesky factor of ``matrix[:size, :size]``, in place.

    Reads the lower triangle and overwrites it with the factor. Every pivot must be
    above ``floor``: zero, or a bound below which the matrix counts as singular.

    Returns:
        False, with the factor incomplete, when a pivot is not above ``floor``.
    """
    for j in range(size):
        total = matrix[j, j]
        for k in range(j):
            total -= matrix[j, k] * matrix[j, k]
        if not total > floor:
            return False
        diagonal = math.sqrt(total)
        matrix[j, j] = diagonal
        for i in range(j + 1, size):
            total = matrix[i, j]
            for k in range(j):
                total -= matrix[i, k] * matrix[j, k]
            matrix[i, j] = total / diagonal
    return True


@jit
def cholesky_solve(
    factor: FloatArray, size: int, right: FloatArray, solution: FloatArray
) -> None:
    """Solve L L' solution = right for the Cholesky factor L in the lower triangle of
    ``factor[:size, :size]``: forward, then backward substitution, dividing by the
    pivots."""
    for i in range(size):
        total = right[i]
        for j in range(i):
            total -= factor[i, j] * solution[j]
        solution[i] = total / factor[i, i]
    for i in range(size - 1, -1, -1):
        total = solution[i]
        for j in range(i + 1, size):
            total -= factor[j, i] * solution[j]
        solution[i] = total / factor[i, i]


def symmetric(matrices: FloatArray) -> FloatArray:
    """Average each trailing square matrix with its transpose."""
    result: FloatArray = (matrices + np.swapaxes(matrices, -1, -2)) / 2.0
    return result


def correlation_matrix(
    covariance: FloatArray, *, unit_diagonal: bool = False
) -> FloatArray:
    """Correlations of a covariance matrix, zero for a coefficient of zero variance.

    A coefficient with zero variance contributes nothing to a contrast's variance
    whatever its correlations, so they are set to zero. With ``unit_diagonal``, its
    correlation with itself is 1: limma gives such a coefficient (an all-zero
    contrast) correlations of zero and keeps it in the rank of the F-test (found by
    probing).
    """
    scale = np.sqrt(np.diag(covariance))
    constant = scale == 0.0
    safe = np.where(constant, 1.0, scale)
    correlations: FloatArray = covariance / np.outer(safe, safe)
    correlations[constant, :] = 0.0
    correlations[:, constant] = 0.0
    if unit_diagonal:
        correlations[constant, constant] = 1.0
    return correlations


@jit
def orthogonalise(
    vector: FloatArray, gene_basis: FloatArray, rank: int, m: int
) -> None:
    """Remove from ``vector[:m]`` its projection on ``gene_basis[:rank, :m]``, twice.

    Modified Gram-Schmidt with one reorthogonalisation: the projection on each row of
    the orthonormal basis is subtracted in turn, and the whole is done twice, which
    keeps the result orthogonal to the basis to rounding (Giraud, Langou and Rozloznik
    2005).
    """
    for _ in range(2):
        for a in range(rank):
            dot = 0.0
            for t in range(m):
                dot += gene_basis[a, t] * vector[t]
            for t in range(m):
                vector[t] -= dot * gene_basis[a, t]


@jit
def householder_qr(
    work: FloatArray,
    n_rows: int,
    n_columns: int,
    tolerance: float,
    current_norm: bool,
    zero_pivot: float,
    norms: FloatArray,
    aliased: npt.NDArray[np.bool_],
    kept: npt.NDArray[np.int64],
    products: FloatArray,
) -> int:
    """Householder QR decomposition of the first ``n_rows`` rows of the first
    ``n_columns`` columns of ``work``, in place (Golub and Van Loan 2013).

    The columns of ``work`` after the first ``n_columns``, such as a response, are
    reflected with the others but never pivoted. Columns are taken in order. One whose
    part orthogonal to the columns kept before it has a norm of zero, or of at most
    ``tolerance`` times a reference, is aliased and skipped: the reference is the
    column's norm before any reflection, as in R's ``qr()``, or with ``current_norm``
    its norm after the earlier reflections, which is the same in exact arithmetic.
    The diagonal element of R is -norm for a head (the element in the pivot row) of
    at least -``zero_pivot`` times the norm and +norm below, the sign of R's ``qr()``
    when ``zero_pivot`` is 0: a positive ``zero_pivot`` counts a head that small as
    zero, which it is in exact arithmetic in some designs, so that the sign does not
    follow rounding noise. Each reflection, I - 2 v v' / (v'v) for the column's part
    v from the pivot row down, is applied to every other column, those kept before it
    being zero where it acts, with v'v = 2 norm (norm + |head|). Sums over rows run in
    increasing order.

    Args:
        work: The columns, a 2-D array with one row per observation.
        n_rows: The rows to use, from the first.
        n_columns: The columns to decompose, from the first.
        tolerance: The relative norm at or below which a column is aliased.
        current_norm: Whether the reference norm is that after the earlier
            reflections.
        zero_pivot: The relative size below which a head counts as zero.
        norms: Workspace of ``n_columns`` values: the norms of the columns.
        aliased: Workspace of ``n_columns`` values: whether each column is aliased.
        kept: Workspace of ``n_columns`` values: the kept columns, in order.
        products: Workspace of one value per column of ``work``.

    Returns:
        The rank: the number of kept columns. On return, rows ``0:rank`` of the kept
        column ``kept[t]`` hold column t of the triangular factor R, those of an
        aliased column hold its coordinates in the orthonormal basis of the kept
        columns, and rows ``rank:n_rows`` of a column after the first ``n_columns``
        hold its residual effects.
    """
    n_total = work.shape[1]
    if not current_norm:
        norms[:] = 0.0
        for i in range(n_rows):
            for j in range(n_columns):
                norms[j] += work[i, j] * work[i, j]
        for j in range(n_columns):
            norms[j] = math.sqrt(norms[j])
    rank = 0
    for j in range(n_columns):
        aliased[j] = True
        if rank == n_rows:
            continue
        if current_norm:
            total = 0.0
            for i in range(n_rows):
                total += work[i, j] * work[i, j]
            reference = math.sqrt(total)
        else:
            reference = norms[j]
        total = 0.0
        for i in range(rank, n_rows):
            total += work[i, j] * work[i, j]
        norm = math.sqrt(total)
        if norm == 0.0 or norm <= tolerance * reference:
            continue
        aliased[j] = False
        head = work[rank, j]
        alpha = -norm if head >= -zero_pivot * norm else norm
        work[rank, j] = head - alpha
        # v'v = 2 norm (norm + |head|).
        scale = 1.0 / (norm * (norm + abs(head)))
        products[:] = 0.0
        for i in range(rank, n_rows):
            v = work[i, j]
            for k in range(n_total):
                products[k] += v * work[i, k]
        for k in range(n_total):
            products[k] *= scale
        products[j] = 0.0
        for i in range(rank, n_rows):
            v = work[i, j]
            for k in range(n_total):
                work[i, k] -= products[k] * v
        work[rank, j] = alpha
        for i in range(rank + 1, n_rows):
            work[i, j] = 0.0
        kept[rank] = j
        rank += 1
    return rank
