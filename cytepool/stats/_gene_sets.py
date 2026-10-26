"""The inputs and effects that the gene-set tests and the rotation tests share.

Expression values with their weights, gene sets matched to gene ids, the design
reparametrised so that the contrast is its last coefficient (limma's
``contrastAsCoef``, by a Householder reflection; Golub and Van Loan 2013), and the
effects of every gene: the contrast effect and the residual effects of its weighted
least squares fit, which camera (Wu and Smyth 2012), fry, roast and mroast (Wu et al.
2010) and romer test.

References:
    Wu D, Smyth GK (2012). Camera: a competitive gene set test accounting for inter-gene
    correlation. Nucleic Acids Research 40, e133. doi:10.1093/nar/gks461

    Wu D, Lim E, Vaillant F, Asselin-Labat ML, Visvader JE, Smyth GK (2010). ROAST:
    rotation gene set tests for complex microarray experiments. Bioinformatics 26,
    2176-2182. doi:10.1093/bioinformatics/btq401

    Golub GH, Van Loan CF (2013). Matrix Computations, 4th edition. Johns Hopkins
    University Press, Baltimore.
"""

import math
import numbers
from collections.abc import Mapping, Set
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._linear_algebra import RANK_TOLERANCE, householder_qr
from cytepool._masked_arrays import real_array, reject_masked, unmasked_array
from cytepool.design._contrast_weights import is_named_contrasts, weights_on_columns
from cytepool.design._design_matrices import (
    design_matrix,
    refuse_random_effects,
    require_full_rank,
)
from cytepool.design.design import Design
from cytepool.stats.voom import VoomResult

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

# A pivot of the QR decomposition this small relative to the norm of its column is
# zero up to rounding; see _effects_of_last_coefficient.
_ZERO_PIVOT = 1e-12


def contrast_design(
    values: FloatArray, design: Any, contrast: Any, function: str
) -> tuple[FloatArray, int]:
    """The design, checked, with the contrast as its last coefficient, and d."""
    refuse_random_effects(design, function)
    x, names = design_matrix(design, None)
    n_samples, n_coefficients = x.shape
    if n_samples != values.shape[0]:
        raise ValueError(
            f"y has {values.shape[0]} rows but the design has {n_samples}; rows "
            "of both are samples"
        )
    require_full_rank(x, names)
    d = n_samples - n_coefficients
    if d < 1:
        raise ValueError(
            f"the design has {n_coefficients} columns for {n_samples} samples and "
            f"leaves no residual degrees of freedom, which {function} needs to "
            "estimate the variances"
        )
    return _reparametrised_design(x, _contrast(contrast, design, names)), d


def gene_effects(
    values: FloatArray,
    reparametrised: FloatArray,
    weights: FloatArray | None,
    *,
    signed_residuals: bool = False,
) -> FloatArray:
    """The ``(n_genes, d + 1)`` effects of each gene's fit of the design.

    ``signed_residuals`` multiplies the residual effects, like the contrast effect, by
    the sign of the last diagonal element of R: the residual basis of limma's romer
    (found by probing), which camera, fry and roast do not use.
    """
    n_samples = reparametrised.shape[0]
    if weights is None:
        weight_rows = np.ones((1, n_samples))
    elif weights.ndim == 1:
        weight_rows = weights[np.newaxis, :]
    else:
        weight_rows = weights.T
    effects: FloatArray = _effects_of_last_coefficient(
        np.ascontiguousarray(values.T),
        np.ascontiguousarray(weight_rows),
        np.ascontiguousarray(reparametrised),
        RANK_TOLERANCE,
        signed_residuals,
    )
    deficient = np.flatnonzero(np.isnan(effects[:, 0]))
    if deficient.size:
        raise ValueError(
            f"the weights make the design rank deficient for {deficient.size} gene(s), "
            f"such as gene {deficient[0]}; weights that differ by many orders of "
            "magnitude between samples cannot be used"
        )
    return effects


@jit
def _effects_of_last_coefficient(
    responses: FloatArray,
    weight_rows: FloatArray,
    design: FloatArray,
    tolerance: float,
    signed_residuals: bool,
) -> FloatArray:
    """Contrast and residual effects of each gene's weighted least squares fit.

    Householder QR decomposition of each gene's whitened design, in the sign convention
    of R's ``qr()``: column j is reflected onto -s r e_j, with r the norm of its part
    from row j on and s the sign of its element in row j, or s = 1 when that element is
    zero. An element below ``_ZERO_PIVOT`` times r counts as zero: it is zero in exact
    arithmetic, as in an indicator design with samples sorted by group, and its sign
    is rounding noise, which would otherwise decide the basis of the residual effects.

    Args:
        responses: ``(n_genes, n_samples)`` values.
        weight_rows: ``(n_genes, n_samples)`` precision weights, or one row shared by
            all genes; positive.
        design: ``(n_samples, p)`` design whose last coefficient is the contrast.
        tolerance: Relative norm at or below which a column is aliased.
        signed_residuals: Whether the residual effects are multiplied by the same
            sign as the contrast effect.

    Returns:
        ``(n_genes, n_samples - p + 1)`` effects: the p-th element of Q'y for the
        whitened values y, times the sign of the p-th diagonal element of R, so that it
        has the sign of the estimated contrast, then the n - p residual effects. A row
        is NaN when the whitened design of that gene is rank deficient.
    """
    n_genes, n_samples = responses.shape
    p = design.shape[1]
    effects = np.empty((n_genes, n_samples - p + 1))
    work = np.empty((n_samples, p + 1))
    norms = np.empty(p)
    aliased = np.empty(p, dtype=np.bool_)
    kept = np.empty(p, dtype=np.int64)
    products = np.empty(p + 1)
    shared = weight_rows.shape[0] == 1
    for g in range(n_genes):
        w = 0 if shared else g
        for i in range(n_samples):
            scale = math.sqrt(weight_rows[w, i])
            for j in range(p):
                work[i, j] = scale * design[i, j]
            work[i, p] = scale * responses[g, i]
        # The norm after the earlier reflections is the reference of the rank rule, a
        # head within _ZERO_PIVOT of zero counts as zero, and v'v is summed.
        rank = householder_qr(
            work, n_samples, p, tolerance, True, _ZERO_PIVOT, True, norms, aliased,
            kept, products,
        )  # fmt: skip
        if rank < p:
            effects[g, :] = np.nan  # aliased: the design is rank deficient
            continue
        # The diagonal element of R of the contrast, by R's sign convention.
        sign = -1.0 if work[p - 1, p - 1] < 0.0 else 1.0
        effects[g, 0] = sign * work[p - 1, p]
        residual_sign = sign if signed_residuals else 1.0
        for i in range(p, n_samples):
            effects[g, i - p + 1] = residual_sign * work[i, p]
    return effects


def _contrast(contrast: Any, design: Any, names: tuple[str, ...]) -> int | FloatArray:
    """The contrast: the position of a design column, or weights on the columns."""
    reject_masked(contrast, "contrast")
    n_columns = len(names)
    if isinstance(contrast, str):
        if isinstance(design, Design) and contrast in design.contrasts:
            weights = np.array(design.contrasts[contrast], dtype=np.float64)
            return _checked_weights(weights, contrast)
        if contrast in names:
            return names.index(contrast)
        named = list(design.contrasts.names) if isinstance(design, Design) else []
        raise ValueError(
            f"contrast {contrast!r} is neither a named contrast of the design nor "
            "one of its columns; contrasts: "
            f"{', '.join(named) if named else '(none)'}; columns: {', '.join(names)}"
        )
    if isinstance(contrast, np.ndarray) and contrast.ndim == 0:
        contrast = contrast.item()  # a 0-d array is its value
    if isinstance(contrast, (bool, np.bool_)) or (
        isinstance(contrast, numbers.Real)
        and not isinstance(contrast, numbers.Integral)
    ):
        raise TypeError(
            "contrast must be a contrast name, a column name or position (an integer), "
            f"or weights on the design columns, got {type(contrast).__name__}"
        )
    if isinstance(contrast, numbers.Integral):
        position = int(contrast)
        if not -n_columns <= position < n_columns:
            raise ValueError(
                f"contrast {position} is not the position of a design column; the "
                f"design has {n_columns} columns, positions 0 to {n_columns - 1}"
            )
        return position % n_columns
    if isinstance(contrast, Mapping) and not is_named_contrasts(contrast):
        return _weights_by_column(contrast, names)
    remedy = "give one weight per design column, in their order or named by them"
    weights = weights_on_columns(
        contrast,
        names,
        label="the contrast",
        target="the design's columns",
        remedy=remedy,
    )
    try:
        vector = np.array(weights, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(
            "contrast must be a contrast name, a column name or position, or weights "
            f"on the design columns, got {type(contrast).__name__}"
        ) from None
    if vector.ndim == 2 and vector.shape[1] == 1:
        vector = vector[:, 0]
    elif vector.ndim == 2:
        raise ValueError(
            f"{vector.shape[1]} contrasts were given; one contrast is tested at a "
            "time, so select one, for example with contrasts.select(name)"
        )
    if vector.shape != (n_columns,):
        raise ValueError(
            f"the contrast must have one weight per design column ({n_columns}: "
            f"{', '.join(names)}), got shape {vector.shape}"
        )
    return _checked_weights(vector, "the contrast")


def _weights_by_column(
    weights: Mapping[Any, Any], names: tuple[str, ...]
) -> FloatArray:
    """Weights given as a mapping of column name to weight; columns left out weigh 0."""
    vector = np.zeros(len(names))
    for column, weight in weights.items():
        if column not in names:
            raise ValueError(
                f"the contrast weighs {column!r}, which is not a design column; "
                f"columns are: {', '.join(names)}"
            )
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, numbers.Real):
            raise TypeError(
                "a contrast given as a mapping takes one number per design column, got "
                f"{type(weight).__name__} for {column!r}"
            )
        vector[names.index(column)] = float(weight)
    return _checked_weights(vector, "the contrast")


def _checked_weights(vector: FloatArray, label: str) -> FloatArray:
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} has weights that are not finite")
    if not np.any(vector != 0.0):
        raise ValueError(f"{label} is zero on every column, so it compares nothing")
    return vector


def _reparametrised_design(x: FloatArray, contrast: int | FloatArray) -> FloatArray:
    """The design with the contrast as its last coefficient, as limma forms it.

    A column moves to the end. Weights c give the columns X h_2, ..., X h_p and then
    X c / (c'c), where h_j are the columns of the Householder reflection H of c in R's
    sign convention, so that the last coefficient is c'alpha (limma's
    ``contrastAsCoef``, found by probing).
    """
    if isinstance(contrast, int):
        order = [j for j in range(x.shape[1]) if j != contrast] + [contrast]
        return x[:, order]
    reflection = _householder_reflection(contrast)
    last = x @ contrast / float(contrast @ contrast)
    return np.column_stack([x @ reflection[:, 1:], last])


def _householder_reflection(c: FloatArray) -> FloatArray:
    """H = I - u u' / u_0 with u = c / norm + e_0, norm = |c| with the sign of c_0.

    H c = -norm e_0, so the reflection maps c to the first axis as R's ``qr()``
    (LINPACK) does, with a positive norm when c_0 is zero.
    """
    norm = float(np.linalg.norm(c))
    if c[0] != 0.0:
        norm = math.copysign(norm, c[0])
    u = c / norm
    u[0] += 1.0
    reflection: FloatArray = np.eye(c.size) - np.outer(u, u) / u[0]
    return reflection


def expression_values(y: Any, weights: Any) -> tuple[FloatArray, FloatArray | None]:
    """The values (samples x genes) and the precision weights, checked."""
    if isinstance(y, VoomResult):
        if weights is not None:
            raise ValueError(
                "weights cannot be given with a VoomResult, which carries the "
                "precision weights of its log-CPM values"
            )
        return np.asarray(y.log_cpm, dtype=np.float64), np.asarray(
            y.weights, dtype=np.float64
        )
    values = real_array(unmasked_array(y, "y"), "y", boolean=False, objects=False)
    if values.ndim != 2:
        raise ValueError(
            "y must be a 2-D array of shape (n_samples, n_genes), or a VoomResult, got "
            f"shape {values.shape}"
        )
    if values.shape[1] == 0:
        raise ValueError("y must have at least one gene (column)")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "y must be finite: gene-set tests fit every gene to every sample, so "
            "values cannot be missing (NaN) or infinite"
        )
    if weights is None:
        return values, None
    w = real_array(
        unmasked_array(weights, "weights"), "weights", boolean=False, objects=False
    )
    n_samples, n_genes = values.shape
    if w.shape not in ((n_samples,), (n_samples, n_genes)):
        raise ValueError(
            f"weights must have shape ({n_samples},) for sample weights or "
            f"{values.shape} for observation weights, got {w.shape}"
        )
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(w) & (w > 0.0)
    if not np.all(valid):
        raise ValueError(
            "weights must be positive and finite: every sample enters every gene's "
            "fit, so that all genes have the same residual degrees of freedom; "
            f"{int(np.count_nonzero(~valid))} are not"
        )
    return values, np.ascontiguousarray(w)


def checked_gene_sets(
    gene_sets: Any, gene_ids: Any, n_genes: int
) -> dict[Any, int] | None:
    """Check that ``gene_sets`` is a non-empty mapping; the index of ``gene_ids``."""
    reject_masked(gene_sets, "gene_sets")
    if not isinstance(gene_sets, Mapping):
        raise TypeError(
            "gene_sets must be a mapping of set name to genes, such as "
            f"{{'interferon': ['ISG15', 'IFIT1']}}; got {type(gene_sets).__name__}"
        )
    if not gene_sets:
        raise ValueError("gene_sets is empty; give at least one gene set")
    return None if gene_ids is None else _gene_index(gene_ids, n_genes)


def set_label(name: Any) -> str:
    """How messages name a set, after checking its name."""
    if not isinstance(name, str):
        raise TypeError(f"gene set names must be strings, got {name!r}")
    if not name:
        raise ValueError("gene set names must not be empty")
    return f"gene_sets[{name!r}]"


def set_positions(
    genes: Any, label: str, index: dict[Any, int] | None, n_genes: int
) -> tuple[IntArray, int]:
    """The sorted positions of a set's genes, and how many ids were found."""
    if isinstance(genes, (str, bytes)):
        raise TypeError(
            f"{label} is a single string; give the set's genes as a list, such as "
            f"[{genes!r}]"
        )
    if isinstance(genes, Set):
        genes = list(genes)
    array = unmasked_array(genes, label)
    if array.ndim != 1:
        raise ValueError(
            f"{label} must be a list of genes (a 1-D sequence), got shape {array.shape}"
        )
    found = 0
    if array.dtype == np.bool_:
        if array.size != n_genes:
            raise ValueError(
                f"{label} is a boolean mask of {array.size} values; a mask "
                f"needs one value per gene ({n_genes})"
            )
        positions = np.flatnonzero(array)
    elif index is None:
        positions = gene_positions(array, label, n_genes)
    else:
        positions = id_positions(array.tolist(), label, index)
        found = positions.size
    return np.unique(positions).astype(np.intp), found


def id_positions(ids: list[Any], label: str, index: dict[Any, int]) -> IntArray:
    """The positions of the ids that are among the gene ids; others are left out."""
    try:
        found = [index[gene] for gene in ids if gene in index]
    except TypeError:
        raise TypeError(
            f"{label} holds values that cannot be gene ids, such as {ids[0]!r}"
        ) from None
    return np.array(found, dtype=np.intp)


def check_matched(index: dict[Any, int] | None, matched: int) -> None:
    if index is not None and matched == 0:
        raise ValueError(
            "no gene of any set is among gene_ids; the sets and gene_ids must name "
            "genes the same way (for example both by symbol, or both by Ensembl id)"
        )


def _gene_index(gene_ids: Any, n_genes: int) -> dict[Any, int]:
    ids = unmasked_array(gene_ids, "gene_ids")
    if ids.shape != (n_genes,):
        raise ValueError(
            f"gene_ids must have one id per gene ({n_genes}), got shape {ids.shape}"
        )
    index: dict[Any, int] = {}
    try:
        for position, gene in enumerate(ids.tolist()):
            if gene in index:
                raise ValueError(
                    f"gene_ids must be unique, but {gene!r} names genes "
                    f"{index[gene]} and {position}"
                )
            index[gene] = position
    except TypeError:
        raise TypeError("gene ids must be hashable values, such as strings") from None
    return index


def gene_positions(array: npt.NDArray[Any], label: str, n_genes: int) -> IntArray:
    if array.size == 0:
        return np.empty(0, dtype=np.intp)
    if array.dtype.kind not in "iu":
        raise TypeError(
            f"{label} must hold 0-based gene positions (integers) when gene_ids is not "
            f"given; give gene_ids to name genes by id (got dtype {array.dtype})"
        )
    outside = (array < 0) | (array >= n_genes)
    if np.any(outside):
        raise ValueError(
            f"{label} has position {int(array[outside][0])}, but there are {n_genes} "
            f"genes, at positions 0 to {n_genes - 1}"
        )
    return array.astype(np.intp)
