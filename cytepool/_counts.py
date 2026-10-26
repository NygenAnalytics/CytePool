"""Count matrices, library sizes, and the offsets, dispersions and prior weights of
count models, read and checked once for every method.

Counts are samples x features, finite and non-negative. A library size is a sample's
total count, exact for whole numbers whose total is at most 2**53, and the effective
library size is the library size times the sample's normalization factor (Robinson and
Oshlack 2010). The count models take their offsets (the logs of effective library
sizes, or given), dispersions and prior weights as matrices of one row per gene, or
one row for every gene, and one column per sample or one for every sample (McCarthy,
Chen and Smyth 2012).

References:
    Robinson MD, Oshlack A (2010). A scaling normalization method for differential
    expression analysis of RNA-seq data. Genome Biology 11, R25.
    doi:10.1186/gb-2010-11-3-r25

    McCarthy DJ, Chen Y, Smyth GK (2012). Differential expression analysis of
    multifactor RNA-Seq experiments with respect to biological variation. Nucleic Acids
    Research 40, 4288-4297. doi:10.1093/nar/gks042
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ._masked_arrays import unmasked_array
from ._summation import accurate_sums

FloatArray = npt.NDArray[np.float64]


def as_counts(counts: npt.ArrayLike) -> FloatArray:
    array = unmasked_array(counts, "counts")
    if array.dtype.kind not in "iuf":
        raise TypeError(
            f"counts must be a dense integer or float array, got dtype {array.dtype}"
        )
    if array.ndim != 2:
        raise ValueError(
            "counts must be a 2-D array of samples x features, "
            f"got {array.ndim} dimension(s)"
        )
    if 0 in array.shape:
        raise ValueError(
            "counts must have at least one sample and one feature, "
            f"got shape {array.shape}"
        )
    # Writeable, aligned C order: the compiled routines then compile once.
    values = np.require(array, np.float64, ["C", "A", "W"])
    bad = np.argwhere(~(np.isfinite(values) & (values >= 0)))
    if bad.size:
        i, j = bad[0]
        raise ValueError(
            "counts must be finite and non-negative; "
            f"sample {i}, feature {j} is {values[i, j]}"
        )
    return values


def sums_in_range(counts: FloatArray, axis: int) -> FloatArray:
    """Return ``accurate_sums(counts, axis)``; a sum beyond float64 raises OverflowError."""
    sums: FloatArray = accurate_sums(counts, axis)
    beyond = np.flatnonzero(~np.isfinite(sums))
    if beyond.size:
        unit = "sample" if axis == 1 else "feature"
        raise OverflowError(
            f"the counts of {unit} {beyond[0]} sum beyond the float64 range (about "
            "1.8e308)"
        )
    return sums


def _positive_vector(values: npt.ArrayLike, name: str, n_samples: int) -> FloatArray:
    array = unmasked_array(values, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    if array.ndim != 1 or array.shape[0] != n_samples:
        raise ValueError(
            f"{name} must have one value per sample ({n_samples}), "
            f"got shape {array.shape}"
        )
    vector = np.require(array, np.float64, ["C", "A", "W"])
    bad = np.flatnonzero(~(np.isfinite(vector) & (vector > 0)))
    if bad.size:
        raise ValueError(
            f"{name} must be finite and positive; sample {bad[0]} has {vector[bad[0]]}"
        )
    return vector


def library_sizes_of(
    counts: FloatArray, library_sizes: npt.ArrayLike | None
) -> FloatArray:
    if library_sizes is not None:
        return _positive_vector(library_sizes, "library_sizes", counts.shape[0])
    sizes = sums_in_range(counts, 1)
    empty = np.flatnonzero(sizes == 0)
    if empty.size:
        raise ValueError(
            f"sample(s) {empty.tolist()} have no counts, so their library size is "
            "zero; remove them"
        )
    return sizes


def effective_library_sizes(
    counts: FloatArray,
    library_sizes: npt.ArrayLike | None,
    norm_factors: npt.ArrayLike | None,
) -> FloatArray:
    sizes = library_sizes_of(counts, library_sizes)
    if norm_factors is None:
        return sizes
    factors = _positive_vector(norm_factors, "norm_factors", counts.shape[0])
    with np.errstate(over="ignore", under="ignore"):
        sizes = sizes * factors
    if not np.all(np.isfinite(sizes) & (sizes > 0)):
        raise ValueError("library_sizes * norm_factors must be finite and positive")
    return sizes


def real_values(values: Any, name: str) -> FloatArray:
    """An argument as a new float64 array; booleans and non-numbers are refused.

    A copy, so that a fit does not change with the caller's array.
    """
    array = unmasked_array(values, name)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    return np.array(array, dtype=np.float64)


def dispersion_matrix(dispersion: Any, n_samples: int, n_features: int) -> FloatArray:
    """Dispersions as a C-ordered float64 matrix, genes x samples, of shape
    (1 or n_features, 1 or n_samples)."""
    phi = real_values(dispersion, "dispersion")
    if phi.shape == ():
        matrix = phi.reshape(1, 1)
    elif phi.shape == (n_features,):
        matrix = phi.reshape(n_features, 1)
    elif phi.shape == (n_samples, n_features):
        matrix = phi.T
    else:
        raise ValueError(
            "dispersion must be one value, one per feature "
            f"({n_features},) or one per observation ({n_samples}, {n_features}), "
            f"got shape {phi.shape}"
        )
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(matrix) & (matrix >= 0.0))
    if bad.any():
        raise ValueError(
            "dispersion must be finite and non-negative; "
            f"{int(np.count_nonzero(bad))} values are not"
        )
    return np.ascontiguousarray(matrix, dtype=np.float64)


def offset_matrix(
    offsets: Any,
    library_sizes: npt.ArrayLike | None,
    norm_factors: npt.ArrayLike | None,
    counts: FloatArray,
) -> FloatArray:
    """Offsets as a C-ordered float64 matrix, genes x samples, of shape
    (1 or n_features, n_samples)."""
    n_samples, n_features = counts.shape
    if offsets is None:
        sizes = effective_library_sizes(counts, library_sizes, norm_factors)
        return np.log(sizes).reshape(1, n_samples)
    if library_sizes is not None or norm_factors is not None:
        raise ValueError(
            "give offsets, or library sizes and normalization factors, not both: the "
            "offsets are the log library sizes times normalization factors"
        )
    values = real_values(offsets, "offsets")
    if values.shape == ():
        matrix = np.full((1, n_samples), float(values))
    elif values.shape == (n_samples,):
        matrix = values.reshape(1, n_samples)
    elif values.shape == (n_samples, n_features):
        matrix = values.T
    else:
        raise ValueError(
            f"offsets must be one value, one per sample ({n_samples},) or one per "
            f"observation ({n_samples}, {n_features}), got shape {values.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("offsets must be finite")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def weight_matrix(weights: Any, n_samples: int, n_features: int) -> FloatArray:
    """Prior weights as a C-ordered float64 matrix, genes x samples, of shape
    (1 or n_features, 1 or n_samples)."""
    if weights is None:
        return np.ones((1, 1))
    values = real_values(weights, "weights")
    if values.shape == ():
        matrix = values.reshape(1, 1)
    elif values.shape == (n_samples,):
        matrix = values.reshape(1, n_samples)
    elif values.shape == (n_samples, n_features):
        matrix = values.T
    else:
        raise ValueError(
            f"weights must be one value, one per sample ({n_samples},) or one per "
            f"observation ({n_samples}, {n_features}), got shape {values.shape}"
        )
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(matrix) & (matrix > 0.0))
    if bad.any():
        raise ValueError(
            "weights must be finite and positive, as edgeR requires; "
            f"{int(np.count_nonzero(bad))} values are not"
        )
    return np.ascontiguousarray(matrix, dtype=np.float64)
