"""Linear models fitted to many features at once.

Every feature (gene) of a samples x features matrix is fitted with the same
design by ordinary, weighted or generalised least squares, and the fitted
coefficients can be re-expressed as contrasts. The results match limma's
``lmFit`` and ``contrasts.fit``. For feature g with observed responses y_g,

    E(y_g) = X a_g,    var(y_g) = sigma_g^2 W_g^(-1/2) R W_g^(-1/2),

where W_g is the diagonal matrix of the feature's precision weights and R is
the identity, or the correlation matrix of a known correlation between samples
of the same block (Smyth 2004; Smyth, Michaud and Scott 2005).

References:
    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical
    Applications in Genetics and Molecular Biology 3, article 3.

    Smyth GK, Michaud J, Scott HS (2005). Use of within-array replicate spots
    for assessing differential expression in microarray experiments.
    Bioinformatics 21, 2067-2075.
"""

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._linear_algebra import RANK_TOLERANCE, column_space
from cytepool.design.contrasts import (
    _is_named_contrasts,
    _is_pandas,
    _weights_on_columns,
)

__all__ = ["LinearModelFit", "contrasts_fit", "fit_linear_model"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False, repr=False)
class LinearModelFit:
    """Least squares fits of one design to many features.

    Rows of the per-feature arrays are features, in the order of the columns
    of ``y``; columns are coefficients, or contrasts after
    :func:`contrasts_fit`. The arrays are read-only views; the constructor
    does not copy them.

    Attributes:
        coefficients: ``(n_features, n_coefficients)`` estimates. NaN marks a
            coefficient that the observed values of a feature cannot estimate.
        stdev_unscaled: ``(n_features, n_coefficients)`` unscaled standard
            errors: the standard errors are ``stdev_unscaled * sigma[:, None]``.
        sigma: ``(n_features,)`` residual standard deviations, NaN when
            ``df_residual`` is zero.
        df_residual: ``(n_features,)`` residual degrees of freedom: the number
            of samples in a feature's fit minus the rank of the design over
            them. Stored as floats.
        cov_coefficients: ``(n_coefficients, n_coefficients)`` unscaled
            covariance matrix of the coefficients. It is the covariance matrix
            of every feature when ``cov_coefficients_per_feature`` is None and
            ``exact_standard_errors`` is True. Otherwise it is the covariance
            matrix of the design without weights, ``(X' R^-1 X)^-1`` with R
            the block correlation (or the identity), transformed by any
            contrasts: what limma reports, and where
            ``contrasts_fit(method="limma")`` takes its correlations from.
        mean_expression: ``(n_features,)`` unweighted mean of the observed
            values of each feature (limma's ``Amean``), including values whose
            weight is zero.
        coefficient_names: Names of the coefficients or contrasts.
        design: ``(n_samples, n_design_coefficients)`` design matrix.
        cov_coefficients_per_feature: ``(n_features, n_coefficients,
            n_coefficients)`` unscaled covariance matrix of each feature, or
            None. Rows and columns of coefficients that a feature cannot
            estimate are NaN.
        contrasts: ``(n_design_coefficients, n_coefficients)`` matrix mapping
            the design's coefficients to ``coefficients``, or None when the fit
            holds the design's own coefficients.
        exact_standard_errors: False when ``stdev_unscaled`` are limma's
            approximation, from ``contrasts_fit(method="limma")`` of a fit
            with per-feature covariance matrices; such a fit has none.

    Raises:
        TypeError: If ``coefficient_names`` is a string or
            ``exact_standard_errors`` is not a bool.
        ValueError: If shapes disagree, names are not unique, or a fit marked
            exact has neither per-feature covariance matrices nor standard
            errors that ``cov_coefficients`` implies.
    """

    coefficients: FloatArray
    stdev_unscaled: FloatArray
    sigma: FloatArray
    df_residual: FloatArray
    cov_coefficients: FloatArray
    mean_expression: FloatArray
    coefficient_names: tuple[str, ...]
    design: FloatArray
    cov_coefficients_per_feature: FloatArray | None = None
    contrasts: FloatArray | None = None
    exact_standard_errors: bool = True

    def __post_init__(self) -> None:
        coefficients = _read_only(self.coefficients, "coefficients", ndim=2)
        n_features, n_coefficients = coefficients.shape
        design = _read_only(self.design, "design", ndim=2)
        if isinstance(self.coefficient_names, str):
            raise TypeError("coefficient_names must be a sequence of strings")
        names = tuple(self.coefficient_names)
        if len(names) != n_coefficients or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"coefficient_names must be {n_coefficients} strings, one per "
                "column of coefficients"
            )
        names = tuple(str(name) for name in names)
        if len(set(names)) != len(names):
            raise ValueError(f"coefficient_names must be unique, got {names}")
        if self.contrasts is None:
            if design.shape[1] != n_coefficients:
                raise ValueError(
                    f"design has {design.shape[1]} columns but there are "
                    f"{n_coefficients} coefficients; a fit of contrasts needs "
                    "the contrasts matrix"
                )
            contrasts = None
        else:
            contrasts = _read_only(
                self.contrasts, "contrasts", shape=(design.shape[1], n_coefficients)
            )
        square = (n_coefficients, n_coefficients)
        stdev_unscaled = _read_only(
            self.stdev_unscaled, "stdev_unscaled", shape=coefficients.shape
        )
        cov_coefficients = _read_only(
            self.cov_coefficients, "cov_coefficients", shape=square
        )
        if not isinstance(self.exact_standard_errors, (bool, np.bool_)):
            raise TypeError("exact_standard_errors must be a bool")
        exact = bool(self.exact_standard_errors)
        per_feature = self.cov_coefficients_per_feature
        if per_feature is not None:
            if not exact:
                raise ValueError(
                    "a fit with approximate standard errors has no per-feature "
                    "covariance matrices"
                )
            per_feature = _read_only(
                per_feature,
                "cov_coefficients_per_feature",
                shape=(n_features, *square),
            )
        elif exact and not np.allclose(
            stdev_unscaled, np.sqrt(np.diag(cov_coefficients)), rtol=1e-6, atol=0.0
        ):
            raise ValueError(
                "stdev_unscaled differ from the standard errors cov_coefficients "
                "implies; give cov_coefficients_per_feature, or set "
                "exact_standard_errors=False for approximate standard errors"
            )
        fields = {
            "coefficients": coefficients,
            "stdev_unscaled": stdev_unscaled,
            "sigma": _read_only(self.sigma, "sigma", shape=(n_features,)),
            "df_residual": _read_only(
                self.df_residual, "df_residual", shape=(n_features,)
            ),
            "cov_coefficients": cov_coefficients,
            "mean_expression": _read_only(
                self.mean_expression, "mean_expression", shape=(n_features,)
            ),
            "coefficient_names": names,
            "design": design,
            "cov_coefficients_per_feature": per_feature,
            "contrasts": contrasts,
            "exact_standard_errors": exact,
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            f"LinearModelFit(n_features={self.coefficients.shape[0]}, "
            f"n_samples={self.design.shape[0]}, "
            f"coefficient_names={self.coefficient_names!r})"
        )


def fit_linear_model(
    y: npt.ArrayLike,
    design: Any,
    *,
    weights: npt.ArrayLike | None = None,
    block: npt.ArrayLike | None = None,
    correlation: float | None = None,
    coefficient_names: Sequence[str] | None = None,
) -> LinearModelFit:
    """Fit the same linear model to every feature by least squares.

    Each column of ``y`` is regressed on ``design``: by ordinary least
    squares, by weighted least squares with precision ``weights``, and by
    generalised least squares when ``block`` and ``correlation`` set a known
    correlation between samples of the same block. Results match limma's
    ``lmFit``.

    NaN in ``y`` marks a missing value of one feature, and a zero weight
    leaves an observed value out of that feature's fit (it still counts in
    ``mean_expression``). Each feature is fitted to its remaining samples; its
    ``df_residual`` is their number minus the rank of the design over them,
    and coefficients they cannot estimate are NaN. A coefficient is estimable
    when its design column, over those samples, is not a linear combination
    of the other columns. limma instead drops the later of collinear columns
    and reports the rest, so it can return a value for a coefficient that the
    data do not determine, such as the intercept when every sample of the
    reference group is missing; such values are NaN here, and all estimable
    values agree with limma.

    Args:
        y: ``(n_samples, n_features)`` log-scale expression values. NaN marks
            a missing value; infinite values are rejected.
        design: ``(n_samples, n_coefficients)`` design matrix of full column
            rank, an object with ``matrix`` and ``columns`` attributes, or a
            table with ``columns``; column names name the coefficients.
        weights: Precision weights, ``(n_samples,)`` for sample weights or
            ``(n_samples, n_features)`` for observation weights (as from
            voom). They must be finite and non-negative wherever ``y`` is
            observed.
        block: One label per sample; samples with the same label are
            correlated. Give together with ``correlation``.
        correlation: Correlation between samples of the same block, in
            ``(-1, 1)``. With blocks of up to m samples it must exceed
            ``-1 / (m - 1)``, so that the correlation matrix is positive
            definite.
        coefficient_names: Names of the design columns, in order. Defaults to
            the design's column names when it has them, else ``x1, x2, ...``.
            They may rename the design's columns, but not give a column the name
            of another: contrasts are matched to coefficients by name.

    Returns:
        The fit. Without missing values, and with no weights or positive
        sample weights, one factorisation serves every feature. Otherwise each
        feature is fitted on its own, and the fit stores the covariance matrix
        of every feature: ``n_features * n_coefficients**2`` floats.

    Raises:
        TypeError: If an input is not real-valued or a label is not comparable.
        ValueError: If shapes disagree, ``y`` has infinite values, ``design``
            is not finite or not of full column rank (the message names the
            aliased columns), a weight is negative or not finite where ``y`` is
            observed, ``block`` and ``correlation`` are invalid, or
            ``coefficient_names`` move a design column's name to another column.
    """
    x, names = _design_matrix(design, coefficient_names)
    values = _expression_matrix(y, x.shape[0])
    observed = ~np.isnan(values)
    precision = _precision_weights(weights, values.shape, observed)
    codes, counts, rho = _block_structure(block, correlation, x.shape[0])

    aliased = _aliased_columns(x)
    if aliased:
        listed = ", ".join(repr(names[j]) for j in aliased)
        raise ValueError(
            f"design is not of full column rank: column(s) {listed} are linear "
            "combinations of the preceding columns; remove them or "
            "reparametrise the design"
        )
    if precision is not None and precision.ndim == 1:
        # Sample weights apply to every feature, and a zero weight removes the
        # sample from every feature's fit.
        positive = precision > 0.0
        weighted = x[positive] * np.sqrt(precision[positive])[:, np.newaxis]
        aliased = _aliased_columns(weighted)
        if aliased:
            listed = ", ".join(repr(names[j]) for j in aliased)
            raise ValueError(
                "design is not of full column rank once the sample weights are "
                f"applied (samples of zero weight are left out): column(s) {listed} "
                "are linear combinations of the preceding columns"
            )

    complete = bool(observed.all())
    mean_expression = _observed_means(values, observed, complete)
    # Only in this case does limma report the covariance matrix of the
    # weighted design; otherwise it reports that of the design without
    # weights (found by probing).
    if complete and (
        precision is None or (precision.ndim == 1 and bool(np.all(precision > 0.0)))
    ):
        return _fit_shared_design(
            values, x, names, precision, codes, counts, rho, mean_expression
        )
    return _fit_each_feature(
        values, x, names, precision, codes, counts, rho, mean_expression
    )


def contrasts_fit(
    fit: LinearModelFit,
    contrasts: Any,
    *,
    names: Sequence[str] | None = None,
    method: Literal["exact", "limma"] = "exact",
) -> LinearModelFit:
    """Re-express a fit in terms of contrasts of its coefficients.

    For a contrast matrix C whose column c_j defines contrast j, the new
    coefficients of feature g are C' a_g. With ``method="exact"`` their
    unscaled variances are c_j' V_g c_j, with V_g the unscaled covariance
    matrix of the feature. ``method="limma"`` reproduces limma's
    ``contrasts.fit``, which does not use V_g: it combines the feature's
    unscaled standard errors u_g with the correlation matrix P of
    ``cov_coefficients``, as sum_kl C_kj C_lj u_gk u_gl P_kl.

    The two methods agree when the fit factorised one design for every
    feature (no missing values; no weights, or positive sample weights), and
    for contrasts of uncorrelated coefficients, such as group means of a
    one-way design without blocks. Otherwise limma's standard errors are
    approximate, because P then comes from the whole design without weights
    rather than from each feature's own samples and weights. On a design whose
    coefficients are correlated, such as group comparisons adjusted for a
    covariate, they can be off by a factor of two or more; and with sample
    weights, a single missing value anywhere, or a single zero weight, makes
    them approximate for every feature.

    A contrast with a non-zero multiplier on a coefficient that is NaN for a
    feature is NaN for that feature, as in limma. This includes contrasts that
    the feature's data do determine although the coefficients they combine
    are not estimable, such as group C minus group B when the reference group
    is missing; fit the group means without an intercept to obtain those.

    Contrasts that name the coefficients they weigh are matched to
    ``fit.coefficient_names`` by name, whatever their order: a
    :class:`~cytepool.design.Contrasts` (from ``make_contrasts`` or
    ``Design.contrasts``) by its ``columns``, and a pandas DataFrame or Series
    by its index. They must name every coefficient of the fit exactly once.
    Weights without such names (numpy arrays, lists, the vectors that indexing
    a ``Contrasts`` returns, and pandas objects with the default index 0, 1,
    ..., n - 1) are taken in the order of ``fit.coefficient_names``; a pandas
    index of other labels that are not strings is an error. limma's
    ``contrasts.fit`` always takes the rows in order, and only warns when their
    names differ from the coefficient names (found by probing).

    Args:
        fit: A fit from :func:`fit_linear_model` or from this function.
        contrasts: The contrasts, one per column of a ``(n_coefficients,
            n_contrasts)`` matrix: a ``Contrasts``, or any object with
            ``matrix``, ``names`` and ``columns`` attributes laid out the same
            way; a pandas DataFrame with one contrast per column; a mapping of
            contrast name to vector or pandas Series; a matrix; or a single
            vector or Series.
        names: Contrast names, when ``contrasts`` is not a mapping or a
            ``Contrasts``. Defaults to the column names of a DataFrame or the
            name of a Series, else ``contrast1, contrast2, ...``. Names rename
            the DataFrame's columns in order, so none may be the name of another
            column.
        method: ``"exact"`` or ``"limma"``.

    Returns:
        A fit whose coefficients are the contrasts. ``sigma``, ``df_residual``,
        ``mean_expression`` and ``design`` are unchanged, and ``contrasts``
        maps the design's coefficients to the new ones. The exact method
        carries per-feature covariance matrices over. The limma method drops
        them, as they would contradict its standard errors, which it marks as
        approximate unless one covariance matrix served every feature.

    Raises:
        TypeError: If ``fit`` is not a LinearModelFit, ``contrasts`` is not
            real-valued, or it is a design rather than contrasts.
        ValueError: If ``contrasts`` is not finite or has the wrong number of
            rows, names coefficients that are not exactly the fit's, the names
            are invalid, ``method`` is unknown, or the exact method is asked of
            a fit whose standard errors are approximate.
    """
    if not isinstance(fit, LinearModelFit):
        raise TypeError(f"fit must be a LinearModelFit, got {type(fit).__name__}")
    if method not in ("exact", "limma"):
        raise ValueError(f"method must be 'exact' or 'limma', got {method!r}")
    if method == "exact" and not fit.exact_standard_errors:
        raise ValueError(
            "the standard errors of this fit are limma's approximation (from "
            "contrasts_fit with method='limma'), so its contrasts cannot be exact; "
            "contrast the original fit, or use method='limma'"
        )
    matrix, contrast_names = _contrast_matrix(contrasts, names, fit.coefficient_names)
    n_features = fit.coefficients.shape[0]
    n_contrasts = matrix.shape[1]

    unusable = np.isnan(fit.coefficients) | np.isnan(fit.stdev_unscaled)
    involved = (matrix != 0.0).astype(np.float64)
    undefined = unusable.astype(np.float64) @ involved > 0.0
    coefficients = np.where(unusable, 0.0, fit.coefficients) @ matrix
    cov_coefficients = _symmetric(matrix.T @ fit.cov_coefficients @ matrix)

    stored = fit.cov_coefficients_per_feature
    per_feature: FloatArray | None = None
    exact = True
    if stored is None and fit.exact_standard_errors:
        # One covariance matrix for all features: both methods are exact.
        variances = np.tile(np.diag(cov_coefficients), (n_features, 1))
    elif stored is not None and method == "exact":
        covariances = np.where(np.isnan(stored), 0.0, stored)
        transformed = np.matmul(np.matmul(matrix.T, covariances), matrix)
        per_feature = (transformed + transformed.transpose(0, 2, 1)) / 2.0
        variances = np.diagonal(per_feature, axis1=1, axis2=2).copy()
        rows, columns = np.nonzero(undefined)
        per_feature[rows, columns, :] = np.nan
        per_feature[rows, :, columns] = np.nan
    else:
        # limma's contrasts.fit, as found by probing: its help page says only
        # that it does not refactorise the design for each feature.
        exact = False
        stdev = np.where(unusable, 0.0, fit.stdev_unscaled)
        correlations = _correlation_matrix(fit.cov_coefficients)
        variances = np.empty((n_features, n_contrasts))
        for j in range(n_contrasts):
            scaled = stdev * matrix[:, j]
            variances[:, j] = np.einsum("gk,kl,gl->g", scaled, correlations, scaled)

    stdev_unscaled = np.sqrt(np.maximum(variances, 0.0))
    coefficients[undefined] = np.nan
    stdev_unscaled[undefined] = np.nan
    total = matrix if fit.contrasts is None else fit.contrasts @ matrix
    return LinearModelFit(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=fit.sigma,
        df_residual=fit.df_residual,
        cov_coefficients=cov_coefficients,
        mean_expression=fit.mean_expression,
        coefficient_names=contrast_names,
        design=fit.design,
        cov_coefficients_per_feature=per_feature,
        contrasts=total,
        exact_standard_errors=exact,
    )


def _fit_shared_design(
    values: FloatArray,
    x: FloatArray,
    names: tuple[str, ...],
    sample_weights: FloatArray | None,
    codes: IntArray | None,
    counts: IntArray | None,
    correlation: float,
    mean_expression: FloatArray,
) -> LinearModelFit:
    """Fit every feature with one QR factorisation of the whitened design."""
    n_samples, n_coefficients = x.shape
    n_features = values.shape[1]
    whitened_design = _whiten(x, sample_weights, codes, counts, correlation)
    aliased = _aliased_columns(whitened_design)
    if aliased:
        listed = ", ".join(repr(names[j]) for j in aliased)
        raise ValueError(
            "design is numerically rank deficient once the weights and the block "
            f"correlation are applied: column(s) {listed} are linear combinations "
            "of the preceding columns"
        )
    whitened = _whiten(values, sample_weights, codes, counts, correlation)
    q, r = np.linalg.qr(whitened_design)
    effects = q.T @ whitened
    coefficients = np.linalg.solve(r, effects).T
    residuals = whitened - q @ effects
    df = n_samples - n_coefficients
    if df > 0:
        sigma = np.sqrt(np.einsum("ij,ij->j", residuals, residuals) / df)
    else:
        sigma = np.full(n_features, np.nan)
    cov_coefficients = _inverse_cross_product(r)
    return LinearModelFit(
        coefficients=np.ascontiguousarray(coefficients),
        stdev_unscaled=np.tile(np.sqrt(np.diag(cov_coefficients)), (n_features, 1)),
        sigma=sigma,
        df_residual=np.full(n_features, float(df)),
        cov_coefficients=cov_coefficients,
        mean_expression=mean_expression,
        coefficient_names=names,
        design=x,
    )


def _fit_each_feature(
    values: FloatArray,
    x: FloatArray,
    names: tuple[str, ...],
    weights: FloatArray | None,
    codes: IntArray | None,
    counts: IntArray | None,
    correlation: float,
    mean_expression: FloatArray,
) -> LinearModelFit:
    """Fit each feature to its own observed samples and weights."""
    n_samples = x.shape[0]
    if weights is None:
        weight_rows = np.ones((1, n_samples))
    elif weights.ndim == 1:
        weight_rows = weights[np.newaxis, :]
    else:
        weight_rows = weights.T
    if codes is None or counts is None:
        block_codes = np.zeros(n_samples, dtype=np.int64)
        n_blocks = 1
    else:
        block_codes, n_blocks = codes, counts.shape[0]
    # Writable C-ordered arrays give the compiled routine a single signature.
    coefficients, stdev_unscaled, sigma, df_residual, per_feature = (
        _least_squares_per_feature(
            _writable_c_array(values.T, np.float64),
            _writable_c_array(weight_rows, np.float64),
            _writable_c_array(x, np.float64),
            _writable_c_array(block_codes, np.int64),
            n_blocks,
            correlation,
            RANK_TOLERANCE,
        )
    )
    # limma reports the covariance matrix of the design without weights here
    # (found by probing).
    _, r = np.linalg.qr(_whiten(x, None, codes, counts, correlation))
    return LinearModelFit(
        coefficients=coefficients,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=df_residual,
        cov_coefficients=_inverse_cross_product(r),
        mean_expression=mean_expression,
        coefficient_names=names,
        design=x,
        cov_coefficients_per_feature=per_feature,
    )


@jit
def _least_squares_per_feature(
    responses: FloatArray,
    weight_rows: FloatArray,
    design: FloatArray,
    block_codes: IntArray,
    n_blocks: int,
    correlation: float,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Weighted or generalised least squares fit of each feature.

    Args:
        responses: ``(n_features, n_samples)`` values, NaN where missing.
        weight_rows: ``(n_features, n_samples)`` precision weights, or one row
            shared by all features. Read only where the response is observed;
            a zero weight leaves the observation out.
        design: ``(n_samples, n_coefficients)`` design matrix.
        block_codes: ``(n_samples,)`` block index of each sample.
        n_blocks: Number of blocks.
        correlation: Correlation within blocks; zero for none.
        tolerance: Relative norm at or below which a column is aliased.

    Returns:
        Coefficients, unscaled standard errors, residual standard deviations,
        residual degrees of freedom and unscaled covariance matrices, NaN
        where a coefficient cannot be estimated.
    """
    n_features, n_samples = responses.shape
    n_coefficients = design.shape[1]
    coefficients = np.full((n_features, n_coefficients), np.nan)
    stdev_unscaled = np.full((n_features, n_coefficients), np.nan)
    sigma = np.full(n_features, np.nan)
    df_residual = np.zeros(n_features)
    covariances = np.full((n_features, n_coefficients, n_coefficients), np.nan)

    # Row t of work is the t-th sample in a feature's fit, whitened: its
    # design row followed by its response.
    work = np.empty((n_samples, n_coefficients + 1))
    rows = np.empty(n_samples, dtype=np.int64)
    norms = np.empty(n_coefficients)
    products = np.empty(n_coefficients + 1)
    aliased = np.empty(n_coefficients, dtype=np.bool_)
    estimable = np.empty(n_coefficients, dtype=np.bool_)
    kept = np.empty(n_coefficients, dtype=np.int64)
    solution = np.empty(n_coefficients)
    inverse = np.zeros((n_coefficients, n_coefficients))
    block_sums = np.empty((n_blocks, n_coefficients + 1))
    block_counts = np.empty(n_blocks, dtype=np.int64)
    within = 1.0 / math.sqrt(1.0 - correlation)
    shared_weights = weight_rows.shape[0] == 1
    last = n_coefficients

    for g in range(n_features):
        w = 0 if shared_weights else g
        m = 0
        for i in range(n_samples):
            value = responses[g, i]
            if math.isnan(value) or weight_rows[w, i] == 0.0:
                continue
            scale = math.sqrt(weight_rows[w, i])
            rows[m] = i
            for j in range(n_coefficients):
                work[m, j] = scale * design[i, j]
            work[m, last] = scale * value
            m += 1
        if correlation != 0.0 and m > 0:
            _whiten_blocks(
                work,
                m,
                rows,
                block_codes,
                correlation,
                within,
                block_sums,
                block_counts,
            )
        rank = _householder_qr(work, m, tolerance, norms, aliased, kept, products)

        df = m - rank
        df_residual[g] = df
        if df > 0:
            total = 0.0
            for i in range(rank, m):
                total += work[i, last] * work[i, last]
            sigma[g] = math.sqrt(total / df)
        if rank == 0:
            continue

        # The triangular factor of the kept columns is R[s, t] = work[s, kept[t]].
        for t in range(rank - 1, -1, -1):
            total = work[t, last]
            for u in range(t + 1, rank):
                total -= work[t, kept[u]] * solution[u]
            solution[t] = total / work[t, kept[t]]
        for c in range(rank):
            inverse[c, c] = 1.0 / work[c, kept[c]]
            for t in range(c - 1, -1, -1):
                total = 0.0
                for u in range(t + 1, c + 1):
                    total += work[t, kept[u]] * inverse[u, c]
                inverse[t, c] = -total / work[t, kept[t]]

        # An aliased column is the kept columns times inverse @ R[:, column],
        # which includes any share of a later kept column; every kept column
        # with a non-negligible share in it is not estimable.
        for j in range(n_coefficients):
            estimable[j] = not aliased[j]
        for j in range(n_coefficients):
            if not aliased[j]:
                continue
            for t in range(rank):
                total = 0.0
                for u in range(t, rank):
                    total += inverse[t, u] * work[u, j]
                if abs(total) * norms[kept[t]] > tolerance * norms[j]:
                    estimable[kept[t]] = False

        for t in range(rank):
            jt = kept[t]
            if not estimable[jt]:
                continue
            coefficients[g, jt] = solution[t]
            for s in range(rank):
                js = kept[s]
                if not estimable[js]:
                    continue
                total = 0.0
                for u in range(max(s, t), rank):
                    total += inverse[s, u] * inverse[t, u]
                covariances[g, js, jt] = total
            stdev_unscaled[g, jt] = math.sqrt(covariances[g, jt, jt])
    return coefficients, stdev_unscaled, sigma, df_residual, covariances


@jit
def _whiten_blocks(
    work: FloatArray,
    m: int,
    rows: IntArray,
    block_codes: IntArray,
    correlation: float,
    within: float,
    block_sums: FloatArray,
    block_counts: IntArray,
) -> None:
    """Premultiply the first m rows of work by the inverse square root of R.

    Over the m_k samples of block k in a feature's fit, R is (1 - r) I + r 11',
    whose inverse square root maps z to within * (z - mean(z)) + between *
    mean(z), with within = (1 - r)^(-1/2) and between = (1 + (m_k - 1) r)^(-1/2).
    """
    n_columns = work.shape[1]
    block_sums[:, :] = 0.0
    block_counts[:] = 0
    for t in range(m):
        k = block_codes[rows[t]]
        block_counts[k] += 1
        for j in range(n_columns):
            block_sums[k, j] += work[t, j]
    for t in range(m):
        k = block_codes[rows[t]]
        count = block_counts[k]
        between = 1.0 / math.sqrt(1.0 + (count - 1) * correlation)
        shift = (between - within) / count
        for j in range(n_columns):
            work[t, j] = within * work[t, j] + shift * block_sums[k, j]


@jit
def _householder_qr(
    work: FloatArray,
    m: int,
    tolerance: float,
    norms: FloatArray,
    aliased: npt.NDArray[np.bool_],
    kept: IntArray,
    products: FloatArray,
) -> int:
    """Householder QR of the first m rows of the design columns of work.

    The last column of work is the response. Columns are taken in order; one
    whose part orthogonal to the kept columns before it has a norm at most
    ``tolerance`` times its own norm is aliased and skipped, as by R's qr().
    Each reflection is applied in place to every column, the response
    included. On return, rows ``0:rank`` of the kept column ``kept[t]`` hold
    column t of the triangular factor, those of an aliased column hold its
    coordinates in the orthonormal basis of the kept columns, and rows
    ``rank:m`` of the response hold its residual effects.

    Returns:
        The rank: the number of kept columns.
    """
    n_columns = work.shape[1] - 1
    norms[:] = 0.0
    for i in range(m):
        for j in range(n_columns):
            norms[j] += work[i, j] * work[i, j]
    for j in range(n_columns):
        norms[j] = math.sqrt(norms[j])
    rank = 0
    for j in range(n_columns):
        aliased[j] = True
        if rank == m:
            continue
        total = 0.0
        for i in range(rank, m):
            total += work[i, j] * work[i, j]
        norm = math.sqrt(total)
        if norm <= tolerance * norms[j]:
            continue
        aliased[j] = False
        head = work[rank, j]
        alpha = -norm if head >= 0.0 else norm
        work[rank, j] = head - alpha
        # The reflection is I - 2 v v' / (v'v) with v = work[rank:m, j], and
        # v'v = 2 norm (norm + |head|). Kept columns before j are zero from
        # row rank on, so the reflection leaves them unchanged.
        products[:] = 0.0
        for i in range(rank, m):
            v = work[i, j]
            for k in range(n_columns + 1):
                products[k] += v * work[i, k]
        scale = 1.0 / (norm * (norm + abs(head)))
        for k in range(n_columns + 1):
            products[k] *= scale
        products[j] = 0.0
        for i in range(rank, m):
            v = work[i, j]
            for k in range(n_columns + 1):
                work[i, k] -= products[k] * v
        work[rank, j] = alpha
        for i in range(rank + 1, m):
            work[i, j] = 0.0
        kept[rank] = j
        rank += 1
    return rank


def _writable_c_array(values: npt.NDArray[Any], dtype: type) -> npt.NDArray[Any]:
    return np.require(values, dtype=dtype, requirements=["C", "W", "A"])


def _whiten(
    z: FloatArray,
    sample_weights: FloatArray | None,
    codes: IntArray | None,
    counts: IntArray | None,
    correlation: float,
) -> FloatArray:
    """Premultiply the rows of z (one per sample) by R^(-1/2) W^(1/2).

    Any square root of the inverse covariance gives the same fit; this one
    treats each block in closed form, as in :func:`_whiten_blocks`.
    """
    if sample_weights is not None:
        z = z * np.sqrt(sample_weights)[:, np.newaxis]
    if codes is None or counts is None or correlation == 0.0:
        return z
    within = 1.0 / math.sqrt(1.0 - correlation)
    between = 1.0 / np.sqrt(1.0 + (counts - 1) * correlation)
    order = np.argsort(codes, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    sums = np.add.reduceat(z[order], starts, axis=0)
    shift = (between - within) / counts
    return within * z + shift[codes][:, np.newaxis] * sums[codes]


def _aliased_columns(x: FloatArray) -> list[int]:
    """Indices of the columns of x that are linear combinations of earlier ones.

    The rule is R's ``qr()`` rule, as in :func:`column_space`; the compiled
    :func:`_householder_qr` applies the same tolerance to each feature.
    """
    return list(column_space(x).aliased)


def _inverse_cross_product(r: FloatArray) -> FloatArray:
    """(R'R)^-1 from the triangular factor R of a QR factorisation."""
    r_inverse = np.linalg.solve(r, np.eye(r.shape[0]))
    return _symmetric(r_inverse @ r_inverse.T)


def _symmetric(matrix: FloatArray) -> FloatArray:
    return (matrix + matrix.T) / 2.0


def _correlation_matrix(covariance: FloatArray) -> FloatArray:
    """Correlations of a covariance matrix, zero for a constant coefficient.

    A coefficient with zero variance contributes nothing to a contrast's
    variance whatever its correlations, so they are set to zero.
    """
    scale = np.sqrt(np.diag(covariance))
    constant = scale == 0.0
    scale = np.where(constant, 1.0, scale)
    correlations: FloatArray = covariance / np.outer(scale, scale)
    correlations[constant, :] = 0.0
    correlations[:, constant] = 0.0
    return correlations


def _observed_means(
    values: FloatArray, observed: npt.NDArray[np.bool_], complete: bool
) -> FloatArray:
    if complete:
        return values.mean(axis=0)
    totals = np.where(observed, values, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore"):
        return totals / observed.sum(axis=0)


def _design_matrix(
    design: Any, coefficient_names: Sequence[str] | None
) -> tuple[FloatArray, tuple[str, ...]]:
    matrix, columns = _matrix_and_columns(design)
    x = _real_array(matrix, "design")
    if x.ndim != 2:
        raise ValueError(
            "design must be a 2-D array of shape (n_samples, n_coefficients), "
            f"got shape {x.shape}"
        )
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(
            f"design must have at least one row and one column, got shape {x.shape}"
        )
    if not np.isfinite(x).all():
        raise ValueError("design must be finite; it contains NaN or infinite values")
    if coefficient_names is not None:
        names = _checked_names(coefficient_names, x.shape[1], "coefficient_names")
        if columns is not None:
            _check_renaming(
                names, [str(c) for c in columns], "coefficient_names", "the design"
            )
    elif columns is not None:
        names = _checked_names([str(c) for c in columns], x.shape[1], "design columns")
    else:
        names = tuple(f"x{j + 1}" for j in range(x.shape[1]))
    # A copy, so that the fit does not change with the caller's array.
    return np.array(x, dtype=np.float64, order="C"), names


def _expression_matrix(y: npt.ArrayLike, n_samples: int) -> FloatArray:
    values = _real_array(y, "y", boolean=False)
    if values.ndim != 2:
        raise ValueError(
            "y must be a 2-D array of shape (n_samples, n_features), "
            f"got shape {values.shape}"
        )
    if values.shape[0] != n_samples:
        raise ValueError(
            f"y has {values.shape[0]} rows but design has {n_samples}; "
            "rows of both are samples"
        )
    if values.shape[1] == 0:
        raise ValueError("y must have at least one feature (column)")
    if np.isinf(values).any():
        raise ValueError("y contains infinite values; mark missing values with NaN")
    return values


def _precision_weights(
    weights: npt.ArrayLike | None,
    shape: tuple[int, int],
    observed: npt.NDArray[np.bool_],
) -> FloatArray | None:
    if weights is None:
        return None
    w = _real_array(weights, "weights")
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(w) & (w >= 0.0)
    if w.shape == (shape[0],):
        invalid = int(np.count_nonzero(~valid & observed.any(axis=1)))
        kind = "sample weights"
    elif w.shape == shape:
        invalid = int(np.count_nonzero(~valid & observed))
        kind = "observation weights"
    else:
        raise ValueError(
            f"weights must have shape ({shape[0]},) for sample weights or {shape} "
            f"for observation weights, got {w.shape}"
        )
    if invalid:
        raise ValueError(
            f"{kind} must be finite and non-negative wherever y is observed; "
            f"{invalid} are negative or not finite"
        )
    return np.ascontiguousarray(w)


def _block_structure(
    block: npt.ArrayLike | None, correlation: float | None, n_samples: int
) -> tuple[IntArray | None, IntArray | None, float]:
    """Block index of each sample, block sizes and the correlation."""
    if block is None and correlation is None:
        return None, None, 0.0
    if block is None or correlation is None:
        raise ValueError("block and correlation must be given together")
    if isinstance(correlation, bool) or not isinstance(
        correlation, (int, float, np.number)
    ):
        raise TypeError(
            f"correlation must be a real number, got {type(correlation).__name__}"
        )
    rho = float(correlation)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"correlation must lie in (-1, 1), got {rho}")
    labels = np.asarray(block)
    if labels.shape != (n_samples,):
        raise ValueError(
            f"block must have one label per sample ({n_samples}), "
            f"got shape {labels.shape}"
        )
    if labels.dtype.kind in "fc" and np.isnan(labels).any():
        raise ValueError("block labels must not be NaN")
    try:
        _, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
    except TypeError as error:
        raise TypeError(
            "block labels must be comparable, such as integers or strings"
        ) from error
    largest = int(counts.max())
    bound = -1.0 / (largest - 1) if largest > 1 else -1.0
    if rho <= bound:
        raise ValueError(
            f"correlation {rho} is too negative for a block of {largest} samples: "
            f"the correlation matrix is positive definite only above {bound:.6g}"
        )
    return codes.astype(np.int64).reshape(n_samples), counts.astype(np.int64), rho


def _contrast_matrix(
    contrasts: Any, names: Sequence[str] | None, coefficient_names: tuple[str, ...]
) -> tuple[FloatArray, tuple[str, ...]]:
    """The contrast matrix, rows in the order of the coefficients, and its names.

    Weights that name their coefficients are matched to them by name, by the
    rule of :func:`cytepool.design.contrasts._weights_on_columns`; the others
    are taken in order.
    """
    n_coefficients = len(coefficient_names)
    remedy = _contrasts_remedy(coefficient_names)
    named = _is_named_contrasts(contrasts)
    if (named or isinstance(contrasts, Mapping)) and names is not None:
        raise ValueError(
            "names must not be given when contrasts is a mapping or a Contrasts, "
            "which carry their own names"
        )
    if named:
        weights = _weights_on_columns(
            contrasts,
            coefficient_names,
            label="the contrasts",
            target="the fit's coefficients",
            remedy=remedy,
        )
        matrix = _real_array(weights, "contrasts")
        if matrix.shape[1] == 0:
            raise ValueError("contrasts must hold at least one contrast")
        contrast_names = _checked_names(
            contrasts.names, matrix.shape[1], "contrast names"
        )
    elif isinstance(contrasts, Mapping):
        if not contrasts:
            raise ValueError("contrasts must hold at least one contrast")
        vectors = []
        for key, vector in contrasts.items():
            weights = _weights_on_columns(
                vector,
                coefficient_names,
                label=f"contrast {key!r}",
                target="the fit's coefficients",
                remedy=remedy,
            )
            values = _real_array(weights, f"contrast {key!r}")
            if values.shape != (n_coefficients,):
                raise ValueError(
                    f"contrast {key!r} must have one value per coefficient "
                    f"({n_coefficients}), got shape {values.shape}"
                )
            vectors.append(values)
        matrix = np.column_stack(vectors)
        contrast_names = _checked_names(
            list(contrasts.keys()), matrix.shape[1], "contrast names"
        )
    else:
        # The columns of a DataFrame, or the name of a Series, name its contrasts;
        # the index names its rows.
        columns = getattr(contrasts, "columns", None)
        if _is_pandas(contrasts) and columns is None:
            name = getattr(contrasts, "name", None)
            columns = [name] if isinstance(name, str) else None
        weights = _weights_on_columns(
            contrasts,
            coefficient_names,
            label="the contrasts",
            target="the fit's coefficients",
            remedy=remedy,
        )
        matrix = _real_array(weights, "contrasts")
        if matrix.ndim == 1:
            matrix = matrix[:, np.newaxis]
        if matrix.ndim != 2 or matrix.shape[0] != n_coefficients:
            raise ValueError(
                "contrasts must have one row per coefficient of the fit "
                f"({n_coefficients}), got shape {matrix.shape}"
            )
        if matrix.shape[1] == 0:
            raise ValueError("contrasts must hold at least one contrast")
        if names is not None:
            contrast_names = _checked_names(names, matrix.shape[1], "names")
            if columns is not None:
                _check_renaming(
                    contrast_names, [str(c) for c in columns], "names", "the contrasts"
                )
        elif columns is not None:
            contrast_names = _checked_names(
                [str(c) for c in columns], matrix.shape[1], "contrast columns"
            )
        else:
            contrast_names = tuple(f"contrast{j + 1}" for j in range(matrix.shape[1]))
    if not np.isfinite(matrix).all():
        raise ValueError("contrasts must be finite")
    return np.array(matrix, dtype=np.float64, order="C"), contrast_names


def _contrasts_remedy(coefficient_names: tuple[str, ...]) -> str:
    """How to make contrasts that match a fit, for error messages."""
    remedy = "make them with make_contrasts(..., fit.coefficient_names)"
    if coefficient_names == tuple(f"x{j + 1}" for j in range(len(coefficient_names))):
        remedy += (
            "; the fit has the default names x1, x2, ..., so fit the Design itself, "
            "not its matrix, to keep the names of its columns"
        )
    return remedy


def _check_renaming(
    given: tuple[str, ...], existing: Sequence[str], label: str, owner: str
) -> None:
    """Refuse new names that give a column the name another column already has.

    Names given by position rename the columns in order. Coefficients and
    contrasts are matched by name downstream, so moving a name to another
    column would silently swap what it refers to.
    """
    existing = list(existing)
    for position, name in enumerate(given):
        own = position < len(existing) and existing[position] == name
        if name in existing and not own:
            raise ValueError(
                f"{label} give column {position + 1} the name {name!r}, the name of "
                f"column {existing.index(name) + 1} of {owner}; {label} rename "
                f"the columns in order ({', '.join(map(repr, existing))}), so a name "
                "already in use may only label its own column"
            )


def _matrix_and_columns(values: Any) -> tuple[Any, Any]:
    """Split a design-like object into its matrix and its column names."""
    if hasattr(values, "matrix") and hasattr(values, "columns"):
        return values.matrix, values.columns
    return values, getattr(values, "columns", None)


def _checked_names(names: Any, count: int, label: str) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{label} must be a sequence of strings, not a single string")
    checked = tuple(names)
    if not all(isinstance(name, str) for name in checked):
        raise TypeError(f"{label} must be strings")
    checked = tuple(str(name) for name in checked)
    if len(checked) != count:
        raise ValueError(f"{label} must have {count} entries, got {len(checked)}")
    if len(set(checked)) != count:
        raise ValueError(f"{label} must be unique, got {checked}")
    return checked


def _real_array(values: Any, name: str, *, boolean: bool = True) -> FloatArray:
    """Values as float64 without copying when possible; booleans optional.

    Object arrays, such as a table whose columns mix booleans and floats, are
    accepted when every element is a real number.
    """
    array = np.asarray(values)
    if array.dtype.kind == "O":
        flags = (bool, np.bool_)
        kinds = {type(value) for value in array.flat}
        if not all(issubclass(kind, (numbers.Real, np.bool_)) for kind in kinds) or (
            not boolean and any(issubclass(kind, flags) for kind in kinds)
        ):
            raise TypeError(f"{name} must hold real numbers")
        array = array.astype(np.float64)
    if array.dtype.kind not in ("biuf" if boolean else "iuf"):
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    return array.astype(np.float64, copy=False)


def _read_only(
    values: Any,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-D array, got shape {array.shape}")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    view = array.view()
    view.flags.writeable = False
    return view
