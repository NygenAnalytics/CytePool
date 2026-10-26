"""Linear interpolation constant beyond the data, as R's ``approx``.

``approx(x, y, at)`` is R's ``approx(x, y, xout = at, rule = 2, ties = mean)``: the
values of y at tied x are averaged, and between two knots the value is
y_j + (y_j+1 - y_j) ((v - x_j) / (x_j+1 - x_j)), the operations of R's ``approx``
(found by probing; they decide the last bit of voom's weights), the value of a knot
at the knot, and that of the first or last knot beyond them (Becker, Chambers and
Wilks 1988). NaN where ``at`` is NaN. voom's weights interpolate its trend with it,
and the robust estimation of the prior interpolates the prior variances of the
features it cannot use.

References:
    Becker RA, Chambers JM, Wilks AR (1988). The New S Language. Wadsworth &
    Brooks/Cole, Pacific Grove.
"""

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def approx(x: FloatArray, y: FloatArray, at: FloatArray) -> FloatArray:
    """Linear interpolation of y over x at ``at``, constant beyond the range of x."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    knots_x = xs[starts]
    knots_y = np.add.reduceat(ys, starts) / np.diff(np.r_[starts, xs.size])
    if knots_x.size == 1:
        constant: FloatArray = np.where(np.isnan(at), np.nan, knots_y[0])
        return constant
    j = np.searchsorted(knots_x, at, side="right") - 1
    np.clip(j, 0, knots_x.size - 2, out=j)
    x0, x1 = knots_x[j], knots_x[j + 1]
    y0, y1 = knots_y[j], knots_y[j + 1]
    values: FloatArray = y0 + (y1 - y0) * ((at - x0) / (x1 - x0))
    values[at <= knots_x[0]] = knots_y[0]
    values[at >= knots_x[-1]] = knots_y[-1]
    return values
