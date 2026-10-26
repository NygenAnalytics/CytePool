"""The minimum of a function of one variable on an interval, as R's ``optimize``.

Brent's (1973) method: golden section search with parabolic interpolation, with R's
stopping rule, so that the sequence of evaluations, and so the result, is that of R's
``optimize``. limma's estimate of the prior degrees of freedom and edgeR's common
dispersion are found with it, and the smoothing parameter of R's ``smooth.spline``,
whose search is the same with a relative precision of its own.

References:
    Brent RP (1973). Algorithms for Minimization without Derivatives. Prentice-Hall,
    Englewood Cliffs.
"""

import math
from collections.abc import Callable

import numpy as np

_EPS = float(np.finfo(np.float64).eps)
# The golden section ratio of Brent's minimisation, (3 - sqrt(5)) / 2.
_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))


def brent_minimum(
    function: Callable[[float], float],
    low: float,
    high: float,
    tolerance: float,
    *,
    relative: float | None = None,
) -> float:
    """The minimiser of a function on [low, high] by Brent's (1973) method.

    Golden section search with parabolic interpolation, as in Brent's procedure
    ``localmin`` (chapter 5) and R's ``optimize``, with its stopping rule: the search
    ends when the best point lies within 2 (relative |x| + tolerance / 3) of both
    ends of the bracket, ``relative`` being sqrt(eps), as in R's ``optimize``, unless
    given. The sequence of evaluations, and so the result, is that of R's
    ``optimize`` (checked against it).
    """
    eps = math.sqrt(_EPS) if relative is None else relative
    a, b = low, high
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = function(x)
    d = e = 0.0
    while True:
        middle = 0.5 * (a + b)
        tol1 = eps * abs(x) + tolerance / 3.0
        tol2 = 2.0 * tol1
        if abs(x - middle) <= tol2 - 0.5 * (b - a):
            return x
        p = q = r = 0.0
        if abs(e) > tol1:
            # The parabola through (v, fv), (w, fw) and (x, fx).
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(0.5 * q * r) or p <= q * (a - x) or p >= q * (b - x):
            # A golden section step into the larger part of the bracket.
            e = (a - x) if x >= middle else (b - x)
            d = _GOLDEN * e
        else:
            d = p / q
            u = x + d
            # Not too close to the ends of the bracket.
            if u - a < tol2 or b - u < tol2:
                d = tol1 if middle - x >= 0.0 else -tol1
        # Not too close to x.
        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + (tol1 if d >= 0.0 else -tol1)
        fu = function(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
