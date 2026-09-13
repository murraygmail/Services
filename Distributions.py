#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Distributions: random variables with samples, percentiles and confidences.

Created Wed Nov 18 12:28:13 2020. Polished 12 September 2026.

Every distribution is an object that answers four questions: what is the
density here, what is the probability of being below here, what value sits at
this probability, and give me n draws. Everything else in the module is built
on those.

WHAT CHANGED IN THIS VERSION
----------------------------
The public API is unchanged except for one deliberate correction, marked (!).

(!) Distribution.percentile(p) now takes p in [0, 1] and returns inv_cdf(p).
    It previously asserted 0 < p <= 1 and then returned inv_cdf(p / 100), so
    percentile(0.85) gave the 0.85th percentile rather than the 85th. Three
    classes already overrode it without the division (Normal, Triangular,
    Empirical) and three inherited the division (Lognormal, Uniform,
    Logistic). All six now agree with the three that were right.
    find_confidence(pct) is unchanged and still takes 0 to 100.

  - Inverse CDFs are closed form wherever one exists, instead of interpolating
    a 100-point grid. This removes both the interpolation error and the tail
    truncation at the ends of that grid.
  - samples() is vectorised and uses its own Generator. It no longer calls
    np.random.seed, which reseeded the global stream for the whole process.
    Pass random_seed for reproducibility, or rng= to share a stream.
  - Trunc_norm_PDF has an inverse CDF, and its cdf is fixed: the division by
    the truncation mass applied to only one of the two terms, so the cdf
    returned 0.9534 at the upper truncation instead of 1.
  - Weibull_PDF inherits Distribution. inverse_cdf is kept as an alias.
  - ArrayPDF inherits Distribution and no longer calls scipy.integrate.simps
    or cumtrapz, which were removed in scipy 1.14.
  - Logistic_PDF is spelled correctly. Logisitic_PDF is kept as an alias.
  - Empirical_PDF reads its cdf and inverse cdf off the order statistics
    rather than off the histogram, so a percentile of the data is the
    percentile of the data. The histogram is still there for pdf and plotting.
  - New: Quantile_PDF, for the form external tools actually export, a table of
    (p, value) pairs.
  - NumPy only. No scipy, and no Services.Integrators.

Copyright (C) (2022) Murray Cantor All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software solely for the purpose of using it. No other rights are granted,
explicitly or implicitly, including but not limited to the rights to reproduce,
modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import math
from functools import cached_property
from math import comb

import numpy as np

__all__ = [
    'Distribution', 'Normal_PDF', 'Lognormal_PDF', 'Triangular_PDF',
    'Uniform_PDF', 'Logistic_PDF', 'Logisitic_PDF', 'Weibull_PDF',
    'Trunc_norm_PDF', 'Empirical_PDF', 'ArrayPDF', 'Quantile_PDF',
    'Phi', 'PhiInv', 'lognormal', 'normal', 'triangular', 'logistic',
    'weibull', 'geometric', 'binomial', 'Bernoulli', 'bernoulli',
    'cdf', 'inverse_cdf', 'find_minmax', 'lognormal_mean_std_dev',
]

_TINY = 1e-12


# --------------------------------------------------------------------------
#  The standard normal, without scipy
# --------------------------------------------------------------------------
_ERF = np.frompyfunc(math.erf, 1, 1)


def Phi(z):
    """Standard normal CDF. Exact to machine precision, vectorised."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + np.asarray(_ERF(z / math.sqrt(2.0)), dtype=float))


# Acklam's rational approximation, refined by one Halley step, which takes it
# to full double precision.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def PhiInv(p):
    """Standard normal quantile. Vectorised, full double precision."""
    p = np.asarray(p, dtype=float)
    scalar = (p.ndim == 0)
    p = np.atleast_1d(p).astype(float)
    out = np.full(p.shape, np.nan)

    lo = (p > 0) & (p < _P_LOW)
    hi = (p > 1 - _P_LOW) & (p < 1)
    mid = (p >= _P_LOW) & (p <= 1 - _P_LOW)

    if lo.any():
        q = np.sqrt(-2 * np.log(p[lo]))
        out[lo] = ((((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) /
                   ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1))
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - p[hi]))
        out[hi] = -((((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5]) /
                    ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1))
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        out[mid] = ((((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5])*q /
                    (((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1))

    # one Halley refinement
    ok = np.isfinite(out)
    if ok.any():
        x = out[ok]
        e = Phi(x) - p[ok]
        u = e * math.sqrt(2 * math.pi) * np.exp(x * x / 2)
        out[ok] = x - u / (1 + x * u / 2)

    out[p <= 0] = -np.inf
    out[p >= 1] = np.inf
    return float(out[0]) if scalar else out


# --------------------------------------------------------------------------
#  Density functions, kept at module level with their original signatures
# --------------------------------------------------------------------------
def lognormal(mu, sigma, x):
    """Lognormal density. mu and sigma are of the underlying normal."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    if pos.any():
        z = (np.log(x[pos]) - mu) / sigma
        out[pos] = np.exp(-0.5 * z * z) / (x[pos] * sigma * math.sqrt(2 * math.pi))
    return out


def normal(mu, sigma, x):
    """Normal density."""
    x = np.asarray(x, dtype=float)
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))


def triangular(low, expected, high, x):
    """Triangular density on [low, high] with mode at expected."""
    x = np.asarray(x, dtype=float)
    a, c, b = float(low), float(expected), float(high)
    out = np.zeros_like(x)
    if b <= a:
        return out
    rising = (x >= a) & (x < c)
    out[rising] = 2 * (x[rising] - a) / ((b - a) * (c - a)) if c > a else 0.0
    falling = (x > c) & (x <= b)
    out[falling] = 2 * (b - x[falling]) / ((b - a) * (b - c)) if b > c else 0.0
    out[x == c] = 2 / (b - a)
    return out


def logistic(mu, s, x):
    """Logistic density. Written so the exponential cannot overflow."""
    x = np.asarray(x, dtype=float)
    z = -np.abs(x - mu) / s
    e = np.exp(z)
    return e / (s * np.square(1 + e))


def weibull(l, k, x):
    """Weibull density. l is the scale, k the shape."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x >= 0
    t = x[pos] / l
    out[pos] = (k / l) * t ** (k - 1) * np.exp(-(t ** k))
    return out


def geometric(p, k):
    """Probability of k failures before the first success."""
    return p * (1 - p) ** k


def binomial(b, k, n):
    """Probability of k favourable events in n Bernoulli trials with bias b."""
    b = np.asarray(b, dtype=float)
    return comb(n, k) * (b ** k) * (1 - b) ** (n - k)


def Bernoulli(b, k, n):
    """The binomial without its coefficient: one particular sequence."""
    b = np.asarray(b, dtype=float)
    return (b ** k) * (1 - b) ** (n - k)


def bernoulli(p, k, n):
    """Bernoulli over a discretised range of p, scaled by the range length."""
    p = np.asarray(p, dtype=float)
    return (p ** k) * (1 - p) ** (n - k) * len(p)


def lognormal_mean_std_dev(mu, sigma):
    """Mean and standard deviation of a lognormal from (mu, sigma)."""
    mean = math.exp(mu + 0.5 * sigma ** 2)
    return mean, mean * math.sqrt(math.exp(sigma ** 2) - 1)


# --------------------------------------------------------------------------
#  Numerical helpers, kept for callers that use them directly
# --------------------------------------------------------------------------
def cdf(P, a, x, n=1000):
    """Integrate a density P from a to x by the trapezoid rule."""
    t = np.linspace(a, x, n)
    return float(np.trapezoid(np.asarray(P(t), dtype=float), t))


def inverse_cdf(P, a, b, p, tol=1e-9):
    """Invert a density's CDF on [a, b] by bisection."""
    left, right = float(a), float(b)
    while right - left > tol:
        mid = 0.5 * (left + right)
        if cdf(P, a, mid) < p:
            left = mid
        else:
            right = mid
    return right


def find_minmax(P, a, b, edge=0.001):
    """The range of a density that holds all but 2 * edge of its mass."""
    return inverse_cdf(P, a, b, edge), inverse_cdf(P, a, b, 1 - edge)


# --------------------------------------------------------------------------
#  The base class
# --------------------------------------------------------------------------
class Distribution:
    """Base class. A subclass provides pdf(x), cdf(x) and inv_cdf(p).

    Everything below is written once, in terms of those three.
    """

    GRID = 300          # points in the plotting grids
    EDGE = 0.001        # where those grids start and stop

    # -- the three a subclass must provide ---------------------------------
    def pdf(self, x):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def inv_cdf(self, p):
        raise NotImplementedError

    # -- drawing -----------------------------------------------------------
    def samples(self, number=1, random_seed=None, rng=None):
        """n draws by inverse transform.

        Vectorised, and on its own Generator. Nothing here touches the global
        NumPy random state. Pass rng to share a stream across several
        distributions, which is what common random numbers needs.
        """
        gen = rng if rng is not None else np.random.default_rng(random_seed)
        u = gen.random(int(number))
        return np.asarray(self.inv_cdf(u), dtype=float).ravel()

    # -- readings ----------------------------------------------------------
    def percentile(self, p):
        """The value at probability p, with p in [0, 1].

        See the note at the top of this module: this used to divide p by 100
        in the base class while three subclasses did not.
        """
        p = float(p)
        if not 0.0 <= p <= 1.0:
            raise ValueError('percentile takes a probability in [0, 1]; '
                             'for 0 to 100 use find_confidence')
        return float(np.asarray(self.inv_cdf(p)).ravel()[0])

    def find_confidence(self, pct):
        """The value at percentage pct, with pct between 0 and 100."""
        p = float(pct) / 100.0
        if not 0.0 < p < 1.0:
            raise ValueError('find_confidence takes a percentage in (0, 100)')
        return float(np.asarray(self.inv_cdf(p)).ravel()[0])

    def confidence(self, x):
        """P(X > x)."""
        return 1.0 - np.asarray(self.cdf(x), dtype=float)

    def event(self, event):
        """P(a <= X <= b) for event = (a, b). None gives 0."""
        if event is None:
            return 0.0
        a, b = event
        if a > b:
            raise ValueError('event bounds must satisfy a <= b')
        return float(np.asarray(self.cdf(b)) - np.asarray(self.cdf(a)))

    # -- summaries. Closed forms override these. ---------------------------
    def _quantile_grid(self, n=20001):
        u = (np.arange(n) + 0.5) / n
        return np.asarray(self.inv_cdf(u), dtype=float)

    @cached_property
    def mean(self):
        return float(self._quantile_grid().mean())

    @cached_property
    def std(self):
        v = self._quantile_grid()
        return float(np.sqrt(((v - v.mean()) ** 2).mean()))

    @cached_property
    def median(self):
        return float(np.asarray(self.inv_cdf(0.5)).ravel()[0])

    @cached_property
    def mode(self):
        """Where the density peaks, off the plotting grid."""
        return float(self.x[int(np.argmax(self.y))])

    @property
    def min_domain(self):
        return float(np.asarray(self.inv_cdf(self.EDGE)).ravel()[0])

    @property
    def max_domain(self):
        return float(np.asarray(self.inv_cdf(1 - self.EDGE)).ravel()[0])

    # -- plotting grids, built on first use --------------------------------
    @cached_property
    def x(self):
        return np.linspace(self.min_domain, self.max_domain, self.GRID)

    @cached_property
    def y(self):
        return np.asarray(self.pdf(self.x), dtype=float)

    @cached_property
    def cy(self):
        return np.asarray(self.cdf(self.x), dtype=float)

    def __repr__(self):
        return '%s(mean=%.4g, std=%.4g)' % (type(self).__name__, self.mean, self.std)


# --------------------------------------------------------------------------
#  Parameterised distributions
# --------------------------------------------------------------------------
class Normal_PDF(Distribution):
    """Normal, given its mean and standard deviation."""

    def __init__(self, mean, std):
        if std <= 0:
            raise ValueError('std must be positive')
        self._mu, self._sd = float(mean), float(std)
        self.title_pdf = 'Normal Distribution for mean = %g, std = %g' % (mean, std)
        self.title_cdf = 'Normal Cumulative Distribution for mean = %g, std = %g' % (mean, std)

    mean = property(lambda self: self._mu)
    std = property(lambda self: self._sd)
    mode = property(lambda self: self._mu)
    median = property(lambda self: self._mu)

    def pdf(self, x):
        return normal(self._mu, self._sd, x)

    def cdf(self, x):
        return Phi((np.asarray(x, dtype=float) - self._mu) / self._sd)

    def inv_cdf(self, p):
        return self._mu + self._sd * PhiInv(p)


class Lognormal_PDF(Distribution):
    """Lognormal, given mu and sigma of the underlying normal.

    Use from_median_factor when the estimate is stated the way Appendix 7 and
    most schedule tools state it: a median, and a factor that multiplies or
    divides it at one standard deviation.
    """

    def __init__(self, mu, sigma):
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        self.mu, self.sigma = float(mu), float(sigma)
        self.title_pdf = 'Lognormal Distribution for mu = %g, sigma = %g' % (mu, sigma)
        self.title_cdf = 'Lognormal Cumulative Distribution for mu = %g, sigma = %g' % (mu, sigma)

    @classmethod
    def from_median_factor(cls, median, factor):
        if median <= 0 or factor <= 1:
            raise ValueError('median must be positive and factor greater than 1')
        return cls(math.log(median), math.log(factor))

    @cached_property
    def mean(self):
        return lognormal_mean_std_dev(self.mu, self.sigma)[0]

    @cached_property
    def std(self):
        return lognormal_mean_std_dev(self.mu, self.sigma)[1]

    median = property(lambda self: math.exp(self.mu))
    mode = property(lambda self: math.exp(self.mu - self.sigma ** 2))

    def pdf(self, x):
        return lognormal(self.mu, self.sigma, x)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        pos = x > 0
        if pos.any():
            out[pos] = Phi((np.log(x[pos]) - self.mu) / self.sigma)
        return out if out.ndim else float(out)

    def inv_cdf(self, p):
        return np.exp(self.mu + self.sigma * PhiInv(p))


class Triangular_PDF(Distribution):
    """Triangular, given a low, an expected and a high.

    The three-point estimate. Degenerate cases where the mode sits on an end
    are handled by the closed form rather than nudged.
    """

    def __init__(self, low, expected, high):
        a, c, b = float(low), float(expected), float(high)
        if not a <= c <= b:
            raise ValueError('need low <= expected <= high')
        if a == b:
            raise ValueError('low and high must differ')
        self.low, self.expected, self.high = a, c, b
        self.title_pdf = 'Triangular Distribution for (%g, %g, %g)' % (a, c, b)
        self.title_cdf = 'Triangular Cumulative Distribution for (%g, %g, %g)' % (a, c, b)

    mean = property(lambda self: (self.low + self.expected + self.high) / 3.0)
    mode = property(lambda self: self.expected)

    @cached_property
    def std(self):
        a, c, b = self.low, self.expected, self.high
        return math.sqrt((a*a + b*b + c*c - a*b - a*c - b*c) / 18.0)

    @cached_property
    def x(self):
        return np.linspace(self.low, self.high, self.GRID)

    def pdf(self, x):
        return triangular(self.low, self.expected, self.high, x)

    def cdf(self, x):
        a, c, b = self.low, self.expected, self.high
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        rising = (x >= a) & (x < c)
        if rising.any():
            out[rising] = (x[rising] - a) ** 2 / ((b - a) * (c - a))
        falling = (x >= c) & (x <= b)
        if falling.any():
            out[falling] = 1.0 - (b - x[falling]) ** 2 / ((b - a) * (b - c)) if b > c else 1.0
        out[x > b] = 1.0
        return out

    def inv_cdf(self, p):
        a, c, b = self.low, self.expected, self.high
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        split = (c - a) / (b - a)
        with np.errstate(invalid='ignore'):
            lo = a + np.sqrt(p * (b - a) * (c - a))
            hi = b - np.sqrt((1 - p) * (b - a) * (b - c))
        return np.where(p < split, lo, hi)


class Uniform_PDF(Distribution):
    """Uniform on [start, end]."""

    def __init__(self, start, end):
        if end <= start:
            raise ValueError('end must exceed start')
        self._start, self._end = float(start), float(end)
        self.title_pdf = 'Uniform Distribution for (%g, %g)' % (start, end)
        self.title_cdf = 'Uniform Cumulative Distribution for (%g, %g)' % (start, end)

    start = property(lambda self: self._start)
    end = property(lambda self: self._end)
    mean = property(lambda self: 0.5 * (self._start + self._end))
    median = property(lambda self: 0.5 * (self._start + self._end))
    mode = property(lambda self: 0.5 * (self._start + self._end))
    std = property(lambda self: (self._end - self._start) / math.sqrt(12.0))

    @cached_property
    def x(self):
        return np.linspace(self._start, self._end, self.GRID)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x < self._start) | (x > self._end), 0.0,
                        1.0 / (self._end - self._start))

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - self._start) / (self._end - self._start), 0.0, 1.0)

    def inv_cdf(self, p):
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        return self._start + p * (self._end - self._start)


class Logistic_PDF(Distribution):
    """Logistic, given its location mu and scale s."""

    def __init__(self, mu, s):
        if s <= 0:
            raise ValueError('s must be positive')
        self.mu, self.s = float(mu), float(s)
        self.title_pdf = 'Logistic Distribution for mu = %g, s = %g' % (mu, s)
        self.title_cdf = 'Logistic Cumulative Distribution for mu = %g, s = %g' % (mu, s)

    mean = property(lambda self: self.mu)
    median = property(lambda self: self.mu)
    mode = property(lambda self: self.mu)
    std = property(lambda self: self.s * math.pi / math.sqrt(3.0))

    def pdf(self, x):
        return logistic(self.mu, self.s, x)

    def cdf(self, x):
        z = -(np.asarray(x, dtype=float) - self.mu) / self.s
        return 1.0 / (1.0 + np.exp(np.clip(z, -700, 700)))

    def inv_cdf(self, p):
        p = np.clip(np.asarray(p, dtype=float), _TINY, 1 - _TINY)
        return self.mu + self.s * np.log(p / (1 - p))


Logisitic_PDF = Logistic_PDF          # the original spelling, kept working


class Weibull_PDF(Distribution):
    """Weibull, given a shape and a scale."""

    def __init__(self, shape, scale):
        if shape <= 0 or scale <= 0:
            raise ValueError('shape and scale must be positive')
        self.k, self.l = float(shape), float(scale)
        self.title_pdf = 'Weibull Distribution for shape = %g, scale = %g' % (shape, scale)
        self.title_cdf = 'Weibull Cumulative Distribution for shape = %g, scale = %g' % (shape, scale)

    @cached_property
    def mean(self):
        return self.l * math.gamma(1 + 1 / self.k)

    @cached_property
    def std(self):
        g1 = math.gamma(1 + 1 / self.k)
        g2 = math.gamma(1 + 2 / self.k)
        return self.l * math.sqrt(g2 - g1 * g1)

    median = property(lambda self: self.l * math.log(2.0) ** (1 / self.k))

    @property
    def mode(self):
        if self.k <= 1:
            return 0.0
        return self.l * ((self.k - 1) / self.k) ** (1 / self.k)

    def pdf(self, x):
        return weibull(self.l, self.k, x)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x <= 0, 0.0, 1 - np.exp(-((np.maximum(x, 0) / self.l) ** self.k)))

    def inv_cdf(self, p):
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1 - _TINY)
        return self.l * (-np.log1p(-p)) ** (1.0 / self.k)

    inverse_cdf = inv_cdf          # the original name, kept working


class Trunc_norm_PDF(Distribution):
    """A normal restricted to [a, b] and renormalised over that interval."""

    def __init__(self, mu, sigma, a, b):
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        if b <= a:
            raise ValueError('b must exceed a')
        self.mu, self.sigma, self.a, self.b = float(mu), float(sigma), float(a), float(b)
        self.norm = Normal_PDF(self.mu, self.sigma)
        self._Fa = float(self.norm.cdf(self.a))
        self._Fb = float(self.norm.cdf(self.b))
        self._Z = self._Fb - self._Fa
        if self._Z <= 0:
            raise ValueError('the truncation interval carries no probability')

    @property
    def mode(self):
        return min(max(self.mu, self.a), self.b)

    def _alpha_beta(self):
        al = (self.a - self.mu) / self.sigma
        be = (self.b - self.mu) / self.sigma
        phi = lambda z: math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        return al, be, phi(al), phi(be)

    @cached_property
    def mean(self):
        al, be, pa, pb = self._alpha_beta()
        return self.mu + self.sigma * (pa - pb) / self._Z

    @cached_property
    def std(self):
        al, be, pa, pb = self._alpha_beta()
        r = (pa - pb) / self._Z
        var = 1.0 + (al * pa - be * pb) / self._Z - r * r
        return self.sigma * math.sqrt(max(var, 0.0))

    @cached_property
    def x(self):
        return np.linspace(self.a, self.b, self.GRID)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.a) & (x <= self.b),
                        np.asarray(self.norm.pdf(x), dtype=float) / self._Z, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (np.asarray(self.norm.cdf(np.clip(x, self.a, self.b)), dtype=float)
                  - self._Fa) / self._Z
        return np.clip(np.where(x < self.a, 0.0, np.where(x > self.b, 1.0, inside)), 0.0, 1.0)

    def inv_cdf(self, p):
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        return self.norm.inv_cdf(self._Fa + p * self._Z)


# --------------------------------------------------------------------------
#  Distributions built from data
# --------------------------------------------------------------------------
class Empirical_PDF(Distribution):
    """A distribution built from an array of observations or draws.

    The cdf and the inverse cdf are read off the order statistics, so a
    percentile of the object is the percentile of the data. The histogram is
    kept for the density and for plotting.
    """

    def __init__(self, array, name='', nbins=0):
        v = np.asarray(array, dtype=float).ravel()
        v = v[np.isfinite(v)]
        if v.size < 2:
            raise ValueError('need at least two finite observations')
        self.array = v
        self.name = name
        self.bins = 'fd' if nbins == 0 else nbins
        self._sorted = np.sort(v)
        self._grid = (np.arange(v.size) + 0.5) / v.size

        heights, edges = np.histogram(v, bins=self.bins, density=True)
        self.hist = (heights, edges)
        self.centers = 0.5 * (edges[:-1] + edges[1:])
        self._px = np.concatenate(([edges[0]], self.centers, [edges[-1]]))
        self._py = np.concatenate(([0.0], heights, [0.0]))

        self.title = 'Distribution of %s' % name
        self.title_cdf = ('Empirical Cumulative Distribution for mean = %.0f, std = %.0f'
                          % (self.mean, self.std))

    @cached_property
    def x(self):
        return np.linspace(self.centers.min(), self.centers.max(), self.GRID)

    @cached_property
    def mean(self):
        return float(self.array.mean())

    @cached_property
    def std(self):
        return float(self.array.std())

    def pdf(self, x):
        return np.interp(np.asarray(x, dtype=float), self._px, self._py, left=0.0, right=0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.searchsorted(self._sorted, x, side='right') / self._sorted.size

    def inv_cdf(self, p):
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        return np.interp(p, self._grid, self._sorted)


class ArrayPDF(Distribution):
    """A density given as a grid and its heights: an empirical PMF.

    Normalised by the trapezoid rule over the grid, so the heights do not have
    to arrive normalised.
    """

    def __init__(self, x, y, name=''):
        gx = np.asarray(x, dtype=float).ravel()
        gy = np.asarray(y, dtype=float).ravel()
        if gx.size != gy.size or gx.size < 3:
            raise ValueError('x and y must be the same length, at least 3 points')
        order = np.argsort(gx)
        gx, gy = gx[order], np.clip(gy[order], 0.0, None)
        area = float(np.trapezoid(gy, gx))
        if area <= 0:
            raise ValueError('the density integrates to zero')
        self.name = name
        self.gx, self.gy = gx, gy / area
        self.norm_fact = area
        self.norm_y = self.gy
        self.f = lambda t: np.interp(np.asarray(t, dtype=float), gx, gy,
                                     left=0.0, right=0.0)

        c = np.concatenate(([0.0], np.cumsum(0.5 * (self.gy[1:] + self.gy[:-1]) * np.diff(gx))))
        self._cdf_vals = np.clip(c / c[-1], 0.0, 1.0)
        # a strictly increasing cdf, so the inverse is single valued
        self._mono = np.maximum.accumulate(self._cdf_vals + np.arange(gx.size) * 1e-15)
        self.title_pdf = 'Array Distribution %s' % name
        self.title_cdf = 'Array Cumulative Distribution %s' % name

    @cached_property
    def x(self):
        return self.gx

    @cached_property
    def y(self):
        return self.gy

    def pdf(self, x):
        return np.interp(np.asarray(x, dtype=float), self.gx, self.gy, left=0.0, right=0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.clip(np.interp(x, self.gx, self._cdf_vals,
                                 left=0.0, right=1.0), 0.0, 1.0)

    def inv_cdf(self, p):
        p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        return np.interp(p, self._mono, self.gx)


class Quantile_PDF(Distribution):
    """A distribution given as a table of (p, value) pairs.

    This is the form a scheduling or costing tool exports. Interpolation runs
    against the normal quantile of p, in log space when every value is
    positive, so a normal or lognormal source comes back almost exactly. The
    outer segments are extended rather than clipped, so the tail is not
    truncated at the last percentile the source happened to print.
    """

    def __init__(self, points, name='', min_points=5):
        pts = sorted(((float(p), float(v)) for p, v in points), key=lambda t: t[0])
        if len(pts) < min_points:
            raise ValueError('need at least %d points, got %d' % (min_points, len(pts)))
        p = np.array([t[0] for t in pts])
        v = np.array([t[1] for t in pts])
        if p[0] <= 0 or p[-1] >= 1:
            raise ValueError('probabilities must lie strictly inside (0, 1)')
        if np.any(np.diff(p) <= 0):
            raise ValueError('probabilities must be strictly increasing')
        if np.any(np.diff(v) < 0):
            raise ValueError('values must be non-decreasing in p')
        self.points, self.name = pts, name
        self.p, self.v = p, v
        self.z = PhiInv(p)
        self.log = bool(np.all(v > 0))
        self.w = np.log(v) if self.log else v
        self.title_pdf = 'Quantile Distribution %s' % name
        self.title_cdf = 'Quantile Cumulative Distribution %s' % name

    def _w_at(self, z):
        w = np.interp(z, self.z, self.w)
        left, right = z < self.z[0], z > self.z[-1]
        if left.any():
            s = (self.w[1] - self.w[0]) / (self.z[1] - self.z[0])
            w = np.where(left, self.w[0] + s * (z - self.z[0]), w)
        if right.any():
            s = (self.w[-1] - self.w[-2]) / (self.z[-1] - self.z[-2])
            w = np.where(right, self.w[-1] + s * (z - self.z[-1]), w)
        return w

    def inv_cdf(self, p):
        p = np.clip(np.asarray(p, dtype=float), _TINY, 1 - _TINY)
        w = self._w_at(PhiInv(p))
        return np.exp(w) if self.log else w

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        u = np.linspace(1e-6, 1 - 1e-6, 4001)
        return np.clip(np.interp(x, self.inv_cdf(u), u, left=0.0, right=1.0), 0.0, 1.0)

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        h = np.maximum(np.abs(x) * 1e-4, 1e-6)
        return np.maximum((self.cdf(x + h) - self.cdf(x - h)) / (2 * h), 0.0)
