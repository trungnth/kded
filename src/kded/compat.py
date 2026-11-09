"""
`kded1` and `kded2`: drop-in replacements for `kde_diffusion.kde1d` and
`kde_diffusion.kde2d` of the Python package KDE-diffusion (same arguments
and return values), computed with `kdedn`.

Differences from kde_diffusion 1.0.5:
- `grid` holds the cell centres, not the left bin edges (the density
  values belong to the centres).
- The fixed point t = gamma(t) of the paper is solved; kde2d.m solved
  t = (t - gamma)/gamma, which differs by a relative O(t*), about 0.1%.
- Linear binning by default; `binning='simple'` gives the old histogram.
- Default limits are the data range plus 1/10 of it on each side for kded1
  (as kde.m) and plus 1/4 for kded2 (as kde2d.m), like kde_diffusion.
"""

import numpy as np

from .core import kdedn


def _limits_1d(lim):
    if lim is None:
        return None
    if np.isscalar(lim):
        return (-lim, lim)
    return tuple(lim)


def kded1(x, n=2**14, limits=None, binning='linear', refine=False):
    """Returns (density, grid, bandwidth) like kde_diffusion.kde1d, on a grid
    of the requested size (refine=False; see kdedn for the resolution check).
    `limits` is None, a scalar a for (-a, a), or a tuple (xmin, xmax);
    a missing end is the data extreme -/+ 1/10 of the data range."""
    x = np.asarray(x, dtype=float).ravel()
    n = int(2**np.ceil(np.log2(n)))
    lim = _limits_1d(limits)
    if lim is None or None in lim:
        delta = x.max() - x.min()
        lo, hi = (None, None) if lim is None else lim
        lim = (x.min() - delta/10 if lo is None else lo,
               x.max() + delta/10 if hi is None else hi)
    res = kdedn(x[:, None], n=n, limits=[lim], binning=binning, refine=refine)
    return res.density, res.grid[0], float(res.bandwidth[0])


def kded2(x, y, n=256, limits=None, binning='linear', refine=False):
    """Returns (density, (xgrid, ygrid), bandwidth) like kde_diffusion.kde2d
    (on the requested grid unless refine=True),
    with density[i, j] at (xgrid[i], ygrid[j]). `limits` is None, a scalar,
    or a tuple (xlimits, ylimits) whose entries are None, a scalar or
    (min, max)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError('x and y must have the same length.')
    n = int(2**np.ceil(np.log2(n)))
    if limits is None or np.isscalar(limits):
        lims = [_limits_1d(limits)]*2
    else:
        lims = [_limits_1d(lim) for lim in limits]
    res = kdedn(np.c_[x, y], n=n, limits=lims, binning=binning, refine=refine)
    return res.density, res.grid, res.bandwidth
