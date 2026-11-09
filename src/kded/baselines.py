"""
Reference bandwidth selectors to compare against kdedn / kded3.

All functions return the kernel covariance matrix H (d x d) of a Gaussian
KDE in the original data frame.
"""

from math import factorial

import numpy as np
from numpy import pi as PI
from scipy.fft import dctn

from .core import _ISJ, _bin, _parse_limits, _q, _add, _unit, amise_diagonal


def scott(data):
    """scipy.stats.gaussian_kde default: H = N^{-2/(d+4)} Cov."""
    N, d = data.shape
    return N**(-2/(d + 4)) * np.cov(data, rowvar=False)


def silverman(data):
    """scipy.stats.gaussian_kde(bw_method='silverman'):
    H = (N (d+2)/4)^{-2/(d+4)} Cov."""
    N, d = data.shape
    return (N*(d + 2)/4)**(-2/(d + 4)) * np.cov(data, rowvar=False)


def normal_reference_diagonal(data):
    """AMISE optimal diagonal H if the data were N(mu, diag(sigma^2)):
    h_i = sigma_i (4 / ((d+2) N))^{1/(d+4)}."""
    N, d = data.shape
    s = data.std(axis=0, ddof=1)
    return np.diag((s * (4/((d + 2)*N))**(1/(d + 4)))**2)


def _psi_normal(s, sigma):
    """psi_s = int (D^s phi)^2 for phi = N(0, diag(sigma^2))."""
    out = 1.0
    for (j, sg) in zip(s, sigma):
        out *= factorial(2*j) / (2**(2*j + 1) * factorial(j)
                                 * np.sqrt(PI) * sg**(2*j + 1))
    return out


def dpi_diagonal(data, stages=2, n=128, binning='linear'):
    """Classical multi-stage direct plug-in (Wand & Jones 1994; Duong &
    Hazelton 2003 for the diagonal case): like the diffusion estimator,
    but the cascade is started from a normal reference for the highest
    order functionals instead of the fixed point equation."""
    data = np.asarray(data, dtype=float)
    N, d = data.shape
    xmin, xmax = _parse_limits(data, None)
    span = xmax - xmin
    u = (data - xmin) / span
    p = _bin(u, (n,)*d, binning)
    a = dctn(p, type=2)
    for ax in range(d):
        sl = [slice(None)]*d
        sl[ax] = 0
        a[tuple(sl)] /= 2
    L = stages + 2
    isj = _ISJ(a, N, L)
    sigma = u.std(axis=0, ddof=1)
    psi = {s: _psi_normal(s, sigma) for s in isj.levels[L]}
    for m in range(L - 1, 1, -1):
        new = {}
        am = m + d/2
        for s in isj.levels[m]:
            S = sum(psi[_add(s, _unit(d, i))] for i in range(d))
            C = 2*(1 + 2**(-am))/3 * np.prod([_q(si) for si in s])
            new[s] = isj.psi(s, (C / (N*S))**(1/(am + 1)))
        psi = new
    t = amise_diagonal(isj.psi2_matrix(psi), N)
    return np.diag(t * span**2)


def amise_oracle_diagonal(density, N):
    """AMISE optimal diagonal H computed from the true functionals of a
    test density (unattainable in practice; a yardstick)."""
    return np.diag(amise_diagonal(density.psi2(), N))
