"""
Kernel density estimation via diffusion in d dimensions (d = 1, 2, 3, ...).

Generalisation of Botev, Grotowski & Kroese (2010), "Kernel density
estimation via diffusion", Ann. Statist. 38(5), to arbitrary dimension,
with the main use case d = 3.

The algorithm:

1. Rescale the data to the unit cube [0, 1]^d and bin it on a regular
   grid of n_1 x ... x n_d cells (simple or linear binning).
2. Take the d-dimensional DCT of the binned data. In the cosine basis the
   Neumann heat semigroup (= Gaussian smoothing with reflecting boundaries)
   is diagonal, and so are all functionals

       psi_s = || D^s f ||^2 = int (d^{|s|} f / dx_1^{s_1} ... dx_d^{s_d})^2 dx,

   which become weighted sums of squared DCT coefficients.
3. Improved Sheather-Jones (ISJ) fixed point: for a trial time t, set the
   pilot time of every functional of order |s| = L to t, then descend
   the order using the multivariate AMSE-balancing pilot formula

       t_s = ( 2 (1 + 2^{-(|s|+d/2)}) / 3 * prod_i q(s_i)
               / (N * sum_i psi_{s+e_i}) )^{1 / (|s| + d/2 + 1)},

   with q(j) = (2j-1)!! / sqrt(2 pi), until the second order functionals
   psi_{e_i+e_j} are known. They define the AMISE optimal isotropic time

       gamma(t) = ( d / (N (4 pi)^{d/2} S) )^{2/(d+4)},
       S = sum_{i,j} psi_{e_i+e_j},

   and t* solves t = gamma(t).
4. With the second order functionals at t*, minimise the AMISE of a
   diagonal bandwidth matrix diag(t_1, ..., t_d),

       AMISE = 1 / (N (4 pi)^{d/2} sqrt(t_1 ... t_d))
               + 1/4 sum_{i,j} t_i t_j psi_{e_i+e_j},

   which for d = 2 reproduces the closed form used in kde2d.m.
5. Smooth the DCT coefficients with exp(-pi^2 k_i^2 t_i / 2) per axis and
   transform back.

Optionally the data is first rotated onto its principal axes, so that
the diagonal bandwidth in the rotated frame is a full bandwidth matrix
H = R diag(t) R^T in the original frame.
"""

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from functools import reduce
import warnings

import numpy as np
from numpy import pi as PI
from scipy.fft import dctn, idctn, next_fast_len
from scipy.optimize import brentq, minimize
from scipy.interpolate import RegularGridInterpolator


########################################
# Result container                     #
########################################

@dataclass
class KDEResult:
    """Output of `kdedn` / `kded3`."""

    density: np.ndarray
    """Density values on the grid, shape (n_1, ..., n_d), indexed [i, j, k]
    in the order of the data columns (`indexing='ij'`). If `rotate` was
    used, the grid lives in the rotated (principal axes) frame."""
    grid: tuple
    """Cell centres along each axis of the (possibly rotated) frame."""
    bandwidth: np.ndarray
    """Per axis standard deviations of the Gaussian kernel (in the rotated
    frame if `rotate` was used)."""
    H: np.ndarray
    """Bandwidth matrix (kernel covariance) in the original data frame."""
    t_star: float
    """Fixed point of the ISJ equation, in unit cube coordinates."""
    psi2: np.ndarray
    """Matrix of estimated second order functionals int f_ii f_jj,
    in unit cube coordinates."""
    rotation: np.ndarray = None
    """Orthogonal matrix R; rotated coordinates are (x - centre) @ R."""
    centre: np.ndarray = None
    cells: float = None
    """Resolution diagnostic: the smallest kernel standard deviation, at the
    ISJ time or at the final bandwidth, measured in grid cells. Values below
    about 2 mean that the grid does not resolve the bandwidth."""
    _interp: object = field(default=None, repr=False)

    def _to_frame(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if self.rotation is not None:
            points = (points - self.centre) @ self.rotation
        return points

    def evaluate(self, points):
        """Interpolates the gridded density at arbitrary points (M x d) given
        in the original data frame. Points outside the grid get 0."""
        if self._interp is None:
            self._interp = RegularGridInterpolator(
                self.grid, self.density, bounds_error=False, fill_value=0.0)
        return np.maximum(self._interp(self._to_frame(points)), 0.0)

    def evaluate_exact(self, points, data):
        """Evaluates the exact (unbinned) Gaussian KDE with bandwidth
        matrix `H` at `points`. Slow: O(len(points) * len(data))."""
        return gaussian_kde_eval(data, self.H, points)

    def __iter__(self):
        # Allows `density, grid, bandwidth = kdedn(...)`, as in kde_diffusion.
        return iter((self.density, self.grid, self.bandwidth))


########################################
# Helpers                              #
########################################

def _q(j):
    """|q(j)| = (2j-1)!! / sqrt(2 pi), the 1d Gaussian derivative constant."""
    return np.prod(np.arange(1, 2*j, 2, dtype=float)) / np.sqrt(2*PI)


def _multi_indices(d, order):
    """All multi-indices s in N^d with |s| = order, as tuples."""
    out = []
    for combo in combinations_with_replacement(range(d), order):
        s = [0]*d
        for i in combo:
            s[i] += 1
        out.append(tuple(s))
    return out


def _unit(d, i):
    e = [0]*d
    e[i] = 1
    return tuple(e)


def _add(s, e):
    return tuple(a + b for (a, b) in zip(s, e))


def _bin(u, shape, method):
    """Bins points u (N x d, in unit cube) on a grid with cell centres at
    (j + 1/2)/n_i. Returns probabilities summing to the fraction inside."""
    N, d = u.shape
    shape = tuple(shape)
    if method == 'simple':
        idx = [np.clip(np.floor(u[:, i]*shape[i]).astype(int), 0, shape[i]-1)
               for i in range(d)]
        flat = np.ravel_multi_index(idx, shape)
        return np.bincount(flat, minlength=int(np.prod(shape))
                           ).reshape(shape).astype(float) / N
    if method != 'linear':
        raise ValueError("binning must be 'linear' or 'simple'.")
    # Linear binning: split each point's mass among the 2^d neighbouring
    # cell centres; mass beyond the outermost centres is reflected back,
    # consistent with the Neumann boundary of the DCT.
    g = [u[:, i]*shape[i] - 0.5 for i in range(d)]
    j0 = [np.floor(gi).astype(int) for gi in g]
    fr = [gi - ji for (gi, ji) in zip(g, j0)]
    size = int(np.prod(shape))
    out = np.zeros(size)
    for corner in range(2**d):
        w = np.ones(N)
        idx = []
        for i in range(d):
            bit = (corner >> i) & 1
            w = w * (fr[i] if bit else 1 - fr[i])
            idx.append(np.clip(j0[i] + bit, 0, shape[i]-1))
        out += np.bincount(np.ravel_multi_index(idx, shape), weights=w,
                           minlength=size)
    return out.reshape(shape) / N


def _parse_limits(data, limits):
    d = data.shape[1]
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    xmin = lo - rng/4
    xmax = hi + rng/4
    if limits is not None:
        limits = list(limits)
        if len(limits) != d:
            raise ValueError(f'limits must have one entry per dimension ({d}).')
        for i, lim in enumerate(limits):
            if lim is None:
                continue
            if np.isscalar(lim):
                xmin[i], xmax[i] = -lim, +lim
            else:
                if lim[0] is not None:
                    xmin[i] = lim[0]
                if lim[1] is not None:
                    xmax[i] = lim[1]
    if np.any(xmax <= xmin):
        raise ValueError('Each upper limit must exceed its lower limit.')
    return xmin, xmax


########################################
# Bandwidth selection                  #
########################################

class _ISJ:
    """Improved Sheather-Jones machinery on the DCT coefficients."""

    def __init__(self, a, N, max_order):
        self.d = a.ndim
        self.N = N
        self.L = max_order
        self.A2 = a**2
        self.k2 = [np.arange(n, dtype=float)**2 for n in a.shape]
        self.w = []
        for n in a.shape:
            w = np.full(n, 0.5)
            w[0] = 1.0
            self.w.append(w)
        # Multi-indices per order, from L down to 2.
        self.levels = {m: _multi_indices(self.d, m)
                       for m in range(2, max_order + 1)}

    def _vec(self, i, si, t):
        return self.w[i] * self.k2[i]**si * np.exp(-PI**2 * self.k2[i] * t)

    def psi(self, s, t):
        """Estimate of psi_s = int (D^s f)^2 with pilot time t, i.e. the
        binned data smoothed with a Gaussian of variance t/2 per axis
        (equivalently D^{2s} of a kernel of variance t, as in eq. (e54))."""
        return self.psi_many([(s, t)])[0]

    def psi_many(self, items):
        """psi_s(t) for a list of (s, t) pairs with one pass over A2: the
        last axis is contracted for all pairs at once (a matrix product),
        the remaining small axes pair by pair."""
        d = self.d
        V = np.stack([self._vec(d - 1, s[d - 1], t) for s, t in items], axis=1)
        B = np.moveaxis(self.A2 @ V, -1, 0)
        out = []
        for j, (s, t) in enumerate(items):
            acc = B[j]
            for i in range(d - 2, -1, -1):
                acc = acc @ self._vec(i, s[i], t)
            out.append(float(acc) * PI**(2*sum(s)))
        return out

    def second_order(self, t):
        """Runs the pilot cascade for top-level time t and returns the dict
        of second order functionals psi_s, |s| = 2."""
        d, N = self.d, self.N
        top = self.levels[self.L]
        psi = dict(zip(top, self.psi_many([(s, t) for s in top])))
        for m in range(self.L - 1, 1, -1):
            a = m + d/2
            items = []
            for s in self.levels[m]:
                S = sum(psi[_add(s, _unit(d, i))] for i in range(d))
                S = abs(S)
                C = 2*(1 + 2**(-a))/3 * np.prod([_q(si) for si in s])
                items.append((s, (C / (N*S))**(1/(a + 1))))
            psi = dict(zip(self.levels[m], self.psi_many(items)))
        return psi

    def psi2_matrix(self, psi):
        d = self.d
        M = np.empty((d, d))
        for i in range(d):
            for j in range(d):
                M[i, j] = psi[_add(_unit(d, i), _unit(d, j))]
        return M

    def gamma(self, t):
        d, N = self.d, self.N
        S = self.psi2_matrix(self.second_order(t)).sum()
        return (d / (N * (4*PI)**(d/2) * S))**(2/(d + 4))


def amise_diagonal(psi2, N):
    """Minimises the AMISE of a Gaussian KDE with diagonal bandwidth
    diag(t_1..t_d) given the matrix psi2[i,j] = int f_ii f_jj.
    Returns t (squared bandwidths)."""
    d = psi2.shape[0]
    c = 1 / (N * (4*PI)**(d/2))

    def f(logt):
        t = np.exp(logt)
        var = c * np.exp(-0.5*logt.sum())
        bias = 0.25 * t @ psi2 @ t
        grad = -0.5*var + 0.5 * t * (psi2 @ t)
        return var + bias, grad

    # Isotropic optimum as a starting point.
    t0 = (d / (N * (4*PI)**(d/2) * psi2.sum()))**(2/(d + 4))
    res = minimize(f, np.full(d, np.log(t0)), jac=True, method='BFGS',
                   options={'gtol': 1e-12, 'maxiter': 500})
    return np.exp(res.x)


def _solve_fixed_point(isj, matlab_quirk=False):
    """Smallest root of t = gamma(t).

    t - gamma(t) is negative for tiny t and turns positive at the fixed
    point. For larger t the pilot cascade oversmooths and the sign can flip
    back and forth, so (0, 0.1] (the bracket of kde.m and kde_diffusion)
    may hold several roots, and Brent's method may then return a spurious
    large one. The smallest root is found by scanning t upwards on a
    half-decade log grid (Botev's later kde.m does the same by doubling an
    initial interval) and refining the first bracket with Brent's method.
    Returns None if no root exists below t = 1.
    """
    # brentq wraps g in a self-referencing closure that stays alive until the
    # cyclic garbage collector runs; g therefore reaches isj only through
    # `ref`, which is cleared on return, so that the coefficient arrays of
    # isj are released at once (otherwise repeated calls on large grids
    # accumulate memory).
    ref = [isj]
    if matlab_quirk:
        # kde2d.m / kde_diffusion solve t = (t - gamma(t)) / gamma(t), which
        # shifts t* by a relative O(t*). Only used to reproduce them in tests.
        g = lambda t: t - (t - ref[0].gamma(t))/ref[0].gamma(t)
    else:
        g = lambda t: t - ref[0].gamma(t)
    lower = 1e-12
    try:
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            g_lower = g(lower)
            if not np.isfinite(g_lower):
                return None
            prev = lower
            for upper in np.logspace(-11.5, 0, 24):
                g_upper = g(upper)
                if not np.isfinite(g_upper):
                    break
                if np.sign(g_upper) != np.sign(g_lower):
                    return brentq(g, prev, upper, xtol=1e-14, rtol=1e-12)
                prev = upper
        return None
    finally:
        ref.clear()


########################################
# Main                                 #
########################################

def _select(u, N, shape, binning, max_order, matlab_quirk=False):
    """Bins u (points in [0,1]^d) on `shape`, transforms, and selects the
    bandwidth. Returns (a, t_star, psi2, t) in unit cube coordinates."""
    d = len(shape)
    p = _bin(u, shape, binning) * (len(u) / N)
    # DCT coefficients c_k of the binned density on [0,1]^d such that
    # p(x) = sum_k c_k prod_i cos(pi k_i x_i).
    a = dctn(p, type=2)
    for ax in range(d):
        sl = [slice(None)]*d
        sl[ax] = 0
        a[tuple(sl)] /= 2
    isj = _ISJ(a, N, max_order)
    t_star = _solve_fixed_point(isj, matlab_quirk)
    if t_star is None:
        raise ValueError('Bandwidth optimization did not converge.')
    psi2 = isj.psi2_matrix(isj.second_order(t_star))
    t = np.array([t_star]) if d == 1 else amise_diagonal(psi2, N)
    return a, t_star, psi2, t


def _fast_floor(m):
    """Largest k <= m (k >= 1) with an efficient FFT length."""
    m = max(int(m), 1)
    while next_fast_len(m) != m:
        m -= 1
    return m


def _default_grid(d, max_cells=2**24):
    """Default initial grid points per axis: 2**14, 2**8, 2**7 and 48 for
    d = 1, 2, 3 and 4; for d >= 5 the largest efficient size m with
    m**d <= max_cells (integer arithmetic, so that m = 16 for d = 6)."""
    if d in (1, 2, 3, 4):
        return {1: 2**14, 2: 2**8, 3: 2**7, 4: 48}[d]
    budget = max(int(max_cells), 1)
    m = max(int(round(budget**(1 / d))), 1)
    while m > 1 and m**d > budget:
        m -= 1
    while (m + 1)**d <= budget:
        m += 1
    return _fast_floor(m)


def _cells(t_star, t, shape):
    return float((np.sqrt(np.minimum(t, t_star)) * np.asarray(shape)).min())


def _refined_shape(t_star, t, shape, min_cells, max_cells):
    """Grid that resolves the smaller of the ISJ time and the final time on
    each axis by `min_cells` cells, within a budget of `max_cells` cells."""
    tau = np.minimum(t, t_star)
    want = [max(m, next_fast_len(int(np.ceil(min_cells / np.sqrt(ti)))))
            for m, ti in zip(shape, tau)]
    grow = [i for i in range(len(shape)) if want[i] > shape[i]]
    if not grow:
        return tuple(shape)
    fixed = np.prod([shape[i] for i in range(len(shape)) if i not in grow],
                    dtype=float)
    budget = max_cells / fixed
    total = np.prod([want[i] for i in grow], dtype=float)
    if total > budget:
        f = (budget / total)**(1 / len(grow))
        for i in grow:
            want[i] = max(shape[i], _fast_floor(want[i] * f))
    return tuple(int(w) for w in want)


def kdedn(data, n=None, limits=None, binning='linear', max_order=None,
          rotate=False, refine=True, min_cells=2.0, max_cells=2**24,
          max_refine=3, _matlab_quirk=False):
    """
    Kernel density estimation via diffusion for d-dimensional data.

    Parameters
    ----------
    data : array (N, d)
        Observations, one row per point.
    n : int or sequence of int
        Grid points per axis. Default 2**14 / 2**8 / 2**7 / 48 for
        d = 1 / 2 / 3 / 4, and for d >= 5 the largest efficient size whose
        d-th power is within `max_cells`. Powers of two are fastest but not
        required.
    limits : sequence of length d, optional
        Per axis `(min, max)`, a scalar `a` meaning `(-a, a)`, or None.
        Missing limits default to the data range extended by 1/4 on each side.
    binning : 'linear' (default) or 'simple'
        'simple' reproduces the histogram binning of kde.m / kde2d.m.
    max_order : int, optional
        Order L of the functionals at which the pilot cascade starts
        (stages = L - 1). Defaults to 7 for d = 1 (kde.m) and 5 otherwise
        (kde2d.m).
    rotate : bool
        If True, rotate the data onto its principal axes first, giving a
        full (non-diagonal) bandwidth matrix in the original frame.
    refine : bool
        Resolution check (default True). After selection, check that the
        kernel standard deviation (at the ISJ time and at the final
        bandwidth) spans at least `min_cells` grid cells along every axis; if
        not, enlarge the grid on the offending axes, within a budget of
        `max_cells` cells in total, and select again (at most `max_refine`
        times). With refine=False the grid `n` is used as given.
    min_cells, max_cells, max_refine : float, int, int
        Target resolution, grid budget and maximal number of refinements for
        `refine`. The target and the budget are tuning constants of the
        selector: the selected bandwidth depends on the resolution at which
        the functionals are estimated. The achieved resolution is returned as
        `KDEResult.cells`.

    Returns
    -------
    KDEResult
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    N, d = data.shape
    if N <= d:
        raise ValueError('data must be an N x d array with N > d.')
    if not np.all(np.isfinite(data)):
        raise ValueError('data contains NaN or infinite values.')
    if n is None:
        n = _default_grid(d, max_cells)
    shape = (int(n),)*d if np.isscalar(n) else tuple(int(m) for m in n)
    if len(shape) != d:
        raise ValueError('n must be a scalar or have one entry per dimension.')
    if max_order is None:
        max_order = 7 if d == 1 else 5
    if max_order < 3:
        raise ValueError('max_order must be at least 3.')

    centre = R = None
    X = data
    if rotate and d > 1:
        centre = data.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov(data, rowvar=False))
        R = vecs[:, ::-1]
        X = (data - centre) @ R

    xmin, xmax = _parse_limits(X, limits)
    span = xmax - xmin
    u = (X - xmin) / span
    inside = np.all((u >= 0) & (u <= 1), axis=1)
    if not inside.all():
        warnings.warn(f'{(~inside).sum()} points lie outside the limits '
                      'and are ignored.')
        u = u[inside]
    a, t_star, psi2, t = _select(u, N, shape, binning, max_order,
                                 _matlab_quirk)
    if refine:
        for _ in range(max_refine):
            if _cells(t_star, t, shape) >= min_cells:
                break
            new_shape = _refined_shape(t_star, t, shape, min_cells, max_cells)
            if new_shape == tuple(shape):
                break
            shape = new_shape
            a, t_star, psi2, t = _select(u, N, shape, binning, max_order,
                                         _matlab_quirk)
    cells = _cells(t_star, t, shape)
    if refine and cells < min_cells:
        if _refined_shape(t_star, t, shape, min_cells, max_cells) == tuple(shape):
            why = ('the grid budget max_cells is reached. Consider narrower '
                   'limits or a larger max_cells.')
        else:
            why = (f'max_refine = {max_refine} refinements are used up. '
                   'Consider a larger max_refine.')
        warnings.warn(f'The grid {shape} resolves the selected bandwidth by '
                      f'only {cells:.2f} cells (target {min_cells}); {why}')
    elif not refine and cells < 1:
        warnings.warn(f'The grid {shape} resolves the selected bandwidth by '
                      f'only {cells:.2f} cells; the selection may be '
                      'impaired. Consider refine=True, a larger n or '
                      'narrower limits.')

    # Smooth: multiply by heat kernel exp(-pi^2 k^2 t_i / 2) along each axis.
    smoothed = a.copy()
    for ax in range(d):
        k2 = np.arange(shape[ax], dtype=float)**2
        f = np.exp(-PI**2 * k2 * t[ax] / 2)
        smoothed *= f.reshape([-1 if i == ax else 1 for i in range(d)])
    for ax in range(d):
        sl = [slice(None)]*d
        sl[ax] = 0
        smoothed[tuple(sl)] *= 2
    density = idctn(smoothed, type=2) * np.prod(np.array(shape) / span)

    grid = tuple(xmin[i] + (np.arange(shape[i]) + 0.5) * span[i]/shape[i]
                 for i in range(d))
    bandwidth = np.sqrt(t) * span
    Hrot = np.diag(bandwidth**2)
    H = Hrot if R is None else R @ Hrot @ R.T
    return KDEResult(density=density, grid=grid, bandwidth=bandwidth, H=H,
                     t_star=t_star, psi2=psi2, rotation=R, centre=centre,
                     cells=cells)


def kded3(data, n=128, limits=None, binning='linear', max_order=5,
          rotate=False, refine=True, min_cells=2.0, max_cells=2**24,
          max_refine=3):
    """3d kernel density estimation via diffusion. `data` is N x 3.
    See `kdedn` for the parameters."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError('data must be an N x 3 array.')
    return kdedn(data, n=n, limits=limits, binning=binning,
                 max_order=max_order, rotate=rotate, refine=refine,
                 min_cells=min_cells, max_cells=max_cells,
                 max_refine=max_refine)


########################################
# Adaptive (variable bandwidth) KDE     #
########################################

def _abramson(fp, alpha, clip):
    """Abramson factors (f~ / geometric mean)^(-alpha), clipped."""
    logf = np.log(np.maximum(fp, 1e-300))
    return np.clip(np.exp(-alpha * (logf - logf.mean())), *clip)


def _lscv_select(X, H, fp, alphas, scales, clip, chunk=2**21):
    """Least squares cross-validation over (alpha, scale) for the adaptive
    KDE with kernel covariances (scale * lambda_i)^2 H. All terms are in
    closed form for Gaussian kernels:
        LSCV = int fhat^2 - 2/(N(N-1)) sum_{i != j} phi_{s^2 l_j^2 H}(X_i - X_j).
    Returns the minimising (alpha, scale) and the score table."""
    N, d = X.shape
    Linv = np.linalg.cholesky(np.linalg.inv(H))
    Y = X @ Linv
    sq = (Y**2).sum(1)
    norm = 1 / np.sqrt((2*PI)**d * np.linalg.det(H))
    lam2 = {a: _abramson(fp, a, clip)**2 for a in alphas}
    int_f2 = np.zeros((len(alphas), len(scales)))
    loo = np.zeros_like(int_f2)
    step = max(1, chunk // N)
    for i in range(0, N, step):
        D2 = np.maximum(sq[i:i+step, None] + sq[None, :]
                        - 2 * Y[i:i+step] @ Y.T, 0)
        diag = np.arange(i, min(i + step, N))
        for (ia, a) in enumerate(alphas):
            l2 = lam2[a]
            V2 = l2[i:i+step, None] + l2[None, :]
            V1 = np.broadcast_to(l2[None, :], D2.shape)
            for (ic, c) in enumerate(scales):
                c2 = c*c
                A = np.exp(-0.5 * D2 / (c2*V2)) * (c2*V2)**(-d/2)
                B = np.exp(-0.5 * D2 / (c2*V1)) * (c2*V1)**(-d/2)
                B[diag - i, diag] = 0
                int_f2[ia, ic] += A.sum()
                loo[ia, ic] += B.sum()
    score = norm * (int_f2 / N**2 - 2 * loo / (N * (N - 1)))
    ia, ic = np.unravel_index(np.argmin(score), score.shape)
    return alphas[ia], scales[ic], score


def kdedn_adaptive(data, n=None, limits=None, alpha=None, scale=None,
                   rotate=False, classes=24, clip=(0.1, 10.0), pilot=None,
                   cv_max=3000, seed=0):
    """
    Sample-point adaptive KDE (Abramson 1982 square-root law) built on the
    diffusion estimator, for densities whose features live on different
    scales (several modes with different spreads, sharp peaks on a wide
    background, ...).

    1. Pilot: the diffusion KDE `kdedn(data, n=n, limits=limits,
       rotate=rotate)` gives the bandwidth matrix H and a pilot density f~.
       The default grid `n=None` is that of `kdedn` (128 points per axis
       for d = 3, the default of kde3d_adaptive up to version 0.2.2).
    2. Point i gets the kernel covariance (scale * lambda_i)^2 H with
       lambda_i = (f~(X_i) / g)^(-alpha), g the geometric mean of the
       f~(X_i), clipped to `clip`.
    3. `alpha` and `scale` default to `None`: they are then chosen by
       least squares cross-validation (exact, Gaussian closed forms) over
       alpha in {0, 1/4, 1/2} and scale in [0.5, 2.5] -- alpha = 0,
       scale = 1 is the plain diffusion estimator, so the selector can fall
       back to it. For N > `cv_max` the selection runs on a random
       subsample of `cv_max` points (with its own pilot).
    4. The lambdas are grouped into `classes` log-spaced classes. Each
       class is binned and smoothed by the heat kernel in the DCT domain;
       by linearity the smoothed coefficients are summed and transformed
       back once, so the cost is `classes` forward DCTs.

    Returns a KDEResult with extra attributes `lambdas` (per point factor
    including `scale`, i.e. point i has covariance lambdas[i]^2 H), `alpha`
    and `scale`.
    """
    data = np.asarray(data, dtype=float)
    N, d = data.shape
    if pilot is None:
        pilot = kdedn(data, n=n, limits=limits, rotate=rotate)
    fp = pilot.evaluate(data)

    if alpha is None or scale is None:
        alphas = [0.0, 0.25, 0.5] if alpha is None else [alpha]
        scales = list(np.geomspace(0.5, 2.5, 15)) if scale is None else [scale]
        if N > cv_max:
            rng = np.random.default_rng(seed)
            sub = data[rng.choice(N, cv_max, replace=False)]
            sp = kdedn(sub, n=n, limits=limits, rotate=rotate)
            a_sel, s_sel, _ = _lscv_select(sub, sp.H, sp.evaluate(sub),
                                           alphas, scales, clip)
        else:
            a_sel, s_sel, _ = _lscv_select(data, pilot.H, fp,
                                           alphas, scales, clip)
        alpha = a_sel if alpha is None else alpha
        scale = s_sel if scale is None else scale

    lam = _abramson(fp, alpha, clip) * scale
    # Log-spaced classes; each point is represented by its class centre.
    lo, hi = np.log(lam.min()), np.log(lam.max())
    edges = np.linspace(lo, hi + 1e-12, classes + 1)
    cls = np.clip(np.searchsorted(edges, np.log(lam), side='right') - 1,
                  0, classes - 1)
    centres = np.exp((edges[:-1] + edges[1:]) / 2)
    lam = centres[cls]

    X = data if pilot.rotation is None else \
        (data - pilot.centre) @ pilot.rotation
    grid = pilot.grid
    shape = tuple(len(g) for g in grid)
    span = np.array([(g[-1] - g[0]) * len(g) / (len(g) - 1) for g in grid])
    xmin = np.array([g[0] for g in grid]) - span / (2 * np.array(shape))
    u = (X - xmin) / span
    inside = np.all((u >= 0) & (u <= 1), axis=1)
    t = (pilot.bandwidth / span)**2
    k2 = [np.arange(m, dtype=float)**2 for m in shape]
    cell = span / np.array(shape)
    if np.any(lam.min() * pilot.bandwidth < 1.5 * cell):
        warnings.warn('The narrowest adaptive kernels are smaller than 1.5 '
                      'grid cells; the gridded density under-resolves them. '
                      'Increase n or narrow the limits (lambdas, H and '
                      'adaptive_kde_eval are unaffected).')

    total = np.zeros(shape)
    for c in np.unique(cls):
        sel = (cls == c) & inside
        if not sel.any():
            continue
        p = _bin(u[sel], shape, 'linear') * (sel.sum() / N)
        a = dctn(p, type=2)
        for ax in range(d):
            f = np.exp(-PI**2 * k2[ax] * t[ax] * centres[c]**2 / 2)
            a *= f.reshape([-1 if i == ax else 1 for i in range(d)])
        total += a
    density = idctn(total, type=2) * np.prod(np.array(shape) / span)
    res = KDEResult(density=density, grid=grid, bandwidth=pilot.bandwidth,
                    H=pilot.H, t_star=pilot.t_star, psi2=pilot.psi2,
                    rotation=pilot.rotation, centre=pilot.centre)
    res.lambdas = lam
    res.data = data
    res.alpha = alpha
    res.scale = scale
    return res


########################################
# Exact Gaussian KDE evaluation        #
########################################

def adaptive_kde_eval(data, H, lambdas, points):
    """Exact adaptive Gaussian KDE: point i has covariance lambdas[i]^2 H."""
    out = np.zeros(len(np.atleast_2d(points)))
    for l in np.unique(lambdas):
        sel = lambdas == l
        out += gaussian_kde_eval(data[sel], l**2 * H, points) * sel.mean()
    return out


def gaussian_kde_eval(data, H, points, chunk=2**22):
    """Exact Gaussian KDE with kernel covariance H at `points`."""
    data = np.atleast_2d(np.asarray(data, dtype=float))
    points = np.atleast_2d(np.asarray(points, dtype=float))
    d = data.shape[1]
    H = np.atleast_2d(H)
    L = np.linalg.cholesky(np.linalg.inv(H))
    Xd = data @ L
    Xp = points @ L
    norm = 1 / (len(data) * np.sqrt((2*PI)**d * np.linalg.det(H)))
    out = np.empty(len(points))
    step = max(1, chunk // len(data))
    sq_d = (Xd**2).sum(1)
    for i in range(0, len(points), step):
        P = Xp[i:i+step]
        D2 = (P**2).sum(1)[:, None] + sq_d[None, :] - 2 * P @ Xd.T
        out[i:i+step] = np.exp(-0.5*np.maximum(D2, 0)).sum(1)
    return out * norm
