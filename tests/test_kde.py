"""Tests for kded."""

import warnings
from pathlib import Path

import numpy as np
import pytest

from kded import kdedn, kded3, gaussian_kde_eval
from kded.core import amise_diagonal
from kded.baselines import normal_reference_diagonal
from kded.testdensities import benchmark_densities, GaussianMixture

REF = Path(__file__).parent / 'reference'


def _limits(r, names):
    return [(float(r[a]), float(r[b])) for (a, b) in names]


def test_reproduces_kde1d():
    """d = 1 must reproduce kde.m / kde_diffusion.kde1d exactly."""
    r = np.load(REF/'reference1d.npz')
    res = kdedn(r['x'], n=int(r['n']), binning='simple',
                limits=_limits(r, [('xmin', 'xmax')]))
    assert np.isclose(res.bandwidth[0], r['bandwidth'], rtol=1e-10)
    assert np.allclose(res.density, r['density'], rtol=1e-8, atol=1e-12)


def test_reproduces_kde2d():
    """d = 2 reproduces kde2d.m up to its fixed point convention."""
    r = np.load(REF/'reference2d.npz')
    lim = _limits(r, [('xmin', 'xmax'), ('ymin', 'ymax')])
    data = np.c_[r['x'], r['y']]
    res = kdedn(data, n=int(r['n']), binning='simple', limits=lim,
                _matlab_quirk=True)
    assert np.allclose(res.bandwidth, r['bandwidth'], rtol=1e-8)
    assert np.allclose(res.density, r['density'], rtol=1e-6, atol=1e-10)
    # The proper fixed point t = gamma(t) differs only by O(t*).
    res = kdedn(data, n=int(r['n']), binning='simple', limits=lim)
    assert np.allclose(res.bandwidth, r['bandwidth'], rtol=1e-3)


def test_amise_diagonal_matches_2d_closed_form():
    P = np.array([[5.0, 1.3], [1.3, 2.0]])
    N = 500
    t = amise_diagonal(P, N)
    p20, p02, p11 = P[0, 0], P[1, 1], P[0, 1]
    c = 4*np.pi*N*(p11 + np.sqrt(p20*p02))
    t1 = (p02**0.75 / (p20**0.75 * c))**(1/3)
    t2 = (p20**0.75 / (p02**0.75 * c))**(1/3)
    assert np.allclose(t, [t1, t2], rtol=1e-8)


def test_density_normalised_and_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 3))
    res = kded3(X, n=64)
    assert res.density.shape == (64, 64, 64)
    dv = np.prod([g[1] - g[0] for g in res.grid])
    assert np.isclose(res.density.sum()*dv, 1, atol=1e-6)
    assert np.all(res.bandwidth > 0)
    density, grid, bandwidth = res          # tuple unpacking like kde_diffusion.kde2d
    assert len(grid) == 3


def test_gaussian_close_to_normal_reference():
    """For Gaussian data the AMISE optimum is the normal reference rule."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50000, 3)) * [1, 2, 0.5]
    h = kded3(X).bandwidth
    h_ref = np.sqrt(np.diag(normal_reference_diagonal(X)))
    assert np.allclose(h, h_ref, rtol=0.1)


def test_scale_equivariance():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 3))
    h1 = kded3(X).bandwidth
    h2 = kded3(X * [10, 1, 0.1] + 5).bandwidth
    assert np.allclose(h2, h1 * [10, 1, 0.1], rtol=1e-6)


def test_psi2_estimate_converges():
    f = benchmark_densities()[0]
    rng = np.random.default_rng(3)
    X = f.sample(100000, rng)
    res = kded3(X)
    span = np.array([(g[-1] - g[0]) * len(g)/(len(g) - 1) for g in res.grid])
    exact = f.psi2() * np.outer(span**2, span**2) * np.prod(span)
    assert np.allclose(res.psi2, exact, rtol=0.25)


def test_evaluate_matches_grid_and_exact():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(3000, 3))
    res = kded3(X)
    pts = rng.normal(size=(50, 3)) * 0.8
    a = res.evaluate(pts)
    b = res.evaluate_exact(pts, X)
    assert np.allclose(a, b, rtol=0.03, atol=1e-4)


def test_rotate_gives_full_matrix_and_helps_correlated_data():
    f = benchmark_densities()[1]
    rng = np.random.default_rng(5)
    X = f.sample(5000, rng)
    diag = kded3(X)
    rot = kded3(X, rotate=True)
    assert abs(rot.H[0, 1]) > 0
    assert np.allclose(rot.H, rot.H.T)
    assert f.ise(X, rot.H) < f.ise(X, diag.H)
    pts = rng.normal(size=(20, 3)) * 0.5
    assert np.allclose(rot.evaluate(pts), rot.evaluate_exact(pts, X),
                       rtol=0.05, atol=1e-4)


def test_simple_and_linear_binning_agree():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(5000, 3))
    a = kded3(X, binning='simple').bandwidth
    b = kded3(X, binning='linear').bandwidth
    assert np.allclose(a, b, rtol=0.05)


def test_exact_ise_against_quadrature():
    f = GaussianMixture('t', [1], [[0, 0, 0]], [np.eye(3)])
    rng = np.random.default_rng(7)
    X = f.sample(200, rng)
    H = np.diag([0.3, 0.4, 0.5])**2
    g = np.linspace(-6, 6, 61)
    P = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    from kded import gaussian_kde_eval
    ise = ((gaussian_kde_eval(X, H, P) - f.pdf(P))**2).sum() * (g[1]-g[0])**3
    assert np.isclose(f.ise(X, H), ise, rtol=1e-3)


def test_arguments():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(500, 3))
    res = kded3(X, n=(32, 16, 64), limits=[5, (-4, None), None], refine=False)
    assert res.density.shape == (32, 16, 64)
    assert np.isclose(res.grid[0][0], -5 + 10/64)
    with pytest.raises(ValueError):
        kded3(X[:, :2])
    with pytest.raises(ValueError):
        kded3(np.r_[X, [[np.nan, 0, 0]]])
    with pytest.warns(UserWarning):
        kded3(X, limits=[1, 1, 1])


def test_adaptive_grid_matches_exact_and_normalised():
    from kded import kdedn_adaptive, adaptive_kde_eval
    f = benchmark_densities()[4]            # skewed
    rng = np.random.default_rng(9)
    X = f.sample(1500, rng)
    res = kdedn_adaptive(X, alpha=0.5, scale=1.0)
    dv = np.prod([g[1] - g[0] for g in res.grid])
    assert np.isclose(res.density.sum()*dv, 1, atol=1e-3)
    pts = f.sample(30, rng)
    exact = adaptive_kde_eval(X, res.H, res.lambdas, pts)
    assert np.allclose(res.evaluate(pts), exact, rtol=0.05, atol=1e-3)
    # Under-resolved narrow kernels trigger a warning.
    with pytest.warns(UserWarning):
        kdedn_adaptive(benchmark_densities()[5].sample(1500, rng),
                       alpha=0.5, scale=1.0, n=32)
    # LSCV selection runs and returns values from its grid.
    sel = kdedn_adaptive(X)
    assert sel.alpha in (0.0, 0.25, 0.5) and 0.5 <= sel.scale <= 2.5


def test_compat_kded1_kded2_match_kde_diffusion():
    """Drop-in kded1 / kded2 return the old shapes and values (bandwidths
    within the 0.1% fixed point difference; grid is cell centres)."""
    from kded import kded1, kded2
    r = np.load(REF/'reference1d.npz')
    d, g, h = kded1(r['x'], n=int(r['n']), binning='simple',
                    limits=(float(r['xmin']), float(r['xmax'])))
    assert d.shape == r['density'].shape and np.isscalar(h)
    assert np.isclose(h, r['bandwidth'], rtol=1e-10)
    assert np.allclose(g - (g[1] - g[0])/2, r['grid'])
    r = np.load(REF/'reference2d.npz')
    d, g, h = kded2(r['x'], r['y'], n=int(r['n']), binning='simple',
                    limits=((float(r['xmin']), float(r['xmax'])),
                            (float(r['ymin']), float(r['ymax']))))
    assert d.shape == r['density'].shape and len(g) == 2
    assert np.allclose(h, r['bandwidth'], rtol=1e-3)
    assert np.allclose(d, r['density'], rtol=0.01, atol=1e-3*d.max())
    # Scalar limits and a non power of two n, as in Grid1D/2D-cubic.py.
    rng = np.random.default_rng(0)
    d, g, h = kded1(rng.normal(size=500), n=1000, limits=12)
    assert len(g) == 1024 and np.isclose(g[0], -12 + 12/1024)
    d, g, h = kded2(rng.normal(size=500), rng.normal(size=500), n=64, limits=10)
    assert d.shape == (64, 64) and np.isclose(g[1][-1], 10 - 10/64)


def _five_modes(seed):
    """Botev et al. (2010), Table 1, case 14: (1/5) sum_k N(80k, (k+1)^2)."""
    rng = np.random.default_rng(seed)
    k = rng.choice(5, size=1000, p=np.full(5, 0.2))
    return rng.normal(80.0*k, k + 1.0)


def test_fixed_point_takes_smallest_root():
    """t - gamma(t) can change sign three times in (0, 0.1]; the bracket of
    kde.m / kde_diffusion then gives an oversmoothed h ~ 110 (all five modes
    merged). The smallest root gives h ~ 0.6, a fraction of the mode sd."""
    from kded import kded1
    for seed in ([2010, 14, 0, 0], [2010, 14, 0, 6]):
        h = kded1(_five_modes(seed))[2]
        assert 0.3 < h < 1.0
    # Default kdedn limits (range/4 margins): the bracket (0, 0.1] holds two
    # roots, which used to raise "did not converge".
    rng = np.random.default_rng([2010, 13, 0, 2])
    k = rng.choice(3, size=100, p=np.full(3, 1/3))
    x = rng.normal(80.0*k, (k + 1.0)**2)
    res = kdedn(x[:, None], n=2**14)
    assert 0.5 < res.bandwidth[0] < 3.0


def test_kded1_default_limits_match_kde_m():
    """kded1 pads the data range by 1/10 on each side, like kde.m."""
    from kded import kded1
    x = np.random.default_rng(3).normal(size=400)
    lo, hi = x.min() - (x.max() - x.min())/10, x.max() + (x.max() - x.min())/10
    d, g, h = kded1(x)
    d2, g2, h2 = kded1(x, limits=(lo, hi))
    assert np.allclose(g, g2) and h == h2


def test_resolution_check():
    """The default resolution check refines the grid until the selected
    bandwidth spans min_cells cells; refine=False keeps the grid."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20000, 3))
    fixed = kded3(X, n=32, refine=False)
    assert fixed.density.shape == (32, 32, 32)
    res = kded3(X, n=32)
    assert res.cells >= 2 and all(m >= 32 for m in res.density.shape)
    assert fixed.cells < res.cells


def test_default_grid():
    """Default initial grids; for d >= 5 the largest efficient m with
    m**d <= max_cells, in integer arithmetic."""
    from kded.core import _default_grid
    assert [_default_grid(d) for d in (1, 2, 3, 4)] == [2**14, 2**8, 2**7, 48]
    assert [_default_grid(d) for d in (5, 6, 7, 8)] == [27, 16, 10, 8]
    for d in range(5, 13):
        assert _default_grid(d)**d <= 2**24
    assert _default_grid(6, 2**24 - 1) == 15


def test_refinement_limits_and_warnings():
    """max_refine and max_cells bound the refinement, and the warning names
    the limit that was reached."""
    rng = np.random.default_rng(5)
    N = 20000
    X = np.where(rng.random((N, 1)) < 2/3, rng.normal(size=(N, 2)),
                 0.1*rng.normal(size=(N, 2)))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        res = kdedn(X, n=32)
    assert res.cells >= 2
    with pytest.warns(UserWarning, match='max_refine = 0'):
        r0 = kdedn(X, n=32, max_refine=0)
    assert r0.density.shape == (32, 32) and r0.cells < 2
    with pytest.warns(UserWarning, match='max_refine = 1'):
        r1 = kdedn(X, n=32, max_refine=1)
    assert min(r1.density.shape) > 32 and r1.cells < 2
    with pytest.warns(UserWarning, match='max_cells is reached'):
        rb = kdedn(X, n=32, max_cells=64**2)
    assert np.prod(rb.density.shape) <= 64**2 and rb.cells < 2
    with pytest.warns(UserWarning, match='refine=True'):
        kdedn(X, n=32, refine=False)
