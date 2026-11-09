"""Three-dimensional data: three clusters of different sizes, one of them tilted.

Run from the repository root:  python examples/example_3d.py
Writes docs/images/example_3d.png.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from kded import kded3, kdedn_adaptive, adaptive_kde_eval
from kded.testdensities import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'docs', 'images', 'example_3d.png')

# A mixture of three normal clusters with known density (for comparison)
f = GaussianMixture('three clusters', weights=[0.45, 0.35, 0.20],
                    means=[[0, 0, 0], [3, 0, 0], [1.5, 2.5, 1]],
                    covs=[np.diag([0.6, 0.4, 0.5]) ** 2,
                          [[0.36, 0.25, 0], [0.25, 0.36, 0], [0, 0, 0.16]],
                          np.diag([0.25, 0.25, 0.25]) ** 2])
rng = np.random.default_rng(0)
N = 10_000
X = f.sample(N, rng)                  # N x 3 array

t0 = time.perf_counter()
res = kded3(X)                        # same as kdedn(X) for 3-column data
print(f'kded3: {time.perf_counter() - t0:.2f} s, grid {res.density.shape}, '
      f'bandwidths {np.round(res.bandwidth, 3)}, {res.cells:.1f} cells per bandwidth')

# Values of the estimate at arbitrary points: interpolated from the grid, or
# exact (sum over the data with the selected bandwidth matrix H).
pts = np.array([[0, 0, 0], [3, 0, 0], [1.5, 2.5, 1]])
print('true      :', np.round(f.pdf(pts), 5))
print('grid      :', np.round(res.evaluate(pts), 5))
print('exact     :', np.round(res.evaluate_exact(pts, X), 5))

# Experimental adaptive estimator (Abramson's square-root law, parameters by
# least squares cross-validation): narrower kernels in the dense clusters.
ad = kdedn_adaptive(X, pilot=res)
print(f'adaptive  : alpha = {ad.alpha:.2f}, scale = {ad.scale:.2f}')

scott = gaussian_kde(X.T)             # scipy default for comparison

# --- figure --------------------------------------------------------------------
fig = plt.figure(figsize=(14, 4.4))
a0 = fig.add_subplot(1, 3, 1, projection='3d')
idx = rng.choice(N, 3000, replace=False)
dens = res.evaluate(X[idx])
o = np.argsort(dens)
sc = a0.scatter(*X[idx][o].T, c=np.log10(dens[o]), s=3, cmap='viridis')
a0.set(xlabel='x', ylabel='y', zlabel='z', title='(a) sample, colored by log10 kded3 density')
a0.view_init(22, -60)

# (b) marginal density of (x, y): integrate the grid estimate over z
(gx, gy, gz) = res.grid
dz = gz[1] - gz[0]
est = res.density.sum(axis=2) * dz
GX, GY = np.meshgrid(gx, gy, indexing='ij')
a1 = fig.add_subplot(1, 3, 2)
lv = np.linspace(0, est.max(), 12)[1:]
a1.contourf(gx, gy, est.T, lv, cmap='Blues', extend='max')
# true (x, y) marginal of the mixture
from scipy.stats import multivariate_normal as mvn
PXY = np.dstack([GX, GY])
true_xy = sum(w * mvn(m[:2], C[:2, :2]).pdf(PXY) for w, m, C in zip(f.w, f.mu, f.S))
a1.contour(gx, gy, true_xy.T, lv[::2], colors='k', linewidths=0.7)
a1.set(xlim=(-2, 5), ylim=(-2, 4), xlabel='x', ylabel='y',
       title='(b) (x, y) marginal: kded3 (fill), truth (lines)')
a1.set_aspect('equal')

# (c) profile along x through the centre of the narrow cluster (y = 2.5, z = 1)
x = np.linspace(0.3, 2.7, 400)
P = np.c_[x, np.full_like(x, 2.5), np.ones_like(x)]
a2 = fig.add_subplot(1, 3, 3)
a2.plot(x, f.pdf(P), color='0.2', lw=2.2, label='true density')
a2.plot(x, res.evaluate(P), color='C0', lw=1.6, label='kded3')
a2.plot(x, adaptive_kde_eval(X, ad.H, ad.lambdas, P), color='C2', lw=1.4,
        label='kded adaptive (experimental)')
a2.plot(x, scott(P.T), color='C3', lw=1.4, ls='--', label='scipy, Scott')
a2.set(xlabel='x  (y = 2.5, z = 1)', ylabel='density',
       title='(c) profile through the narrowest cluster')
a2.set_ylim(-0.02, 1.05)
a2.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print('saved', os.path.normpath(OUT))
