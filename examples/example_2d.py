"""Two-dimensional data: Old Faithful, and a mixture with correlated components.

Run from the repository root:  python examples/example_2d.py
Writes docs/images/example_2d.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from kded import kded2, kdedn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'docs', 'images', 'example_2d.png')
rng = np.random.default_rng(0)

# --- (a) Old Faithful: eruption time and waiting time (minutes) ---------------
F = np.loadtxt(os.path.join(HERE, 'data', 'faithful.csv'), delimiter=',', skiprows=1)
# spread the rounded values over their rounding intervals (1 s and 1 min)
F = F + rng.uniform(-0.5, 0.5, F.shape) * [1 / 60, 1]
# kded2 follows the calling convention of Botev's kde2d.m
density, (gx, gy), h = kded2(F[:, 0], F[:, 1])
print('faithful: bandwidths (eruption, waiting) =', np.round(h, 3))

# --- (b, c) two anti-correlated components -------------------------------------
means = [(-1.0, -1.0), (1.0, 1.0)]
cov = np.array([[0.36, -0.29], [-0.29, 0.36]])          # sd 0.6, correlation -0.8
N = 5000
k = rng.random(N) < 0.5
X = np.where(k[:, None], rng.multivariate_normal(means[0], cov, N),
             rng.multivariate_normal(means[1], cov, N))

diag = kdedn(X)                  # diagonal bandwidth matrix
rot = kdedn(X, rotate=True)      # bandwidth matrix in the principal axes: full H
print('diagonal H:\n', np.round(diag.H, 4))
print('rotate=True H:\n', np.round(rot.H, 4))

t = np.linspace(-3.2, 3.2, 200)
TX, TY = np.meshgrid(t, t, indexing='ij')
P = np.c_[TX.ravel(), TY.ravel()]
true = (0.5 * mvn(means[0], cov).pdf(P) + 0.5 * mvn(means[1], cov).pdf(P)).reshape(TX.shape)
levels = np.linspace(0.03, 0.27, 5)

fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
lv = np.linspace(0, density.max(), 15)[1:]      # tiny negative ripples of the DCT are left blank
ax[0].contourf(gx, gy, density.T, lv, cmap='Blues', extend='max')
ax[0].plot(F[:, 0], F[:, 1], '.', color='0.25', ms=2.5)
ax[0].set(xlim=(1.3, 5.5), ylim=(38, 100), xlabel='eruption time (min)',
          ylabel='waiting time (min)', title=f'(a) Old Faithful, N = {len(F)}')

for a, res, name in ((ax[1], diag, 'kded, diagonal H'), (ax[2], rot, 'kded, rotate=True')):
    est = res.evaluate(P).reshape(TX.shape)
    a.contour(TX, TY, true, levels, colors='0.55', linewidths=1.0)
    a.contour(TX, TY, est, levels, colors='C0', linewidths=1.3)
    a.plot([], [], color='0.55', label='true density')
    a.plot([], [], color='C0', label=name)
    a.set(xlim=(-3.6, 3.2), ylim=(-3.2, 3.6), xlabel='x', aspect='equal')
    a.legend(frameon=False, fontsize=8, loc='upper left', handlelength=1.2)
ax[1].set_title(f'(b) correlated mixture, N = {N}')
ax[2].set_title('(c) same data, full bandwidth matrix')
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print('saved', os.path.normpath(OUT))
