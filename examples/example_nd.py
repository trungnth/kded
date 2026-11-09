"""Data in more than three dimensions: the four body measurements of the
Palmer penguins, estimated jointly with kdedn and shown as a corner plot of the
one- and two-dimensional marginals of the 4-d estimate.

Run from the repository root:  python examples/example_nd.py
Writes docs/images/example_nd.png.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kded import kdedn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'docs', 'images', 'example_nd.png')

cols = ['bill length (mm)', 'bill depth (mm)', 'flipper length (mm)', 'body mass (g)']
X = np.loadtxt(os.path.join(HERE, 'data', 'penguins.csv'), delimiter=',',
               skiprows=1, usecols=(1, 2, 3, 4))
# spread the rounded values over their rounding intervals (0.1 mm, 1 mm, 25 g)
rng = np.random.default_rng(0)
X = X + rng.uniform(-0.5, 0.5, X.shape) * [0.1, 0.1, 1, 25]
N, d = X.shape

t0 = time.perf_counter()
res = kdedn(X)                 # default grid for d = 4: 48^4 points
print(f'kdedn: N = {N}, d = {d}, {time.perf_counter() - t0:.1f} s, grid {res.density.shape}')
print('bandwidths:', np.round(res.bandwidth, 3))

# density of a new bird (exact Gaussian KDE with the selected H)
new = [[45.0, 15.0, 215.0, 5000.0]]
print('density at', new[0], '=', res.evaluate_exact(new, X)[0])

# marginals of the gridded 4-d estimate: integrate over the other axes
cell = [g[1] - g[0] for g in res.grid]
f = np.clip(res.density, 0, None)

fig, ax = plt.subplots(d, d, figsize=(9.5, 9.5))
for i in range(d):
    for j in range(d):
        a = ax[i, j]
        if j > i:
            a.axis('off')
            continue
        if i == j:
            other = tuple(k for k in range(d) if k != i)
            m = f.sum(axis=other) * np.prod([cell[k] for k in other])
            a.plot(res.grid[i], m, color='C0')
            a.hist(X[:, i], bins=25, density=True, color='0.85', edgecolor='white')
            a.set_yticks([])
        else:
            other = tuple(k for k in range(d) if k not in (i, j))
            m = f.sum(axis=other) * np.prod([cell[k] for k in other])   # axes (j, i)
            lv = np.linspace(0, m.max(), 9)[1:]
            a.contourf(res.grid[j], res.grid[i], m.T, lv, cmap='Blues', extend='max')
            a.plot(X[:, j], X[:, i], '.', color='0.2', ms=1.5)
        lo, hi = np.percentile(X[:, j], [0, 100])
        pad = 0.1 * (hi - lo)
        a.set_xlim(lo - pad, hi + pad)
        if i != j:
            lo, hi = np.percentile(X[:, i], [0, 100])
            pad = 0.1 * (hi - lo)
            a.set_ylim(lo - pad, hi + pad)
        if i == d - 1:
            a.set_xlabel(cols[j])
        else:
            a.set_xticklabels([])
        if j == 0 and i > 0:
            a.set_ylabel(cols[i])
        elif j > 0:
            a.set_yticklabels([])
fig.suptitle(f'Palmer penguins: 4-d kded estimate, N = {N} (marginals)', y=0.995)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print('saved', os.path.normpath(OUT))
