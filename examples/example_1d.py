"""One-dimensional data: a multimodal density and the Old Faithful waiting times.

Run from the repository root:  python examples/example_1d.py
Writes docs/images/example_1d.png.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

from kded import kded1, kdedn

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'docs', 'images', 'example_1d.png')

# --- (a) Marron-Wand "claw": a normal density with five narrow spikes -------
rng = np.random.default_rng(1)
N = 2000
comp = rng.choice(6, size=N, p=[0.5] + [0.1] * 5)
mu = np.r_[0.0, np.arange(5) / 2 - 1]
sd = np.r_[1.0, [0.1] * 5]
x = rng.normal(mu[comp], sd[comp])


def claw(t):
    return 0.5 * norm.pdf(t, 0, 1) + sum(0.1 * norm.pdf(t, k / 2 - 1, 0.1) for k in range(5))


# kded1 follows the calling convention of Botev's kde.m: (density, grid, bandwidth)
density, grid, h = kded1(x)
print(f'claw:      kded bandwidth h = {h:.4f}')

scott = gaussian_kde(x)                          # scipy default (Scott's rule)
print(f'claw:      scipy (Scott) bandwidth h = {np.sqrt(scott.covariance[0, 0]):.4f}')

# --- (b) Old Faithful waiting times between eruptions (minutes) --------------
w = np.loadtxt(os.path.join(HERE, 'data', 'faithful.csv'), delimiter=',',
               skiprows=1, usecols=1)
# The times are rounded to whole minutes (51 distinct values in 272 points).
# Ties look like point masses to any plug-in selector, and the ISJ bandwidth
# collapses to the rounding scale. Spread each value uniformly over its
# rounding interval (+-0.5 min) first:
w = w + np.random.default_rng(0).uniform(-0.5, 0.5, w.size)
res = kdedn(w)                                   # general interface, N x d or 1-d array
print(f'faithful:  kded bandwidth h = {res.bandwidth[0]:.3f} min')

# --- figure ------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
t = np.linspace(-3, 3, 1000)
ax[0].plot(t, claw(t), color='0.2', lw=2.2, label='true density')
m = (grid > -3) & (grid < 3)
ax[0].plot(grid[m], density[m], color='C0', lw=1.6, label=f'kded (h = {h:.3f})')
ax[0].plot(t, scott(t), color='C3', lw=1.4, ls='--',
           label=f'scipy, Scott (h = {np.sqrt(scott.covariance[0, 0]):.3f})')
ax[0].set_title(f'(a) Claw density, N = {N}')
ax[0].set_xlabel('x')
ax[0].set_ylabel('density')
ax[0].set_ylim(-0.02, 0.8)
ax[0].legend(frameon=False, fontsize=9, loc='upper left')

g = res.grid[0]
m = (g > 35) & (g < 105)
ax[1].hist(w, bins=30, density=True, color='0.85', edgecolor='white', label='histogram')
ax[1].plot(g[m], res.density[m], color='C0', lw=1.8,
           label=f'kded (h = {res.bandwidth[0]:.2f} min)')
ax[1].plot(g[m], gaussian_kde(w)(g[m]), color='C3', lw=1.4, ls='--', label='scipy, Scott')
ax[1].plot(w, np.full_like(w, -0.001), '|', color='0.3', ms=6)
ax[1].set_title(f'(b) Old Faithful waiting time, N = {len(w)}')
ax[1].set_xlabel('minutes')
ax[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print('saved', os.path.normpath(OUT))
