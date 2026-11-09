# Changelog

## 0.3.0 (2026-11-06)

- Renamed the package `kde3d` to `kded` (module `kde_diffusion_nd` to
  `kded`) and the functions `kde_nd`, `kde1d`, `kde2d`, `kde3d`,
  `kde3d_adaptive` to `kdedn`, `kded1`, `kded2`, `kded3`, `kdedn_adaptive`.
  For the same input the results are identical, bit for bit, to 0.2.2.
- The default grid of `kdedn_adaptive` is now that of `kdedn` for every
  dimension (unchanged for d = 3).

## 0.2.2

- Default initial grid for d ≥ 5 computed in integer arithmetic (16 instead
  of 15 points per axis for d = 6); more tests.

## 0.2.1

- Number of refinements of the resolution check is an argument
  (`max_refine`, default 3); clearer warning.

## 0.2.0

- Resolution check: the grid is refined until the selected kernel spans at
  least `min_cells` cells, within a budget of `max_cells` cells.
- Default initial grid for d ≥ 4.

## 0.1.1

- The fixed point of the ISJ equation is the smallest root, found by an
  upward scan. The bracketing solvers of `kde.m` (2010) and `KDE-diffusion`
  1.0.5 can return a spurious larger root (case 14 of the 1-D benchmark of
  Botev et al. 2010).

## 0.1.0

- First version: diffusion KDE with the improved Sheather–Jones bandwidth in
  any dimension, interfaces of `kde.m` and `kde2d.m`, rotation, adaptive
  estimator (experimental).
