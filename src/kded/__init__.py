"""kded: kernel density estimation via diffusion for data in any dimension.

    from kded import kdedn          # d-dimensional data, an N x d array
    from kded import kded1, kded2   # one- and two-dimensional data, with the
                                    # calling conventions of kde.m and kde2d.m
    from kded import kded3          # three-dimensional data, an N x 3 array

Up to version 0.2.2 the package was named kde3d (module kde_diffusion_nd).
Version 0.3.0 renames the package and its functions; the computations are
unchanged.
"""
from .core import (kdedn, kded3, kdedn_adaptive, KDEResult, amise_diagonal,
                   gaussian_kde_eval, adaptive_kde_eval)
from .compat import kded1, kded2

__version__ = '0.3.0'

__all__ = ['kdedn', 'kded1', 'kded2', 'kded3', 'kdedn_adaptive', 'KDEResult',
           'amise_diagonal', 'gaussian_kde_eval', 'adaptive_kde_eval']
