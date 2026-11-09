"""Bandwidth selectors of the benchmark, called through the public API of each
library. Each selector maps a training sample X (N x d) to a dict with the
bandwidth matrix 'H' (kernel covariance, d x d) and, for the adaptive
estimator, the per-point factors 'lambdas' (point i has covariance
lambdas[i]^2 H). Every estimator is then scored as the exact Gaussian KDE
with this H, so the benchmark compares bandwidth selectors, not grids or
interpolation.
"""
import numpy as np


# ---------------------------------------------------------------- kded
def kded_default(X):
    from kded import kdedn
    return {'H': kdedn(X).H}


def kded_rotate(X):
    from kded import kdedn
    return {'H': kdedn(X, rotate=True).H}


def kded_adaptive(X):
    from kded import kdedn_adaptive
    r = kdedn_adaptive(X)
    return {'H': r.H, 'lambdas': r.lambdas}


# ---------------------------------------------------------------- scipy
def scipy_scott(X):
    from scipy.stats import gaussian_kde
    return {'H': np.atleast_2d(gaussian_kde(X.T, 'scott').covariance)}


def scipy_silverman(X):
    from scipy.stats import gaussian_kde
    return {'H': np.atleast_2d(gaussian_kde(X.T, 'silverman').covariance)}


# ---------------------------------------------------------------- statsmodels
def statsmodels_nr(X):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    k = KDEMultivariate(X, var_type='c' * X.shape[1], bw='normal_reference')
    return {'H': np.diag(np.asarray(k.bw, dtype=float) ** 2)}


def statsmodels_cv_ml(X):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    k = KDEMultivariate(X, var_type='c' * X.shape[1], bw='cv_ml')
    return {'H': np.diag(np.asarray(k.bw, dtype=float) ** 2)}


# ---------------------------------------------------------------- scikit-learn
def sklearn_cv(X):
    """The usual scikit-learn recipe: standardize each axis, then choose one
    bandwidth by 5-fold cross-validated log-likelihood (GridSearchCV) over 40
    values from 0.05 to 2 times Scott's factor."""
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.neighbors import KernelDensity
    n, d = X.shape
    sd = X.std(axis=0, ddof=1)
    Z = X / sd
    scott = n ** (-1 / (d + 4))
    grid = {'bandwidth': scott * np.logspace(np.log10(0.05), np.log10(2), 40)}
    gs = GridSearchCV(KernelDensity(kernel='gaussian'), grid,
                      cv=KFold(5, shuffle=True, random_state=0))
    gs.fit(Z)
    b = gs.best_params_['bandwidth']
    return {'H': np.diag((b * sd) ** 2)}


# ---------------------------------------------------------------- KDEpy (d = 1)
def kdepy_isj(X):
    from KDEpy import FFTKDE
    return {'H': np.array([[FFTKDE(bw='ISJ').fit(X[:, 0]).bw ** 2]])}


def kdepy_silverman(X):
    from KDEpy import FFTKDE
    return {'H': np.array([[FFTKDE(bw='silverman').fit(X[:, 0]).bw ** 2]])}


# name: (function, dimensions, label in the report)
METHODS = {
    'kded':             (kded_default,     (1, 2, 3, 4), 'kded'),
    'kded_rot':         (kded_rotate,      (2, 3, 4), 'kded, rotate=True'),
    'kded_adaptive':    (kded_adaptive,    (1, 2, 3, 4), 'kded adaptive (experimental)'),
    'scipy_scott':      (scipy_scott,      (1, 2, 3, 4), 'scipy gaussian_kde (Scott)'),
    'scipy_silverman':  (scipy_silverman,  (1, 2, 3, 4), 'scipy gaussian_kde (Silverman)'),
    'sm_nr':            (statsmodels_nr,   (1, 2, 3, 4), 'statsmodels normal_reference'),
    'sm_cv_ml':         (statsmodels_cv_ml, (1, 2, 3, 4), 'statsmodels cv_ml'),
    'sklearn_cv':       (sklearn_cv,       (1, 2, 3, 4), 'scikit-learn GridSearchCV'),
    'kdepy_isj':        (kdepy_isj,        (1,),      'KDEpy ISJ'),
    'kdepy_silverman':  (kdepy_silverman,  (1,),      'KDEpy Silverman'),
}

# Selectors of the R package ks, run by ks_selectors.R (labels for the report).
KS_METHODS = {
    'ks_hpi':   'R ks Hpi (plug-in)',
    'ks_hscv':  'R ks Hscv (smoothed CV)',
    'ks_hlscv': 'R ks Hlscv (least squares CV)',
    'r_bwsj':   'R bw.SJ (Sheather-Jones)',
}
