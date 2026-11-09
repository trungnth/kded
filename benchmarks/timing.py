"""Computing time: automatic bandwidth selection plus the density on a grid.

Data: N points from a two-component normal mixture in d = 1, 2, 3 dimensions.
Grid: 1024 points (d = 1), 128 x 128 (d = 2), 64 x 64 x 64 (d = 3).
Every measurement runs in its own process on one thread, with a time limit;
the time is the wall clock of selection plus evaluation on the grid, the
median of `--repeat` runs (one run when it takes longer than `--long` s).

  kded            kdedn(X, n=grid)              (selection + grid density)
  scipy           gaussian_kde(X.T) (Scott) evaluated on the grid
  statsmodels     KDEMultivariate(bw='normal_reference').pdf(grid)
  statsmodels CV  KDEMultivariate(bw='cv_ml') (selection only)
  scikit-learn    KernelDensity(Scott bandwidth on standardized data),
                  score_samples(grid) (no cross-validation)
  KDEpy           FFTKDE(bw='ISJ') for d = 1, FFTKDE(Scott, standardized
                  data) for d > 1, evaluated on the grid

Usage: python timing.py [--N 1000,10000,100000,1000000] [--dims 1,2,3]
                        [--repeat 3] [--long 60] [--timeout 600]
                        [--out results/timing/timing.jsonl]
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(v, '1')

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np

GRID = {1: 1024, 2: 128, 3: 64}


def sample(N, d, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.random(N) < 0.5
    X = rng.normal(size=(N, d))
    X[k] = X[k] * 0.5 + 2.0
    return X


def _grid_points(X, m):
    lo, hi = X.min(0), X.max(0)
    axes = [np.linspace(a, b, m) for a, b in zip(lo, hi)]
    G = np.meshgrid(*axes, indexing='ij')
    return np.stack([g.ravel() for g in G], axis=1)


def t_kded(X):
    from kded import kdedn
    r = kdedn(X, n=GRID[X.shape[1]])
    return r.density


def t_scipy(X):
    from scipy.stats import gaussian_kde
    k = gaussian_kde(X.T)
    return k(_grid_points(X, GRID[X.shape[1]]).T)


def t_sm_nr(X):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    k = KDEMultivariate(X, var_type='c' * X.shape[1], bw='normal_reference')
    return k.pdf(_grid_points(X, GRID[X.shape[1]]))


def t_sm_cv(X):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    return KDEMultivariate(X, var_type='c' * X.shape[1], bw='cv_ml').bw


def t_sklearn(X):
    from sklearn.neighbors import KernelDensity
    n, d = X.shape
    sd = X.std(0, ddof=1)
    k = KernelDensity(bandwidth=n ** (-1 / (d + 4))).fit(X / sd)
    return np.exp(k.score_samples(_grid_points(X, GRID[d]) / sd))


def t_kdepy(X):
    from KDEpy import FFTKDE
    n, d = X.shape
    if d == 1:
        return FFTKDE(bw='ISJ').fit(X[:, 0]).evaluate(GRID[1])
    sd = X.std(0, ddof=1)
    return FFTKDE(bw=n ** (-1 / (d + 4))).fit(X / sd).evaluate(GRID[d])


METHODS = {
    'kded': (t_kded, 'kded (ISJ, diffusion)'),
    'kdepy': (t_kdepy, 'KDEpy FFTKDE (ISJ for d = 1, Scott for d > 1)'),
    'scipy': (t_scipy, 'scipy gaussian_kde (Scott)'),
    'sklearn': (t_sklearn, 'scikit-learn KernelDensity (Scott, no CV)'),
    'sm_nr': (t_sm_nr, 'statsmodels normal_reference'),
    'sm_cv_ml': (t_sm_cv, 'statsmodels cv_ml (selection only)'),
}


def _work(m, N, d, rep, q):
    X = sample(N, d, seed=rep)
    t0 = time.perf_counter()
    METHODS[m][0](X)
    q.put(time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', default='1000,10000,100000,1000000')
    ap.add_argument('--dims', default='1,2,3')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--long', type=float, default=60)
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'results', 'timing', 'timing.jsonl'))
    a = ap.parse_args()
    import warnings
    warnings.filterwarnings('ignore')
    import kded, scipy.stats, sklearn.neighbors, KDEpy  # noqa: F401,E401
    import statsmodels.nonparametric.kernel_density    # noqa: F401
    ctx = mp.get_context('fork')
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r['method'], r['d'], r['N']))
    fh = open(a.out, 'a')
    for d in map(int, a.dims.split(',')):
        for m in a.methods.split(','):
            for N in map(int, a.N.split(',')):
                if (m, d, N) in done:
                    continue
                times, status = [], 'ok'
                for rep in range(a.repeat):
                    q = ctx.Queue()
                    p = ctx.Process(target=_work, args=(m, N, d, rep, q))
                    t0 = time.time()
                    p.start()
                    while p.is_alive() and q.empty() and time.time() - t0 < a.timeout:
                        time.sleep(0.02)
                    if not q.empty():
                        times.append(q.get())
                        p.join()
                        if times[-1] > a.long:
                            break          # long runs are not repeated
                    else:
                        p.kill()
                        p.join()
                        status = 'timeout' if time.time() - t0 >= a.timeout else 'crashed'
                        break
                rec = {'method': m, 'd': d, 'N': N, 'status': status,
                       'time': float(np.median(times)) if times else None,
                       'times': times, 'timeout': a.timeout}
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
                print(json.dumps(rec), flush=True)
                if status != 'ok':
                    break          # larger N would also time out


if __name__ == '__main__':
    main()
