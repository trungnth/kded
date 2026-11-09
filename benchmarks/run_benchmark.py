"""Held-out validation of kded and other KDE bandwidth selectors on open data.

For every data set (datasets.py), split (5 random halvings, both halves) and
selector (bw_selectors.py), the selector runs on the training half and the
Gaussian KDE with the selected bandwidth matrix is scored on the test half:

  log-likelihood   mean of log f(Y_j) over the test points (higher is better)
  L2 score         int f^2 - 2 mean f(Y_j), an unbiased estimate of
                   ISE(f) - int p^2 (lower is better)

Each fit runs in its own process with one thread and a time limit; the
selection time (wall clock) is recorded. Results are appended to a JSON-lines
file, so an interrupted run resumes where it stopped.

Usage
  python run_benchmark.py [--datasets a,b] [--methods m1,m2] [--timeout 600]
                          [--jobs 4] [--out results/local/python.jsonl]
  python run_benchmark.py --export-splits splits/   # for ks_selectors.R
  python run_benchmark.py --score-ks results/local/ks_H.csv --out results/local/ks.jsonl
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
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datasets as ds                         # noqa: E402
from bw_selectors import METHODS             # noqa: E402


# ------------------------------------------------------------------ scores
def log_kde(A, H, Y, lam=None, chunk=2**22):
    """log f(Y) for the Gaussian KDE with covariance lam_i^2 H at A_i."""
    n, d = A.shape
    L = np.linalg.cholesky(np.linalg.inv(H))
    Ad, Yd = A @ L, Y @ L
    lam = np.ones(n) if lam is None else np.asarray(lam, dtype=float)
    c = (-0.5 * np.log((2 * np.pi) ** d * np.linalg.det(H))
         - d * np.log(lam) - np.log(n))
    sa = (Ad ** 2).sum(1)
    out = np.empty(len(Y))
    step = max(1, chunk // n)
    for i in range(0, len(Y), step):
        P = Yd[i:i + step]
        D2 = np.maximum((P ** 2).sum(1)[:, None] + sa[None, :] - 2 * P @ Ad.T, 0)
        out[i:i + step] = logsumexp(-0.5 * D2 / lam[None, :] ** 2 + c[None, :], axis=1)
    return out


def int_f2(A, H, lam=None, chunk=2**22):
    """int f^2 = mean_ik phi_{(lam_i^2 + lam_k^2) H}(A_i - A_k)."""
    n, d = A.shape
    L = np.linalg.cholesky(np.linalg.inv(H))
    Ad = A @ L
    lam = np.ones(n) if lam is None else np.asarray(lam, dtype=float)
    sa = (Ad ** 2).sum(1)
    tot = 0.0
    step = max(1, chunk // n)
    for i in range(0, n, step):
        P = Ad[i:i + step]
        D2 = np.maximum((P ** 2).sum(1)[:, None] + sa[None, :] - 2 * P @ Ad.T, 0)
        v = lam[i:i + step, None] ** 2 + lam[None, :] ** 2
        tot += (np.exp(-0.5 * D2 / v) / v ** (d / 2)).sum()
    return tot / (n ** 2 * (2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(H)))


def score(train, test, H, lam=None):
    H = np.atleast_2d(np.asarray(H, dtype=float))
    if not np.all(np.isfinite(H)) or np.linalg.eigvalsh(H).min() <= 0:
        return {'status': 'invalid H'}
    lf = log_kde(train, H, test, lam)
    f = np.exp(lf)
    return {'loglik': float(lf.mean()),
            'l2': float(int_f2(train, H, lam) - 2 * f.mean())}


# ------------------------------------------------------------------ worker
def _work(method, key, train, test, q):
    try:
        fn = METHODS[method][0]
        t0 = time.perf_counter()
        sel = fn(train)
        dt = time.perf_counter() - t0
        out = {'time': dt, 'H': np.atleast_2d(sel['H']).tolist()}
        out.update(score(train, test, sel['H'], sel.get('lambdas')))
        out.setdefault('status', 'ok')
    except Exception as e:                       # noqa: BLE001
        out = {'status': f'error: {type(e).__name__}: {e}'[:300]}
    q.put(out)


def _tasks(names, methods, done):
    for name in names:
        d = ds.DATASETS[name][0]
        for key, train, test in ds.splits(name):
            for m in methods:
                if d in METHODS[m][1] and (key, m) not in done:
                    yield name, key, m, train, test


def _preload():
    """Imports the libraries before forking, so that import time is not
    counted as selection time."""
    import warnings
    warnings.filterwarnings('ignore')
    for mod in ('kded', 'scipy.stats', 'scipy.fft', 'sklearn.neighbors',
                'sklearn.model_selection',
                'statsmodels.nonparametric.kernel_density', 'KDEpy'):
        try:
            __import__(mod)
        except ImportError:
            print(f'warning: cannot import {mod}', flush=True)


def run(names, methods, out, timeout, jobs):
    _preload()
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r['key'], r['method']))
    tasks = list(_tasks(names, methods, done))
    # A selector that timed out on one split of a data set is skipped on the
    # other splits of that data set.
    timed_out = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            if r.get('status') == 'timeout':
                timed_out.add((r['dataset'], r['method']))
    print(f'{len(tasks)} fits to run, {len(done)} already in {out}', flush=True)
    ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
    running = []
    fh = open(out, 'a')

    def finish(entry, res):
        (name, key, m, n_train), _, _, _ = entry
        rec = {'dataset': name, 'key': key, 'method': m,
               'd': ds.DATASETS[name][0], 'n_train': n_train}
        rec.update(res)
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        print(f"{key:24s} {m:16s} {rec.get('status'):10s} "
              f"t={rec.get('time', float('nan')):8.2f}s "
              f"loglik={rec.get('loglik', float('nan')):9.4f} "
              f"l2={rec.get('l2', float('nan')):.4g}", flush=True)
        if rec['status'] == 'timeout':
            timed_out.add((name, m))

    while tasks or running:
        while tasks and len(running) < jobs:
            name, key, m, train, test = tasks.pop(0)
            if (name, m) in timed_out:
                finish(((name, key, m, len(train)), None, None, None),
                       {'status': 'skipped (timeout on another split)'})
                continue
            q = ctx.Queue()
            p = ctx.Process(target=_work, args=(m, key, train, test, q))
            p.start()
            running.append(((name, key, m, len(train)), p, q, time.time()))
        for entry in list(running):
            _, p, q, t0 = entry
            if not q.empty():
                res = q.get()
                p.join()
                running.remove(entry)
                finish(entry, res)
            elif not p.is_alive():
                running.remove(entry)
                finish(entry, {'status': f'crashed (exit code {p.exitcode})'})
            elif time.time() - t0 > timeout:
                p.kill()
                p.join()
                running.remove(entry)
                finish(entry, {'status': 'timeout', 'time': timeout})
        time.sleep(0.05)
    fh.close()


# ------------------------------------------------------------------ R ks
def export_splits(folder):
    """Writes every training half as CSV for ks_selectors.R."""
    os.makedirs(folder, exist_ok=True)
    keys = []
    for name in ds.available():
        for key, train, _ in ds.splits(name):
            f = key.replace('/', '__') + '.csv'
            np.savetxt(os.path.join(folder, f), train, delimiter=',', fmt='%.17g')
            keys.append(f)
    open(os.path.join(folder, 'keys.txt'), 'w').write('\n'.join(keys) + '\n')
    print(f'wrote {len(keys)} training halves to {folder}')


def score_ks(hfile, out):
    """Scores the bandwidth matrices written by ks_selectors.R."""
    import csv
    test_of = {}
    for name in ds.available():
        for key, train, test in ds.splits(name):
            test_of[key] = (name, train, test)
    with open(hfile) as f, open(out, 'w') as fh:
        for row in csv.reader(f):
            key, m, t, status = row[0].replace('__', '/'), row[1], float(row[2]), row[3]
            if key not in test_of:
                print(f'skipping {key}: unknown data set', flush=True)
                continue
            name, train, test = test_of[key]
            d = train.shape[1]
            rec = {'dataset': name, 'key': key, 'method': m, 'd': d,
                   'n_train': len(train), 'time': t}
            if status != 'ok':
                rec['status'] = status
            else:
                H = np.array(row[4:4 + d * d], dtype=float).reshape(d, d)
                rec['H'] = H.tolist()
                rec.update(score(train, test, H))
                rec.setdefault('status', 'ok')
            fh.write(json.dumps(rec) + '\n')
            print(key, m, rec['status'], rec.get('loglik'), rec.get('l2'), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', default=None)
    ap.add_argument('--methods', default=None)
    ap.add_argument('--timeout', type=float, default=600.0)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--out', default=os.path.join(ds.HERE, 'results', 'local', 'python.jsonl'))
    ap.add_argument('--export-splits', default=None)
    ap.add_argument('--score-ks', default=None)
    a = ap.parse_args()
    if a.export_splits:
        export_splits(a.export_splits)
    elif a.score_ks:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        score_ks(a.score_ks, a.out)
    else:
        names = a.datasets.split(',') if a.datasets else ds.available()
        methods = a.methods.split(',') if a.methods else list(METHODS)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        run(names, methods, a.out, a.timeout, a.jobs)
