"""Tables and figures of the held-out benchmark and of the timing study.

Usage: python report.py [--scores results/cluster] [--times results/local]
                        [--timing results/timing] [--figures ../docs/images]

The held-out scores are read from the python*.jsonl and ks.jsonl files of
--scores, the selection times from those of --times (the same benchmark run
on another machine; the --scores folder itself if --times holds no results),
and the timing study from the timing*.jsonl files of --timing. Writes
summary.md, summary.csv and readme_table.md to the results folder, and
validation_heldout.png and validation_timing.png to --figures.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from datasets import DATASETS             # noqa: E402
from bw_selectors import METHODS, KS_METHODS   # noqa: E402

LABEL = {k: v[2] for k, v in METHODS.items()}
LABEL.update(KS_METHODS)
ORDER = ['kded', 'kded_rot', 'kded_adaptive', 'ks_hpi', 'ks_hscv', 'ks_hlscv',
         'r_bwsj', 'sm_cv_ml', 'sklearn_cv', 'kdepy_isj', 'scipy_scott',
         'scipy_silverman', 'sm_nr', 'kdepy_silverman']
SHORT = {'kded': 'kded', 'kded_rot': 'kded rot', 'kded_adaptive': 'kded adapt.',
         'ks_hpi': 'ks Hpi', 'ks_hscv': 'ks Hscv', 'ks_hlscv': 'ks Hlscv',
         'r_bwsj': 'R bw.SJ', 'sm_cv_ml': 'sm cv_ml', 'sklearn_cv': 'sklearn CV',
         'kdepy_isj': 'KDEpy ISJ', 'scipy_scott': 'scipy Scott',
         'scipy_silverman': 'scipy Silv.', 'sm_nr': 'sm NR',
         'kdepy_silverman': 'KDEpy Silv.'}


def load(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, 'python*.jsonl')) +
                    glob.glob(os.path.join(folder, 'ks.jsonl'))):
        for line in open(f):
            rows.append(json.loads(line))
    r = pd.DataFrame(rows)
    r = r.drop_duplicates(['key', 'method'], keep='last')
    r = r[r.dataset.isin(list(DATASETS))]
    return r


def paired(r):
    """Per data set and method: mean over the halves of the difference to
    kded (log-likelihood, nats per point; L2 score in % of |L2 of kded|),
    its standard error and the number of valid halves."""
    ok = r[r.status == 'ok']
    base = ok[ok.method == 'kded'].set_index('key')
    out = []
    for (name, m), g in r.groupby(['dataset', 'method']):
        gg = g[g.status == 'ok'].set_index('key')
        keys = gg.index.intersection(base.index)
        dl = gg.loc[keys, 'loglik'] - base.loc[keys, 'loglik']
        dq = 100 * (gg.loc[keys, 'l2'] - base.loc[keys, 'l2']) / base.loc[keys, 'l2'].abs()
        n = len(keys)
        status = g.status.iloc[0] if n == 0 else 'ok'
        out.append(dict(dataset=name, method=m, n=n, n_all=len(g),
                        loglik=gg.loglik.mean() if n else np.nan,
                        dll=dl.mean() if n else np.nan,
                        se_dll=dl.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                        l2=gg.l2.mean() if n else np.nan,
                        dl2=dq.mean() if n else np.nan,
                        se_dl2=dq.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                        status=status))
    return pd.DataFrame(out)


def time_stats(r):
    """Per data set and method: median and largest selection time and the
    number of selections stopped at the time limit (recorded at the limit)."""
    out = []
    for (name, m), g in r.groupby(['dataset', 'method']):
        t = g.time.dropna() if 'time' in g else pd.Series(dtype=float)
        out.append(dict(dataset=name, method=m,
                        time=t.median() if len(t) else np.nan,
                        time_max=t.max() if len(t) else np.nan,
                        timeouts=int((g.status == 'timeout').sum())))
    return pd.DataFrame(out)


def fmt_time(t):
    if not np.isfinite(t):
        return '–'
    if t < 0.01:
        return '<0.01'
    if t < 10:
        return f'{t:.2f}'
    return f'{t:.0f}'


def tables(P):
    names = [n for n in DATASETS if n in set(P.dataset)]
    methods = [m for m in ORDER if m in set(P.method)]
    md = []
    hdr = '| | ' + ' | '.join(names) + ' |\n|---|' + '---:|' * len(names)
    for title, col, f in (
            ('Held-out log-likelihood of kded (nats per point), and difference '
             'method − kded (positive: better than kded)', 'dll',
             lambda v: f'{v:+.3f}' if abs(v) < 10 else f'{v:+.3g}'),
            ('Held-out L2 score: difference method − kded in % of |L2(kded)| '
             '(negative: better than kded)', 'dl2', lambda v: f'{v:+.1f}'),
            ('Median selection time (s, one thread)', 'time', fmt_time)):
        md.append(f'### {title}\n\n{hdr}')
        if col == 'dll':
            v = [P[(P.dataset == n) & (P.method == 'kded')].loglik for n in names]
            md.append('| kded (absolute) | ' + ' | '.join(
                f'{x.iloc[0]:.3f}' if len(x) else '–' for x in v) + ' |')
        for m in methods:
            cells = []
            for n in names:
                x = P[(P.dataset == n) & (P.method == m)]
                if not len(x):
                    cells.append('')
                elif col != 'time' and x.n.iloc[0] == 0:
                    cells.append(x.status.iloc[0].split(':')[0][:12])
                else:
                    v = x[col].iloc[0]
                    if col == 'time':
                        s = fmt_time(v)
                        if x.timeouts.iloc[0] and v >= x.time_max.iloc[0]:
                            s = '> ' + s        # most selections hit the limit
                    elif m == 'kded':
                        s = '0'
                    else:
                        s = f(v)
                    if col != 'time' and 0 < x.n.iloc[0] < x.n_all.iloc[0]:
                        s += f' ({x.n.iloc[0]}/{x.n_all.iloc[0]})'
                    cells.append(s)
            md.append(f'| {LABEL.get(m, m)} | ' + ' | '.join(cells) + ' |')
        md.append('')
    return '\n'.join(md)


def ranks(P):
    """Normalized rank (0 = best, 1 = worst) of each method on each data set
    on which it ran. The ranks use the differences to kded on the same
    halves, so that a method scored on part of the halves is compared
    fairly."""
    out = {}
    for key, col, asc in (('loglik', 'dll', False), ('l2', 'dl2', True)):
        R = []
        for n, g in P[P.n > 0].groupby('dataset'):
            rk = g.set_index('method')[col].rank(ascending=asc)
            R.append((rk - 1) / max(len(g) - 1, 1))
        out[key] = pd.concat(R, axis=1)
    return out


def method_summary(P):
    """One line per method for the main README: data sets run (* when only
    part of the halves of some data set finished), average normalized rank
    for both scores, median (over the data sets) and largest selection time."""
    rk = ranks(P)
    lines = ['| Selector | data sets | log-likelihood: mean rank | '
             'L2 score: mean rank | median time (s) | largest time (s) |',
             '|---|---:|---:|---:|---:|---:|']
    for m in ORDER:
        g = P[(P.method == m)]
        if not len(g):
            continue
        ok = g[g.n > 0]
        a = rk['loglik'].loc[m] if m in rk['loglik'].index else pd.Series(dtype=float)
        b = rk['l2'].loc[m] if m in rk['l2'].index else pd.Series(dtype=float)
        failed = len(g) - len(ok)
        runs = f'{len(ok)}' + ('*' if (ok.n < ok.n_all).any() else '') + \
            (f' (+{failed} ×)' if failed else '')
        t = g.time.dropna()
        tmax = g.time_max.max()
        largest = fmt_time(tmax) if np.isfinite(tmax) else '–'
        if g.timeouts.sum():
            largest = '> ' + largest           # a selection hit the time limit
        name = f'**{LABEL[m]}**' if m.startswith('kded') else LABEL.get(m, m)
        lines.append(
            f'| {name} | {runs} | {a.mean():.2f} | {b.mean():.2f} | '
            f'{fmt_time(t.median()) if len(t) else "–"} | {largest} |')
    return '\n'.join(lines)


def heatmap(P, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    names = [n for n in DATASETS if n in set(P.dataset)]
    methods = [m for m in ORDER if m in set(P.method) and m != 'kded']
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(methods) + 2.2))
    for ax, col, title, scale in (
            (axes[0], 'dll', 'held-out log-likelihood: method − kded (nats/point)', 0.3),
            (axes[1], 'dl2', 'held-out L2 score: method − kded (% of |L2|)', 30)):
        A = np.full((len(methods), len(names)), np.nan)
        for i, m in enumerate(methods):
            for j, n in enumerate(names):
                x = P[(P.dataset == n) & (P.method == m)]
                if len(x) and x.n.iloc[0] > 0:
                    A[i, j] = x[col].iloc[0]
        sign = 1 if col == 'dll' else -1          # green = better than kded
        im = ax.imshow(np.clip(sign * A, -scale, scale), cmap='RdYlGn',
                       norm=TwoSlopeNorm(0, -scale, scale), aspect='auto')
        for i in range(len(methods)):
            for j in range(len(names)):
                v = A[i, j]
                if np.isfinite(v):
                    if col == 'dll':
                        s = f'{v:+.2f}' if abs(v) < 10 else ('< −10' if v < 0 else '> +10')
                    else:
                        s = f'{v:+.0f}' if abs(v) < 1000 else ('> +999' if v > 0 else '< −999')
                    x = P[(P.dataset == names[j]) & (P.method == methods[i])]
                    if x.n.iloc[0] < x.n_all.iloc[0]:
                        s += '*'                 # only part of the halves
                    ax.text(j, i, s, ha='center', va='center', fontsize=6.5)
                else:
                    x = P[(P.dataset == names[j]) & (P.method == methods[i])]
                    if len(x):
                        ax.text(j, i, '×', ha='center', va='center', fontsize=8,
                                color='0.4')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([f'{n} ({DATASETS[n][0]}D)' for n in names],
                           rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([SHORT.get(m, m) for m in methods], fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.text(0.5, 0.005, 'green: better than kded, red: worse; * only part of the '
             'halves finished within the time limit; × none finished; blank: not '
             'applicable', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=130)


def timing_figure(folder, path):
    rows = []
    for f in glob.glob(os.path.join(folder, 'timing*.jsonl')):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        return None
    T = pd.DataFrame(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lab = {'kded': 'kded (ISJ, diffusion)', 'ks_hpi': 'R ks Hpi + kde (binned)',
           'kdepy': 'KDEpy FFTKDE (ISJ in 1-D, Scott for d > 1)',
           'scipy': 'scipy gaussian_kde (Scott)',
           'sklearn': 'scikit-learn KernelDensity (Scott)',
           'sm_nr': 'statsmodels normal_reference',
           'sm_cv_ml': 'statsmodels cv_ml (selection only)'}
    sty = {'kded': dict(color='C0', lw=2.4, marker='o'), 'kdepy': dict(color='C1', marker='s'),
           'scipy': dict(color='C3', marker='^'), 'sklearn': dict(color='C4', marker='v'),
           'sm_nr': dict(color='C5', marker='D'), 'sm_cv_ml': dict(color='C7', marker='x', ls='--'),
           'ks_hpi': dict(color='C2', marker='P')}
    dims = sorted(T.d.unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(4.6 * len(dims), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    grid = {1: 'grid of 1024 points', 2: '128² grid', 3: '64³ grid'}
    tmax = T['timeout'].max() if 'timeout' in T and T['timeout'].notna().any() else None
    for ax, d in zip(axes, dims):
        for m in lab:
            g = T[(T.d == d) & (T.method == m) & (T.status == 'ok')].sort_values('N')
            if len(g):
                ax.plot(g.N, g.time, label=lab[m], ms=5, **sty[m])
        if tmax:
            ax.axhline(tmax, color='0.4', ls=':', lw=1)
            ax.text(1.1e3, tmax * 1.15, f'time limit ({tmax:.0f} s)', fontsize=7.5, color='0.3')
        ax.set(xscale='log', yscale='log', xlabel='sample size N',
               title=f'd = {d}: selection + density on a {grid.get(d, "grid")}')
        ax.grid(alpha=0.3, which='both')
    axes[0].set_ylabel('seconds (one thread)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=130)
    return T


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    res = os.path.join(HERE, 'results')
    ap.add_argument('--scores', default=os.path.join(res, 'cluster'))
    ap.add_argument('--times', default=os.path.join(res, 'local'))
    ap.add_argument('--timing', default=os.path.join(res, 'timing'))
    ap.add_argument('--figures', default=os.path.join(HERE, '..', 'docs', 'images'))
    ap.add_argument('--out', default=res)
    a = ap.parse_args()
    os.makedirs(a.figures, exist_ok=True)
    r = load(a.scores)
    if not glob.glob(os.path.join(a.times, 'python*.jsonl')):
        a.times = a.scores
    rt = load(a.times)
    P = paired(r).merge(time_stats(rt), on=['dataset', 'method'], how='left')
    P.to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    src = (f'Scores from `{os.path.relpath(a.scores, a.out)}`, selection times from '
           f'`{os.path.relpath(a.times or a.scores, a.out)}`.')
    md = ['# Held-out benchmark on open data\n',
          'Mean over 10 held-out fits (5 random halvings x 2) per data set; see '
          '`run_benchmark.py`. "(k/10)": only k fits finished within the time '
          'limit. ' + src + '\n', tables(P)]
    rk = ranks(P)
    md.append('### Average normalized rank over the data sets (0 = best, 1 = worst)\n')
    md.append('| method | log-likelihood | L2 score | data sets |\n|---|---:|---:|---:|')
    for m in ORDER:
        if m in rk['loglik'].index:
            x = rk['loglik'].loc[m]
            y = rk['l2'].loc[m]
            md.append(f'| {LABEL.get(m, m)} | {x.mean():.2f} | {y.mean():.2f} | {x.notna().sum()} |')
    open(os.path.join(a.out, 'summary.md'), 'w').write('\n'.join(md) + '\n')
    open(os.path.join(a.out, 'readme_table.md'), 'w').write(method_summary(P) + '\n')
    heatmap(P, os.path.join(a.figures, 'validation_heldout.png'))
    timing_figure(a.timing, os.path.join(a.figures, 'validation_timing.png'))
    print('\n'.join(md))
