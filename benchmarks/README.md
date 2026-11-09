# Benchmarks

Scripts and raw results behind the [Validation](../README.md#validation)
section of the main README. They compare the bandwidth selection of `kded`
with other KDE implementations on open data (held-out goodness of fit) and
measure the computing time.

| File | Purpose |
|---|---|
| `datasets.py` | the 14 open data sets (1-D to 4-D) and their random halvings |
| `bw_selectors.py` | the selectors, called through each library's public API |
| `run_benchmark.py` | fits every selector on one half, scores it on the other half |
| `ks_selectors.R` | the selectors of the R package `ks` (and `bw.SJ`) on the same halves |
| `timing.py`, `timing_ks.R` | selection + density on a grid, for N = 10³ to 10⁶ |
| `report.py` | tables (`results/summary.md`) and figures (`docs/images/validation_*.png`) |
| `export_ks_data.R` | writes the `ks` data sets (quake, hsct, air) to `data/` |
| `results/` | the raw results used in the README (see below) |

## Results

| Folder | Content |
|---|---|
| `results/cluster/` | held-out benchmark run on the JINR HybriLIT cluster (Intel Xeon Phi 7290 nodes), time limit 12 hours per selection: the scores in the README |
| `results/local/` | the same benchmark on one 4-core machine (Intel Xeon, 2.1 GHz), time limit 10 minutes per selection: the selection times in the README |
| `results/timing/` | the timing study, on the same 4-core machine |
| `results/summary.md` | tables per data set (`python report.py`) |

The two runs of the held-out benchmark selected the same bandwidth
matrices (to within 2·10⁻⁴, relative), so they gave the same scores; with
its longer limit the cluster run also finished part of the statsmodels
`cv_ml` selections on the air quality data (5 of the 10 halves of
`air_co2`, 4 of `air`). The times are taken from the 4-core machine because
on the cluster's Xeon Phi nodes the selections were 8 to 60 times slower,
by a factor that depends on the library. To redo the tables and figures of
the README: `python report.py` (scores from `results/cluster`, times from
`results/local`, timing study from `results/timing`).

## Reproducing

```bash
pip install -e "..[benchmark]"           # kded + scipy, scikit-learn, statsmodels, KDEpy
Rscript export_ks_data.R                 # needs R with the ks package
R=results/myrun                          # a new folder for your run
python run_benchmark.py --out $R/python.jsonl
python run_benchmark.py --export-splits splits
Rscript ks_selectors.R splits $R/ks_H.csv
python run_benchmark.py --score-ks $R/ks_H.csv --out $R/ks.jsonl
python timing.py --out $R/timing.jsonl; Rscript timing_ks.R $R/timing_ks.jsonl
python report.py --scores $R --times $R --timing $R --out $R --figures $R
```

Every fit runs in its own process on one thread with a time limit
(`--timeout`, default 600 s). A selector that hits the limit on one half of
a data set is skipped on its other halves. Results are appended to
JSON-lines files and the fits already in a file are skipped when a script
is run again, so an interrupted run resumes where it stopped. For the same
reason, write a new run to a new folder: the default output folders
(`results/local`, `results/timing`) hold the published results.

## Design

- **Scores.** Each selector returns a bandwidth matrix H; the exact Gaussian
  KDE with this H (or, for the adaptive estimator, with the per-point
  matrices) is scored on the held-out half by the mean log-likelihood
  (higher is better) and by the L2 score ∫f̂² − 2·mean f̂(Y) (lower is
  better; it estimates the integrated squared error up to a constant).
  So the comparison is about bandwidth selection, not about grids or
  interpolation.
- **Splits.** Five random halvings per data set, both halves used for
  fitting: ten scores per selector and data set. The hourly air data are
  split by ISO week.
- **Rounded values** are jittered uniformly over their rounding interval
  (otherwise ties act as point masses and every plug-in selector picks a
  bandwidth at the rounding scale). Flow cytometry events at 0 (censored)
  and earthquakes at the catalogue default depths 10 and 33 km are removed.
- **Selectors.** `kded` (default, `rotate=True`, adaptive), scipy
  `gaussian_kde` (Scott, Silverman), statsmodels `KDEMultivariate`
  (`normal_reference`, `cv_ml`), scikit-learn `KernelDensity` with
  `GridSearchCV` (5-fold, 40 bandwidths, standardized data), KDEpy
  `FFTKDE` (ISJ and Silverman, 1-D only), R `ks` (`Hpi`, `Hscv`, `Hlscv`
  with their defaults; `hpi`, `hscv`, `hlscv` in 1-D) and R `bw.SJ`.
