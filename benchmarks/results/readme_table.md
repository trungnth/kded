| Selector | data sets | log-likelihood: mean rank | L2 score: mean rank | median time (s) | largest time (s) |
|---|---:|---:|---:|---:|---:|
| **kded** | 14 | 0.38 | 0.22 | 0.16 | 19 |
| **kded, rotate=True** | 9 | 0.56 | 0.37 | 4.15 | 21 |
| **kded adaptive (experimental)** | 14 | 0.29 | 0.23 | 7.26 | 51 |
| R ks Hpi (plug-in) | 14 | 0.48 | 0.38 | 0.07 | 9.85 |
| R ks Hscv (smoothed CV) | 14 | 0.40 | 0.38 | 0.81 | 210 |
| R ks Hlscv (least squares CV) | 14 | 0.70 | 0.58 | 0.76 | 191 |
| R bw.SJ (Sheather-Jones) | 5 | 0.32 | 0.05 | <0.01 | <0.01 |
| statsmodels cv_ml | 14* | 0.24 | 0.35 | 8.64 | > 600 |
| scikit-learn GridSearchCV | 14 | 0.24 | 0.45 | 1.15 | 244 |
| KDEpy ISJ | 5 | 0.75 | 0.68 | <0.01 | 0.02 |
| scipy gaussian_kde (Scott) | 14 | 0.71 | 0.79 | <0.01 | 0.01 |
| scipy gaussian_kde (Silverman) | 14 | 0.72 | 0.80 | <0.01 | 0.02 |
| statsmodels normal_reference | 14 | 0.75 | 0.90 | <0.01 | 0.02 |
| KDEpy Silverman | 5 | 0.60 | 0.82 | <0.01 | <0.01 |
