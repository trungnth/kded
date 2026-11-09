# Held-out benchmark on open data

Mean over 10 held-out fits (5 random halvings x 2) per data set; see `run_benchmark.py`. "(k/10)": only k fits finished within the time limit. Scores from `cluster`, selection times from `local`.

### Held-out log-likelihood of kded (nats per point), and difference method − kded (positive: better than kded)

| | faithful_waiting | faithful_eruptions | penguins_flipper | hsct12_cd45 | air_co2 | faithful | penguins_bill | quake_map | penguins | quake | hsct6 | hsct12 | air | penguins4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kded (absolute) | -3.836 | -1.034 | -3.952 | -5.599 | -6.515 | -4.232 | -4.895 | -10.113 | -8.137 | -14.576 | -17.909 | -17.620 | -15.913 | -15.464 |
| kded | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kded, rotate=True |  |  |  |  |  | -0.040 | -0.019 | -0.012 | -0.087 | +0.009 | -0.009 | +0.009 | -0.001 | -0.192 |
| kded adaptive (experimental) | -0.015 | -0.005 | -0.012 | +0.023 | +0.247 | -0.017 | -0.009 | -0.398 | -0.003 | -0.152 | +0.192 | +0.468 | +0.297 | +0.049 |
| R ks Hpi (plug-in) | +0.001 | -0.011 | +0.001 | -0.001 | +0.064 | -0.041 | +0.002 | +0.774 | -0.191 | +0.196 | -0.402 | +0.100 | -1.390 | -0.357 |
| R ks Hscv (smoothed CV) | +0.001 | -0.009 | +0.001 | +0.005 | +0.045 | -0.041 | -0.005 | +0.745 | -0.052 | +0.107 | -0.026 | +0.199 | -0.543 | -0.012 |
| R ks Hlscv (least squares CV) | +0.000 | -0.003 | -0.003 | -0.006 | -0.045 | -0.117 | -0.006 | -3.58e+06 | -0.102 | -6.14e+06 | -4.48e+07 | -1.75e+06 | -4.15e+08 | -0.070 |
| R bw.SJ (Sheather-Jones) | +0.002 | +0.000 | +0.003 | -0.001 | +0.028 |  |  |  |  |  |  |  |  |  |
| statsmodels cv_ml | -0.001 | +0.003 | +0.005 | +0.009 | -0.403 (5/10) | -0.003 | -0.011 | +1.171 | -0.010 | +0.461 | +0.103 | +0.306 | +0.222 (4/10) | +0.012 |
| scikit-learn GridSearchCV | -0.001 | +0.004 | +0.005 | +0.011 | -0.242 | -0.021 | -0.000 | +1.091 | -0.004 | +0.196 | +0.105 | +0.283 | +0.153 | +0.028 |
| KDEpy ISJ | -0.006 | -0.244 | +0.004 | -0.083 | -0.764 |  |  |  |  |  |  |  |  |  |
| scipy gaussian_kde (Scott) | -0.033 | -0.157 | -0.018 | -0.059 | +0.198 | -0.203 | -0.079 | +0.245 | -0.276 | -0.498 | -0.205 | -0.611 | +0.182 | -0.307 |
| scipy gaussian_kde (Silverman) | -0.041 | -0.178 | -0.022 | -0.071 | +0.202 | -0.203 | -0.079 | +0.245 | -0.261 | -0.462 | -0.188 | -0.573 | +0.179 | -0.278 |
| statsmodels normal_reference | -0.040 | -0.177 | -0.022 | -0.072 | +0.202 | -0.348 | -0.073 | +0.211 | -0.270 | -0.560 | -0.185 | -0.707 | +0.178 | -0.490 |
| KDEpy Silverman | -0.041 | -0.178 | -0.022 | +0.011 | +0.202 |  |  |  |  |  |  |  |  |  |

### Held-out L2 score: difference method − kded in % of |L2(kded)| (negative: better than kded)

| | faithful_waiting | faithful_eruptions | penguins_flipper | hsct12_cd45 | air_co2 | faithful | penguins_bill | quake_map | penguins | quake | hsct6 | hsct12 | air | penguins4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kded | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kded, rotate=True |  |  |  |  |  | +5.2 | +0.8 | -0.1 | +5.0 | +2.0 | +0.2 | +1.6 | +0.0 | +15.9 |
| kded adaptive (experimental) | -0.2 | +2.3 | +0.0 | +0.0 | +0.1 | +0.1 | +0.9 | -35.1 | +0.6 | -55.9 | -0.2 | -3.3 | -0.1 | -8.5 |
| R ks Hpi (plug-in) | -0.1 | -0.3 | -0.1 | -0.0 | -0.0 | +5.3 | +0.3 | +68.6 | +5.9 | +57.6 | +3.0 | +26.2 | +7.8 | +13.2 |
| R ks Hscv (smoothed CV) | -0.1 | -0.4 | -0.1 | -0.0 | -0.0 | +5.0 | +1.2 | +69.6 | +1.8 | +65.8 | +3.8 | +32.1 | +3.7 | -2.4 |
| R ks Hlscv (least squares CV) | -0.0 | +1.4 | +0.5 | +0.0 | +0.0 | +4.4 | -0.3 | +3751.1 | +2.6 | +5663.5 | +11468.6 | +3637.4 | +18724.1 | -8.6 |
| R bw.SJ (Sheather-Jones) | -0.2 | -0.6 | -0.2 | -0.0 | -0.0 |  |  |  |  |  |  |  |  |  |
| statsmodels cv_ml | +0.1 | +0.2 | +0.1 | +0.3 | +0.1 (5/10) | -0.2 | +0.4 | +34.7 | -0.2 | +37.1 | +3.7 | +13.7 | +3.4 (4/10) | -6.7 |
| scikit-learn GridSearchCV | +0.2 | -0.1 | -0.1 | +0.4 | +1.0 | +1.3 | +0.1 | +43.1 | -0.4 | +58.8 | +4.5 | +20.5 | +9.3 | -6.9 |
| KDEpy ISJ | +0.8 | +13.1 | +0.3 | +0.1 | +0.1 |  |  |  |  |  |  |  |  |  |
| scipy gaussian_kde (Scott) | +3.0 | +10.3 | +1.4 | +9.2 | +0.4 | +19.5 | +8.3 | +81.9 | +23.1 | +85.2 | +28.9 | +76.3 | +7.0 | +25.7 |
| scipy gaussian_kde (Silverman) | +3.8 | +12.0 | +1.9 | +10.5 | +0.5 | +19.5 | +8.3 | +81.9 | +21.4 | +84.5 | +27.3 | +75.2 | +6.4 | +22.4 |
| statsmodels normal_reference | +3.7 | +11.9 | +1.9 | +10.5 | +0.5 | +27.7 | +7.8 | +82.5 | +25.7 | +86.3 | +33.4 | +79.4 | +11.0 | +42.1 |
| KDEpy Silverman | +3.8 | +12.0 | +1.9 | +0.0 | +0.5 |  |  |  |  |  |  |  |  |  |

### Median selection time (s, one thread)

| | faithful_waiting | faithful_eruptions | penguins_flipper | hsct12_cd45 | air_co2 | faithful | penguins_bill | quake_map | penguins | quake | hsct6 | hsct12 | air | penguins4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kded | 0.03 | 0.03 | 0.03 | 0.05 | 0.04 | 0.03 | 0.03 | 0.27 | 0.56 | 12 | 2.73 | 7.13 | 9.24 | 4.02 |
| kded, rotate=True |  |  |  |  |  | 0.03 | 0.03 | 0.27 | 0.59 | 13 | 5.66 | 5.17 | 9.28 | 4.15 |
| kded adaptive (experimental) | 0.06 | 0.06 | 0.06 | 13 | 11 | 0.07 | 0.08 | 5.38 | 2.15 | 42 | 25 | 45 | 37 | 9.15 |
| R ks Hpi (plug-in) | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | 0.07 | 0.08 | 1.15 | 0.06 | 7.78 | 7.14 | 7.47 | 3.94 | 0.20 |
| R ks Hscv (smoothed CV) | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | 0.81 | 0.77 | 3.63 | 0.81 | 48 | 63 | 82 | 34 | 2.00 |
| R ks Hlscv (least squares CV) | <0.01 | <0.01 | <0.01 | 0.48 | 3.08 | 0.17 | 0.16 | 28 | 0.71 | 180 | 60 | 62 | 85 | 0.81 |
| R bw.SJ (Sheather-Jones) | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |  |  |  |  |  |  |  |  |  |
| statsmodels cv_ml | 0.07 | 0.07 | 0.08 | 17 | 384 | 0.26 | 0.28 | 16 | 0.70 | 28 | 153 | 130 | > 600 | 1.57 |
| scikit-learn GridSearchCV | 0.17 | 0.17 | 0.17 | 24 | 174 | 0.18 | 0.18 | 2.10 | 0.20 | 2.23 | 21 | 8.99 | 169 | 0.20 |
| KDEpy ISJ | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |  |  |  |  |  |  |  |  |  |
| scipy gaussian_kde (Scott) | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |
| scipy gaussian_kde (Silverman) | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |
| statsmodels normal_reference | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |
| KDEpy Silverman | <0.01 | <0.01 | <0.01 | <0.01 | <0.01 |  |  |  |  |  |  |  |  |  |

### Average normalized rank over the data sets (0 = best, 1 = worst)

| method | log-likelihood | L2 score | data sets |
|---|---:|---:|---:|
| kded | 0.38 | 0.22 | 14 |
| kded, rotate=True | 0.56 | 0.37 | 9 |
| kded adaptive (experimental) | 0.29 | 0.23 | 14 |
| R ks Hpi (plug-in) | 0.48 | 0.38 | 14 |
| R ks Hscv (smoothed CV) | 0.40 | 0.38 | 14 |
| R ks Hlscv (least squares CV) | 0.70 | 0.58 | 14 |
| R bw.SJ (Sheather-Jones) | 0.32 | 0.05 | 5 |
| statsmodels cv_ml | 0.24 | 0.35 | 14 |
| scikit-learn GridSearchCV | 0.24 | 0.45 | 14 |
| KDEpy ISJ | 0.75 | 0.68 | 5 |
| scipy gaussian_kde (Scott) | 0.71 | 0.79 | 14 |
| scipy gaussian_kde (Silverman) | 0.72 | 0.80 | 14 |
| statsmodels normal_reference | 0.75 | 0.90 | 14 |
| KDEpy Silverman | 0.60 | 0.82 | 5 |
