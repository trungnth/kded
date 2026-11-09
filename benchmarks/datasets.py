"""Open data sets of the validation benchmark and their train/test splits.

Every data set is split at random into two halves, five times (the air data
by ISO week, so that hourly readings of one week stay together). A selector
is run on one half and scored on the other, and the halves are then swapped:
ten scores per selector and data set.

Values recorded with a finite resolution (integer counts, minutes, mm) are
jittered by a uniform amount of half the resolution, so that ties do not
create point masses. (The earthquake depths are not used as a 1-D data set:
44 % of them are multiples of 5 km, a rounding that a 1-km jitter does not
remove.) Fixed seeds make the data and the splits reproducible.

Sources
-------
faithful   Old Faithful geyser, 272 eruptions: eruption time and waiting time
           (R `datasets::faithful`, from Haerdle 1991); bundled in
           data/faithful.csv.
penguins   Palmer penguins, 342 birds (Horst, Hill and Gorman 2020, CC0);
           bundled in data/penguins.csv.
quake      severe earthquakes worldwide, years 100-2016: longitude,
           latitude, depth (R package `ks`, data `quake`, from NOAA NGDC;
           the 2,978 events with a known depth).
hsct       Flow cytometry of hematopoietic stem cell transplants, 4 mice
           (R package `ks`, data `hsct`).
air        hourly NO2, PM10 and CO2 in the Chatelet underground station of
           the Paris metro, 2013-2016 (R package `ks`, data `air`, from RATP).
The ks data are licensed with the R package (GPL) and are not redistributed
here: run `Rscript export_ks_data.R` once to write data/quake3.csv,
data/hsct.csv and data/air.csv.
"""
import os
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
SPLITS = 5


def _jitter(X, res, rng):
    res = np.broadcast_to(np.asarray(res, dtype=float), X.shape[1:])
    return X + rng.uniform(-0.5, 0.5, X.shape) * res


def _faithful():
    return pd.read_csv(os.path.join(DATA, 'faithful.csv'))


def _penguins():
    return pd.read_csv(os.path.join(DATA, 'penguins.csv'))


def _quake():
    return pd.read_csv(os.path.join(DATA, 'quake3.csv'))


def _hsct(subject):
    h = pd.read_csv(os.path.join(DATA, 'hsct.csv'))
    X = h.loc[h.subject == subject,
              ['FITC.CD45.1', 'PE.Ly65Mac1', 'APC.CD45.2']].to_numpy(float)
    return X[np.all(X > 0, axis=1)]        # drop events censored at 0


def _hsct_marker(subject, marker):
    h = pd.read_csv(os.path.join(DATA, 'hsct.csv'))
    x = h.loc[h.subject == subject, [marker]].to_numpy(float)
    return x[x[:, 0] > 0]


def _air():
    a = pd.read_csv(os.path.join(DATA, 'air.csv')).dropna(
        subset=['no2', 'pm10', 'co2'])
    week = pd.to_datetime(a['date']).dt.isocalendar()
    blocks = week.year.to_numpy() * 100 + week.week.to_numpy()
    return a[['no2', 'pm10', 'co2']].to_numpy(float), blocks


def _quake_clean(q):
    z = q['depth'].to_numpy()
    return q[(z != 10) & (z != 33)]         # catalogue default depths


# name: (dimension, loader returning (X, blocks), description)
def _make():
    D = {}
    rng = lambda k: np.random.default_rng(1000 + k)
    sec = 1 / 60                            # eruption times: minutes, 1 s resolution

    D['faithful_waiting'] = (1, lambda: (_jitter(
        _faithful()[['waiting']].to_numpy(float), 1, rng(1)), None),
        'Old Faithful, waiting time (min)')
    D['faithful_eruptions'] = (1, lambda: (_jitter(
        _faithful()[['eruptions']].to_numpy(float), sec, rng(2)), None),
        'Old Faithful, eruption time (min)')
    D['penguins_flipper'] = (1, lambda: (_jitter(
        _penguins()[['flipper_length_mm']].to_numpy(float), 1, rng(3)), None),
        'Penguins, flipper length (mm)')
    D['hsct12_cd45'] = (1, lambda: (_jitter(_hsct_marker(12, 'APC.CD45.2'), 1, rng(4)), None),
                        'Flow cytometry, mouse 12, APC CD45.2 marker')
    D['air_co2'] = (1, lambda: (lambda X, b: (_jitter(X[:, 2:], 1, rng(5)), b))(*_air()),
                    'Paris metro station air, CO2, hourly')

    D['faithful'] = (2, lambda: (_jitter(
        _faithful()[['eruptions', 'waiting']].to_numpy(float), [sec, 1], rng(11)), None),
        'Old Faithful (eruption, waiting)')
    D['penguins_bill'] = (2, lambda: (_jitter(
        _penguins()[['bill_length_mm', 'bill_depth_mm']].to_numpy(float), 0.1, rng(12)), None),
        'Penguins (bill length, bill depth)')
    D['quake_map'] = (2, lambda: (_quake()[['long', 'lat']].to_numpy(float), None),
                      'Severe earthquakes (longitude, latitude)')

    D['penguins'] = (3, lambda: (_jitter(
        _penguins()[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm']].to_numpy(float),
        [0.1, 0.1, 1], rng(21)), None),
        'Penguins (bill length, bill depth, flipper)')
    D['quake'] = (3, lambda: (_jitter(
        _quake_clean(_quake())[['long', 'lat', 'depth']].to_numpy(float),
        [0, 0, 1], rng(22)), None),
        'Severe earthquakes (longitude, latitude, depth)')
    D['hsct6'] = (3, lambda: (_jitter(_hsct(6), 1, rng(23)), None),
                  'Flow cytometry, mouse 6 (3 markers)')
    D['hsct12'] = (3, lambda: (_jitter(_hsct(12), 1, rng(24)), None),
                   'Flow cytometry, mouse 12 (3 markers)')
    D['air'] = (3, lambda: (lambda X, b: (_jitter(X, 1, rng(25)), b))(*_air()),
                'Paris metro station air (NO2, PM10, CO2), hourly')

    D['penguins4'] = (4, lambda: (_jitter(
        _penguins()[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
                     'body_mass_g']].to_numpy(float), [0.1, 0.1, 1, 25], rng(31)), None),
        'Penguins (bill length, bill depth, flipper, body mass)')
    return D


DATASETS = _make()


def load(name):
    """Returns (X, blocks) of a data set; X is an N x d array."""
    return DATASETS[name][1]()


def splits(name, n_splits=SPLITS):
    """Yields (key, train, test) for the 2 * n_splits held-out fits."""
    X, blocks = load(name)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    for r in range(n_splits):
        if blocks is None:
            m = rng.permutation(len(X)) < len(X) // 2
        else:
            ub = np.unique(blocks)
            pick = ub[rng.permutation(len(ub)) < len(ub) // 2]
            m = np.isin(blocks, pick)
        yield f'{name}/{r}a', X[m], X[~m]
        yield f'{name}/{r}b', X[~m], X[m]


def available():
    """Data sets whose files are present."""
    out = []
    for name in DATASETS:
        try:
            load(name)
            out.append(name)
        except FileNotFoundError:
            pass
    return out


if __name__ == '__main__':
    for name in DATASETS:
        try:
            X, _ = load(name)
            print(f'{name:20s} d={X.shape[1]}  N={len(X):6d}  {DATASETS[name][2]}')
        except FileNotFoundError:
            print(f'{name:20s} missing (run Rscript export_ks_data.R)')
